"""Video rendering step implementation for Shorts Studio.

Composes vertical short videos (1080x1920, 30 fps, H.264 / AAC) by animating
each storyboard scene's vertical image according to its timing metadata and
muxing the complete continuous narration audio WAV using FFmpeg.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
import uuid

logger = logging.getLogger(__name__)

from sqlalchemy.orm import object_session

from app.animation import (
    DEFAULT_SUBJECT_Y,
    build_ffmpeg_zoompan_filter,
    plan_scene_animation,
)
from app.animation.crop import (
    QUESTIONABLE_REUSE_MAX_SCALE,
    SAFE_REUSE_MAX_SCALE,
    calculate_portrait_crop,
    evaluate_beat_crop_feasibility,
    plan_beat_camera_motion,
)
from app.config import settings
from app.models import Video, VideoStep
from app.subtitles import (
    SubtitleError,
    compute_aligned_scene_timings,
    get_subtitle_provider,
    load_or_compute_alignment,
)


class PipelineError(Exception):
    """Raised when preconditions or render pipeline execution fails."""


def _probe_video_file(file_path: Path) -> dict[str, Any]:
    """Inspect a rendered MP4 video using ffprobe and return parsed metadata."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate,channels,sample_rate,pix_fmt",
        "-show_entries", "format=duration,size,bit_rate",
        "-of", "json",
        str(file_path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=30.0,
        )
        return json.loads(proc.stdout)
    except subprocess.CalledProcessError as exc:
        raise PipelineError(f"ffprobe inspection failed for '{file_path}': {exc.stderr.strip()}") from exc
    except Exception as exc:
        raise PipelineError(f"Failed to inspect rendered video with ffprobe: {exc}") from exc


def _resolve_aligned_bounds(
    video: Video,
    vo_data: dict[str, Any],
    valid_audio: dict[int, dict[str, Any]],
    all_indices: list[int],
    continuous_audio_abs: Path,
    continuous_duration: float,
    storage_root: Path,
) -> tuple[dict[int, tuple[float, float]] | None, str, str | None, bool]:
    """Resolve per-scene (start, end) intervals from actual narration word timestamps.

    1. Use voiceover items already carrying ``timing_type == "aligned"``.
    2. Otherwise (legacy voiceover output with estimated timing), load the persisted alignment
       or run forced alignment once on the continuous WAV and persist it for reuse.
    3. Return (None, "estimated", ...) when alignment is unavailable.
    """
    inner = vo_data.get("voiceover") if isinstance(vo_data.get("voiceover"), dict) else {}
    items = [valid_audio[idx] for idx in all_indices]

    def _bounds_from(entries: list[dict[str, Any]]) -> dict[int, tuple[float, float]] | None:
        bounds: dict[int, tuple[float, float]] = {}
        prev_end = 0.0
        for idx, it in zip(all_indices, entries):
            try:
                s = float(it["start_time"])
                e = float(it["end_time"])
            except (KeyError, TypeError, ValueError):
                return None
            if e <= s or abs(s - prev_end) > 0.002:
                return None
            bounds[idx] = (s, e)
            prev_end = e
        if abs(prev_end - continuous_duration) > 0.25:
            return None
        return bounds

    alignment_hash = inner.get("alignment_hash") or vo_data.get("alignment_hash")
    if items and all(it.get("timing_type") == "aligned" for it in items):
        bounds = _bounds_from(items)
        if bounds is not None:
            return bounds, "aligned", alignment_hash, True

    # Legacy/estimated voiceover output: try the shared alignment cache (or align once).
    vo_meta = vo_data.get("meta") if isinstance(vo_data.get("meta"), dict) else {}
    if vo_meta.get("provider") == "mock" or vo_data.get("mock"):
        return None, "estimated", None, False
    if not all(str(it.get("narration") or "").strip() for it in items):
        return None, "estimated", None, False

    try:
        aligner = get_subtitle_provider()
    except Exception:
        return None, "estimated", None, False
    if aligner.provider_name == "mock":
        return None, "estimated", None, False

    scenes = []
    for idx, it in zip(all_indices, items):
        entry: dict[str, Any] = {"index": idx, "narration": str(it.get("narration") or "").strip()}
        if it.get("visual_beats"):
            entry["visual_beats"] = it["visual_beats"]
        scenes.append(entry)

    try:
        alignment, alignment_hash, reused = load_or_compute_alignment(
            video_id=video.id,
            storage_root=storage_root,
            sorted_scenes=scenes,
            narration_script=vo_data.get("narration_script") or inner.get("narration_script"),
            continuous_audio_abs=continuous_audio_abs,
            audio_hash=vo_data.get("audio_hash") or inner.get("audio_hash"),
            provider=aligner,
        )
        timings = compute_aligned_scene_timings(scenes, alignment, continuous_duration)
    except SubtitleError:
        return None, "estimated", None, False

    for t in timings:
        t_idx = t.get("index")
        if t_idx in valid_audio and "visual_beats" in t:
            valid_audio[t_idx]["visual_beats"] = t["visual_beats"]

    bounds = {t["index"]: (float(t["start_time"]), float(t["end_time"])) for t in timings}
    return bounds, "aligned", alignment_hash, reused


def _resolve_beat_image(
    beat: dict[str, Any],
    scene_idx: int,
    valid_images: dict[int, dict[str, Any]],
    img_items: list[dict[str, Any]],
    storage_root: Path,
) -> tuple[Path, str, bool]:
    """Resolve the source image file for an individual visual beat.

    Attempts to resolve an explicitly assigned ``source_image_id`` first (by scene index,
    image item ID, or storage file path). Falls back to the parent scene's generated image.

    Returns:
        tuple of (image_abs_path, image_rel_path, is_distinct_image_from_parent)
    """
    parent_rel = str(valid_images[scene_idx]["file"])
    parent_abs = (storage_root / parent_rel).resolve()

    source_image_id = beat.get("source_image_id")
    if not source_image_id or not str(source_image_id).strip():
        return parent_abs, parent_rel, False

    sid = str(source_image_id).strip()
    if sid == str(scene_idx):
        return parent_abs, parent_rel, False

    # 1. Match as integer scene index
    try:
        sid_int = int(sid)
        if sid_int in valid_images:
            cand_rel = str(valid_images[sid_int]["file"])
            cand_abs = (storage_root / cand_rel).resolve()
            if cand_abs.is_file():
                return cand_abs, cand_rel, (sid_int != scene_idx)
    except ValueError:
        pass

    # 2. Match in img_items by id, image_id, or file
    for it in img_items:
        if not isinstance(it, dict):
            continue
        if str(it.get("id") or "") == sid or str(it.get("image_id") or "") == sid or str(it.get("file") or "") == sid:
            cand_rel = it.get("file")
            if cand_rel and isinstance(cand_rel, str):
                cand_abs = (storage_root / cand_rel).resolve()
                if cand_abs.is_file():
                    return cand_abs, cand_rel, (cand_rel != parent_rel)

    # 3. Direct path relative to storage root
    cand_abs = (storage_root / sid).resolve()
    if cand_abs.is_file():
        return cand_abs, sid, (cand_abs != parent_abs)

    # Fallback to parent image
    return parent_abs, parent_rel, False


def _resolve_scene_beats(
    scene_idx: int,
    scene_meta: dict[str, Any],
    audio_meta: dict[str, Any],
    scene_start_time: float,
    scene_end_time: float,
) -> tuple[list[dict[str, Any]], str | None]:
    """Extract, validate, and normalize usable visual beats for a scene.

    Normalizes beat timestamps strictly at the whole-scene collection level:
    - If all beats use continuous-timeline coordinates ([scene_start, scene_end]), preserves them.
    - If all beats use intra-scene relative coordinates ([0, scene_dur]), shifts all beats uniformly by scene_start_time.
    - If timing is missing, invalid, non-monotonic, or out-of-bounds, returns ([], fallback_reason).
    - If no beats were provided (normal legacy scene), returns ([], None).

    Returns:
        tuple of (usable_beats_list, fallback_reason)
    """
    raw_beats: list[Any] = audio_meta.get("visual_beats") or scene_meta.get("visual_beats") or []
    if not isinstance(raw_beats, list) or len(raw_beats) == 0:
        return [], None

    # Map storyboard beats by order or id to retrieve visual direction metadata
    sb_beats_by_order: dict[int, dict[str, Any]] = {}
    sb_beats_list = scene_meta.get("visual_beats") or []
    if isinstance(sb_beats_list, list):
        for idx_b, b in enumerate(sb_beats_list, start=1):
            if isinstance(b, dict):
                o = int(b.get("order") or idx_b)
                sb_beats_by_order[o] = b
            else:
                o = int(getattr(b, "order", idx_b) or idx_b)
                sb_beats_by_order[o] = b.model_dump() if hasattr(b, "model_dump") else dict(b)

    parsed_beats: list[dict[str, Any]] = []
    for idx_b, b in enumerate(raw_beats, start=1):
        b_dict = dict(b) if isinstance(b, dict) else (b.model_dump() if hasattr(b, "model_dump") else {})
        order = int(b_dict.get("order") or idx_b)
        sb_meta = sb_beats_by_order.get(order, {})

        # Merge storyboard visual metadata and voiceover timing metadata
        merged = {**sb_meta, **b_dict}
        for v_field in (
            "visual_description",
            "visual_purpose",
            "shot_type",
            "camera_angle",
            "focal_point",
            "visual_technique",
            "image_strategy",
            "source_image_id",
        ):
            if sb_meta.get(v_field) and not merged.get(v_field):
                merged[v_field] = sb_meta[v_field]

        # Timing validation
        s_t = merged.get("start_time")
        e_t = merged.get("end_time")
        if s_t is None or e_t is None:
            return [], "missing_or_invalid_timing"

        try:
            s_val = float(s_t)
            e_val = float(e_t)
        except (TypeError, ValueError):
            return [], "missing_or_invalid_timing"

        if e_val <= s_val:
            return [], "non_monotonic_or_overlapping"

        merged["start_time"] = s_val
        merged["end_time"] = e_val
        merged["duration_seconds"] = round(e_val - s_val, 3)
        merged["order"] = order
        parsed_beats.append(merged)

    # Validate strictly monotonic ordering within their initial coordinate system
    for i in range(len(parsed_beats) - 1):
        if parsed_beats[i + 1]["start_time"] < parsed_beats[i]["start_time"] - 0.005:
            return [], "non_monotonic_or_overlapping"
        # Overlap check: beat i end cannot significantly exceed beat i+1 start
        if parsed_beats[i]["end_time"] > parsed_beats[i + 1]["start_time"] + 0.05:
            return [], "non_monotonic_or_overlapping"

    scene_dur = round(scene_end_time - scene_start_time, 3)
    first_s = parsed_beats[0]["start_time"]
    last_e = parsed_beats[-1]["end_time"]

    # Negative timestamps are always invalid
    if first_s < -0.05:
        return [], "out_of_bounds_timing"

    explicit_aligned = any(
        str(b.get("timing_type") or "").lower() in ("aligned", "fallback_proportional")
        for b in parsed_beats
    )
    explicit_relative = any(
        str(b.get("timing_type") or "").lower() in ("relative", "scene_relative")
        for b in parsed_beats
    )

    # 1. First scene (starts at 0.0): continuous and relative are identical [0, scene_dur]
    if scene_start_time <= 0.05:
        if last_e > scene_dur + 0.35:
            return [], "out_of_bounds_timing"
        parsed_beats[0]["start_time"] = max(0.0, parsed_beats[0]["start_time"])
        parsed_beats[-1]["end_time"] = scene_end_time
        for b in parsed_beats:
            b["duration_seconds"] = round(b["end_time"] - b["start_time"], 3)
        return parsed_beats, None

    # 2. Non-zero start scene
    if explicit_relative:
        if last_e > scene_dur + 0.35:
            return [], "out_of_bounds_timing"
        for b in parsed_beats:
            b["start_time"] = round(scene_start_time + b["start_time"], 3)
            b["end_time"] = round(scene_start_time + b["end_time"], 3)
            b["duration_seconds"] = round(b["end_time"] - b["start_time"], 3)
        parsed_beats[0]["start_time"] = scene_start_time
        parsed_beats[-1]["end_time"] = scene_end_time
        for b in parsed_beats:
            b["duration_seconds"] = round(b["end_time"] - b["start_time"], 3)
        return parsed_beats, None

    if explicit_aligned:
        if first_s < scene_start_time - 0.25 or last_e > scene_end_time + 0.35:
            return [], "out_of_bounds_timing"
        parsed_beats[0]["start_time"] = scene_start_time
        parsed_beats[-1]["end_time"] = scene_end_time
        for b in parsed_beats:
            b["duration_seconds"] = round(b["end_time"] - b["start_time"], 3)
        return parsed_beats, None

    # No explicit timing_type: evaluate continuous-timeline vs scene-relative across the entire collection
    is_continuous = (
        first_s >= scene_start_time - 0.25
        and last_e <= scene_end_time + 0.35
    )
    is_relative = (
        first_s <= 0.15
        and last_e <= scene_dur + 0.35
    )

    if is_continuous and not is_relative:
        parsed_beats[0]["start_time"] = scene_start_time
        parsed_beats[-1]["end_time"] = scene_end_time
        for b in parsed_beats:
            b["duration_seconds"] = round(b["end_time"] - b["start_time"], 3)
        return parsed_beats, None

    elif is_relative and not is_continuous:
        for b in parsed_beats:
            b["start_time"] = round(scene_start_time + b["start_time"], 3)
            b["end_time"] = round(scene_start_time + b["end_time"], 3)
            b["duration_seconds"] = round(b["end_time"] - b["start_time"], 3)
        parsed_beats[0]["start_time"] = scene_start_time
        parsed_beats[-1]["end_time"] = scene_end_time
        for b in parsed_beats:
            b["duration_seconds"] = round(b["end_time"] - b["start_time"], 3)
        return parsed_beats, None

    elif is_continuous and is_relative:
        return [], "ambiguous_timing"

    else:
        return [], "out_of_bounds_timing"


def _calculate_beat_frame_allocations(
    beats: list[dict[str, Any]],
    scene_start_frame: int,
    scene_total_frames: int,
    target_fps: int = 30,
) -> list[tuple[dict[str, Any], int, int, float]]:
    """Calculate deterministic, contiguous frame allocations for beats within a scene.

    Guarantees:
    1. Returns [] if scene_total_frames < number of beats (insufficient frames defensive guard).
    2. Every allocated beat gets at least 1 frame.
    3. Zero frame gaps: beat[k].start_frame == beat[k-1].end_frame.
    4. Zero overlaps.
    5. Final beat ends exactly at scene_start_frame + scene_total_frames.
    6. Sum of beat frames equals scene_total_frames.

    Returns:
        List of tuples: (beat_dict, start_frame, frame_count, duration_seconds)
    """
    n_beats = len(beats)
    if n_beats == 0:
        return []

    # Defensive guard: if available frames is fewer than beats, reject cleanly
    # to trigger graceful fallback to scene-level rendering without negative/inverted frames
    if scene_total_frames < n_beats:
        return []

    scene_end_frame = scene_start_frame + scene_total_frames
    allocations: list[tuple[dict[str, Any], int, int, float]] = []
    prev_cum_frame = scene_start_frame

    for i, beat in enumerate(beats):
        if i == n_beats - 1:
            cum_frame = scene_end_frame
        else:
            raw_end = float(beat.get("end_time", 0.0))
            raw_cum = int(round(raw_end * target_fps))
            # Bound cumulative frame so every remaining beat gets >= 1 frame
            min_allowed = prev_cum_frame + 1
            max_allowed = scene_end_frame - (n_beats - 1 - i)
            cum_frame = max(min_allowed, min(max_allowed, raw_cum))

        b_frames = max(1, cum_frame - prev_cum_frame)
        b_start_frame = prev_cum_frame
        b_dur = round(b_frames / float(target_fps), 4)
        allocations.append((beat, b_start_frame, b_frames, b_dur))
        prev_cum_frame = cum_frame

    return allocations


SUPPORTED_MUSIC_EXTENSIONS: frozenset[str] = frozenset({".mp3", ".wav"})


def resolve_and_validate_music_track(
    music_path: str | Path | None = None,
    music_enabled: bool | None = None,
    channel_settings: dict[str, Any] | None = None,
    storage_root: Path | None = None,
) -> Path | None:
    """Resolve and validate an optional user-provided background music track.

    Precedence order:
    1. Explicit `music_path` and `music_enabled` arguments.
    2. Channel settings (`channel_settings["background_music"]` or `channel_settings["music_path"]`,
       with `channel_settings["music_enabled"]`).
    3. Global settings (`settings.render_music_path` and `settings.render_music_enabled`).

    Returns:
        Resolved absolute Path to the music track if enabled and valid, or None if disabled.

    Raises:
        PipelineError: If music is requested/enabled but the configured file is missing,
            empty, or has an unsupported format (only .mp3 and .wav are supported).
    """
    # 1. Check explicit disablement
    if music_enabled is False:
        return None

    ch_settings = channel_settings if isinstance(channel_settings, dict) else {}
    if music_enabled is None and ch_settings.get("music_enabled") is False:
        return None

    # 2. Resolve candidate path and whether music is explicitly enabled
    candidate: str | Path | None = music_path
    explicitly_enabled: bool = (music_enabled is True) or bool(ch_settings.get("music_enabled") is True)

    if candidate is None:
        candidate = ch_settings.get("background_music") or ch_settings.get("music_path")

    if candidate is None:
        global_enabled = bool(getattr(settings, "render_music_enabled", False))
        if global_enabled or explicitly_enabled:
            candidate = getattr(settings, "render_music_path", None)
            if global_enabled:
                explicitly_enabled = True

    # 3. If music is not enabled and no path was provided, it remains disabled
    if not explicitly_enabled and not candidate:
        return None

    # 4. If music was explicitly enabled but no track path is provided/configured:
    if explicitly_enabled and (candidate is None or (isinstance(candidate, str) and not candidate.strip())):
        raise PipelineError("Background music is enabled, but no music track path is configured")

    if candidate is None:
        return None

    raw_path_str = str(candidate).strip()
    if not raw_path_str:
        return None

    p = Path(raw_path_str)

    # 5. Validate file extension (.mp3 and .wav only)
    if p.suffix.lower() not in SUPPORTED_MUSIC_EXTENSIONS:
        raise PipelineError(
            f"Unsupported audio format for background music: '{p.suffix}'. Only .mp3 and .wav are supported."
        )

    # 6. Resolve relative paths
    if not p.is_absolute():
        root = storage_root or getattr(settings, "resolved_storage_dir", Path(settings.storage_dir).resolve())
        candidate_rel_root = (root / p).resolve()
        if candidate_rel_root.is_file():
            p = candidate_rel_root
        else:
            p = p.resolve()

    # 7. Validate file existence
    if not p.is_file():
        raise PipelineError(f"Background music file not found at '{p}'")

    # 8. Validate file size > 0
    if p.stat().st_size == 0:
        raise PipelineError(f"Background music file is empty: '{p}'")

    return p


def build_audio_mux_command(
    temp_video_mp4: Path,
    continuous_audio_abs: Path,
    temp_final_mp4: Path,
    music_file_abs: Path | None = None,
    idle_volume: float = 0.18,
    duck_ratio: float = 4.0,
    duck_threshold: float = 0.03,
    attack_ms: float = 150.0,
    release_ms: float = 350.0,
) -> list[str]:
    """Construct FFmpeg command for audio multiplexing with optional sidechain-ducked background music.

    Signal flow when music is provided:
    1. Input 0: Concatenated silent video (temp_video_mp4).
    2. Input 1: Canonical continuous narration audio (continuous_audio_abs).
    3. Input 2: Looped background music track (-stream_loop -1 -i music_file_abs).
    4. Audio filter graph:
       - [2:a]volume=<idle_vol>[bg]: scale music to baseline idle level during pauses (~ -15 dB).
       - [bg][1:a]sidechaincompress=...[ducked]: downward compression of music using narration as control signal.
       - [1:a][ducked]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[mixed]: mix narration and ducked music without reducing narration volume.
       - [mixed]alimiter=limit=0.95[aout]: peak lookahead limiter preventing digital clipping overshoots.
    """
    if music_file_abs is None:
        return [
            "ffmpeg",
            "-y",
            "-i", str(temp_video_mp4),
            "-i", str(continuous_audio_abs),
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-af", "apad",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "44100",
            "-ac", "2",
            "-shortest",
            "-movflags", "+faststart",
            str(temp_final_mp4),
        ]

    filter_complex = (
        f"[2:a]volume={idle_volume:.2f}[bg];"
        f"[bg][1:a]sidechaincompress=threshold={duck_threshold:.3f}:ratio={duck_ratio:.1f}:attack={attack_ms:.0f}:release={release_ms:.0f}[ducked];"
        f"[1:a][ducked]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[mixed];"
        f"[mixed]alimiter=limit=0.95[aout]"
    )

    return [
        "ffmpeg",
        "-y",
        "-i", str(temp_video_mp4),
        "-i", str(continuous_audio_abs),
        "-stream_loop", "-1",
        "-i", str(music_file_abs),
        "-filter_complex", filter_complex,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-ar", "44100",
        "-ac", "2",
        "-shortest",
        "-movflags", "+faststart",
        str(temp_final_mp4),
    ]


def generate_render_step(
    video: Video,
    prev: dict[str, Any],
    step: VideoStep | None = None,
    music_path: str | Path | None = None,
    music_enabled: bool | None = None,
) -> dict[str, Any]:
    """Execute the video rendering step using FFmpeg.

    Args:
        video: Video ORM model instance.
        prev: Dictionary containing outputs of preceding pipeline steps.
        step: Current VideoStep instance.

    Returns:
        Structured dictionary conforming to the pipeline handler interface.
    """
    render_start = time.perf_counter()

    # 1. Validate images step
    img_data = prev.get("images", {})
    if isinstance(img_data, dict) and isinstance(img_data.get("images"), dict):
        img_items = img_data["images"].get("items", [])
    elif isinstance(img_data, dict) and isinstance(img_data.get("items"), list):
        img_items = img_data["items"]
    else:
        img_items = []

    valid_images = {
        it["index"]: it
        for it in img_items
        if isinstance(it, dict) and it.get("index") and it.get("status") == "ok" and it.get("file")
    }
    if not valid_images:
        raise PipelineError("Images step must be completed with valid generated images before rendering")

    # 2. Validate voiceover step
    vo_data = prev.get("voiceover", {})
    if isinstance(vo_data, dict) and isinstance(vo_data.get("voiceover"), dict):
        vo_items = vo_data["voiceover"].get("items", [])
    elif isinstance(vo_data, dict) and isinstance(vo_data.get("items"), list):
        vo_items = vo_data["items"]
    else:
        vo_items = []

    valid_audio = {
        it["index"]: it
        for it in vo_items
        if isinstance(it, dict) and it.get("index") and it.get("status", "ok") == "ok"
    }
    if not valid_audio:
        raise PipelineError("Voiceover step must be completed with valid generated audio before rendering")

    # 3. Check scene alignment between images and audio
    all_indices = sorted(set(valid_images.keys()) | set(valid_audio.keys()))
    missing_images = [idx for idx in all_indices if idx not in valid_images]
    missing_audio = [idx for idx in all_indices if idx not in valid_audio]

    if missing_images or missing_audio:
        err_parts = []
        if missing_images:
            err_parts.append(f"scenes missing images: {missing_images}")
        if missing_audio:
            err_parts.append(f"scenes missing audio: {missing_audio}")
        raise PipelineError(f"Cannot render: {', '.join(err_parts)}")

    # 4. Resolve storage paths
    storage_root = getattr(settings, "resolved_storage_dir", Path(settings.storage_dir).resolve())
    video_dir = storage_root / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)

    final_mp4_path = video_dir / "final_video.mp4"
    temp_dir = video_dir / f"tmp_render_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_final_mp4 = temp_dir / "temp_final.mp4"

    # Extract channel settings if available
    channel = getattr(video, "channel", None)
    channel_settings = getattr(channel, "settings", None) if channel else None

    # Resolve and validate optional background music track
    try:
        resolved_music = resolve_and_validate_music_track(
            music_path=music_path,
            music_enabled=music_enabled,
            channel_settings=channel_settings,
            storage_root=storage_root,
        )
    except PipelineError:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    # 5. Detect and validate continuous narration audio asset
    cont_audio_rel: str | None = None
    if isinstance(vo_data, dict):
        if vo_data.get("audio_file") and isinstance(vo_data["audio_file"], str):
            cont_audio_rel = vo_data["audio_file"]
        elif vo_data.get("file") and isinstance(vo_data["file"], str) and vo_data["file"].endswith((".wav", ".mp3", ".aac", ".ogg", ".flac", ".m4a")):
            cont_audio_rel = vo_data["file"]
        elif isinstance(vo_data.get("voiceover"), dict):
            inner_vo = vo_data["voiceover"]
            if inner_vo.get("audio_file") and isinstance(inner_vo["audio_file"], str):
                cont_audio_rel = inner_vo["audio_file"]
            elif inner_vo.get("file") and isinstance(inner_vo["file"], str) and inner_vo["file"].endswith((".wav", ".mp3", ".aac", ".ogg", ".flac", ".m4a")):
                cont_audio_rel = inner_vo["file"]

    if not cont_audio_rel:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise PipelineError("Continuous narration audio file is missing in voiceover step output")

    continuous_audio_abs = (storage_root / cont_audio_rel).resolve()
    if not continuous_audio_abs.is_file():
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise PipelineError(f"Continuous narration audio file does not exist: {continuous_audio_abs}")

    try:
        import soundfile as sf
        audio_info = sf.info(str(continuous_audio_abs))
        continuous_duration = float(audio_info.duration)
    except Exception as exc:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise PipelineError(f"Failed to decode continuous narration audio '{continuous_audio_abs}': {exc}") from exc

    if continuous_duration <= 0.0:
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        raise PipelineError(f"Continuous narration audio file has zero duration: {continuous_audio_abs}")

    # 6. Allocate per-scene visual display intervals.
    # Preferred: actual narration word timestamps (forced alignment), so visual cuts land in the
    # pauses between scenes. Fallback: legacy estimated durations scaled to the audio length.
    aligned_bounds, timing_source, alignment_hash, alignment_reused = _resolve_aligned_bounds(
        video=video,
        vo_data=vo_data if isinstance(vo_data, dict) else {},
        valid_audio=valid_audio,
        all_indices=all_indices,
        continuous_audio_abs=continuous_audio_abs,
        continuous_duration=continuous_duration,
        storage_root=storage_root,
    )

    scene_durations: dict[int, float] = {}
    prelim_durations: dict[int, float] = {}
    for idx in all_indices:
        sc_audio = valid_audio[idx]
        raw_dur = float(sc_audio.get("duration_seconds", 0.0) or 0.0)
        if raw_dur <= 0.0 and "end_time" in sc_audio and "start_time" in sc_audio:
            raw_dur = float(sc_audio["end_time"]) - float(sc_audio["start_time"])
        if raw_dur <= 0.0:
            raw_dur = continuous_duration / len(all_indices)
        prelim_durations[idx] = max(0.5, round(raw_dur, 3))

    # Adjust scene durations so total visual timeline matches continuous audio duration
    if len(all_indices) == 1:
        scene_durations[all_indices[0]] = round(continuous_duration, 3)
    else:
        total_prelim = sum(prelim_durations.values())
        scale = (continuous_duration / total_prelim) if (total_prelim > 0 and abs(total_prelim - continuous_duration) > 0.05) else 1.0

        accum = 0.0
        for i, idx in enumerate(all_indices):
            if i == len(all_indices) - 1:
                final_dur = max(0.5, round(continuous_duration - accum, 3))
                scene_durations[idx] = final_dur
            else:
                d = max(0.5, round(prelim_durations[idx] * scale, 3))
                scene_durations[idx] = d
                accum += d

    # Render configurations
    target_width = getattr(settings, "render_width", 1080)
    target_height = getattr(settings, "render_height", 1920)
    target_fps = getattr(settings, "render_fps", 30)
    timeout_sec = getattr(settings, "render_timeout_seconds", 300.0)

    # Cumulative scene end times on the narration timeline.
    if aligned_bounds is not None:
        scene_end_times = {idx: aligned_bounds[idx][1] for idx in all_indices}
        scene_end_times[all_indices[-1]] = continuous_duration
    else:
        scene_end_times = {}
        running = 0.0
        for idx in all_indices:
            running += scene_durations[idx]
            scene_end_times[idx] = running

    # Frame-accurate segment lengths derived from cumulative end timestamps so that rounding
    # never accumulates drift between the concatenated picture and the continuous narration.
    scene_frames: dict[int, int] = {}
    scene_start_frames: dict[int, int] = {}
    prev_cum_frames = 0
    for idx in all_indices:
        if idx == all_indices[-1]:
            # Ceil final scene frame boundary to guarantee video covers entire continuous audio duration
            cum_frames = int(math.ceil(round(scene_end_times[idx] * target_fps, 4)))
        else:
            cum_frames = int(round(scene_end_times[idx] * target_fps))
        frames = max(1, cum_frames - prev_cum_frames)
        scene_frames[idx] = frames
        scene_start_frames[idx] = prev_cum_frames
        prev_cum_frames += frames
        scene_durations[idx] = round(frames / float(target_fps), 4)

    try:
        # 7. Render individual scene segments into temp_dir
        segment_paths: list[Path] = []
        expected_total_duration = 0.0

        # Extract storyboard scene metadata if available to guide camera motion
        sb_outer = prev.get("storyboard") or {}
        sb_inner = (
            sb_outer.get("storyboard")
            if isinstance(sb_outer, dict) and isinstance(sb_outer.get("storyboard"), dict)
            else (sb_outer if isinstance(sb_outer, dict) else {})
        )
        sb_scenes = sb_inner.get("scenes") if isinstance(sb_inner.get("scenes"), list) else []
        scenes_by_idx = {
            s["index"]: s
            for s in sb_scenes
            if isinstance(s, dict) and s.get("index")
        }

        rendered_animations: list[dict[str, Any]] = []
        prev_anim_type: str | None = None

        for idx in all_indices:
            img_rel = valid_images[idx]["file"]
            img_abs = (storage_root / img_rel).resolve()
            if not img_abs.is_file():
                raise PipelineError(f"Image file for scene {idx} does not exist: {img_abs}")

            scene_dur = scene_durations[idx]
            expected_total_duration += scene_dur

            sc_meta = scenes_by_idx.get(idx, {})
            sc_audio = valid_audio.get(idx, {})

            scene_start_t = round(scene_start_frames[idx] / float(target_fps), 3)
            scene_end_t = round((scene_start_frames[idx] + scene_frames[idx]) / float(target_fps), 3)

            usable_beats, fallback_reason = _resolve_scene_beats(
                scene_idx=idx,
                scene_meta=sc_meta,
                audio_meta=sc_audio,
                scene_start_time=scene_start_t,
                scene_end_time=scene_end_t,
            )

            if usable_beats:
                allocations = _calculate_beat_frame_allocations(
                    beats=usable_beats,
                    scene_start_frame=scene_start_frames[idx],
                    scene_total_frames=scene_frames[idx],
                    target_fps=target_fps,
                )
                if not allocations:
                    usable_beats = []
                    fallback_reason = "insufficient_frames"

            if usable_beats:
                # -------------------------------------------------------------
                # Beat-level rendering path
                # -------------------------------------------------------------
                for alloc_idx, (beat, b_start_frame, b_frames, b_dur) in enumerate(allocations, start=1):
                    b_order = int(beat.get("order") or alloc_idx)
                    b_id = str(beat.get("id") or f"beat_{idx}_{b_order}")
                    seg_path = temp_dir / f"scene_{idx:02d}_beat_{b_order:02d}.mp4"

                    b_img_abs, b_img_rel, is_distinct = _resolve_beat_image(
                        beat=beat,
                        scene_idx=idx,
                        valid_images=valid_images,
                        img_items=img_items,
                        storage_root=storage_root,
                    )
                    if not b_img_abs.is_file():
                        raise PipelineError(f"Image file for beat {b_id} does not exist: {b_img_abs}")

                    # Determine image dimensions
                    try:
                        from PIL import Image
                        with Image.open(b_img_abs) as im:
                            src_w, src_h = im.size
                    except Exception:
                        src_w, src_h = 1024, 1536

                    feasibility = evaluate_beat_crop_feasibility(
                        beat=beat,
                        parent_scene=sc_meta,
                        source_width=src_w,
                        source_height=src_h,
                        image_path=b_img_abs,
                    )

                    status = feasibility.status
                    warnings = list(feasibility.warnings)
                    fallback_applied: str | None = None
                    fallback_reason_item: str | None = None

                    # Extract any explicit camera plan movement from the beat or parent scene
                    explicit_anim = None
                    cam_p = beat.get("camera_plan") if isinstance(beat, dict) else getattr(beat, "camera_plan", None)
                    if isinstance(cam_p, dict) and cam_p.get("movement"):
                        explicit_anim = str(cam_p["movement"]).lower().replace("_", "-").strip()
                    elif isinstance(cam_p, str) and cam_p.strip():
                        explicit_anim = cam_p.lower().replace("_", "-").strip()
                    elif hasattr(cam_p, "movement") and getattr(cam_p, "movement"):
                        explicit_anim = str(getattr(cam_p, "movement")).lower().replace("_", "-").strip()

                    if status == "safe_reuse":
                        crop_reg = feasibility.proposed_crop or calculate_portrait_crop(
                            source_width=src_w,
                            source_height=src_h,
                            scale_factor=feasibility.scale_factor,
                        )
                        camera_plan = plan_beat_camera_motion(
                            beat=beat,
                            crop_region=crop_reg,
                            source_width=src_w,
                            source_height=src_h,
                            focal_x=crop_reg.center_x,
                            focal_y=crop_reg.center_y,
                            duration_seconds=b_dur,
                            preferred_animation=explicit_anim,
                            prev_anim_type=prev_anim_type,
                        )
                        anim_plan = camera_plan.to_animation_plan(
                            focal_desc=str(beat.get("focal_point") or ""),
                            max_scale=QUESTIONABLE_REUSE_MAX_SCALE,
                        )
                    elif status == "questionable_reuse":
                        conservative_scale = min(SAFE_REUSE_MAX_SCALE, feasibility.scale_factor)
                        base_fx = feasibility.proposed_crop.center_x if feasibility.proposed_crop else 0.5
                        base_fy = feasibility.proposed_crop.center_y if feasibility.proposed_crop else DEFAULT_SUBJECT_Y
                        conservative_crop = calculate_portrait_crop(
                            source_width=src_w,
                            source_height=src_h,
                            focal_x=base_fx,
                            focal_y=base_fy,
                            scale_factor=conservative_scale,
                        )
                        camera_plan = plan_beat_camera_motion(
                            beat=beat,
                            crop_region=conservative_crop,
                            source_width=src_w,
                            source_height=src_h,
                            focal_x=conservative_crop.center_x,
                            focal_y=conservative_crop.center_y,
                            duration_seconds=b_dur,
                            preferred_animation=explicit_anim,
                            prev_anim_type=prev_anim_type,
                        )
                        anim_plan = camera_plan.to_animation_plan(
                            focal_desc=str(beat.get("focal_point") or ""),
                            max_scale=SAFE_REUSE_MAX_SCALE,
                        )
                        fallback_applied = "conservative_scale_clamp"
                        fallback_reason_item = "Reframing scale clamped to 1.45x to maintain visual sharpness."
                        warnings.append(fallback_reason_item)
                    else:  # new_image_needed
                        if is_distinct:
                            distinct_crop = calculate_portrait_crop(
                                source_width=src_w,
                                source_height=src_h,
                                scale_factor=1.00,
                            )
                            camera_plan = plan_beat_camera_motion(
                                beat=beat,
                                crop_region=distinct_crop,
                                source_width=src_w,
                                source_height=src_h,
                                focal_x=0.5,
                                focal_y=DEFAULT_SUBJECT_Y,
                                duration_seconds=b_dur,
                                preferred_animation=explicit_anim,
                                prev_anim_type=prev_anim_type,
                            )
                            anim_plan = camera_plan.to_animation_plan(
                                focal_desc=str(beat.get("focal_point") or ""),
                                max_scale=QUESTIONABLE_REUSE_MAX_SCALE,
                            )
                        else:
                            fallback_crop = calculate_portrait_crop(
                                source_width=src_w,
                                source_height=src_h,
                                scale_factor=1.00,
                            )
                            camera_plan = plan_beat_camera_motion(
                                beat=beat,
                                crop_region=fallback_crop,
                                source_width=src_w,
                                source_height=src_h,
                                focal_x=0.5,
                                focal_y=DEFAULT_SUBJECT_Y,
                                duration_seconds=b_dur,
                                preferred_animation=explicit_anim,
                                prev_anim_type=prev_anim_type,
                            )
                            anim_plan = camera_plan.to_animation_plan(
                                focal_desc=str(beat.get("focal_point") or ""),
                                max_scale=1.15,
                            )
                            fallback_applied = "conservative_parent_framing"
                            fallback_reason_item = feasibility.reason
                            warnings.append(
                                f"Beat {b_id} requires new image ({feasibility.reason}); "
                                "rendered with conservative full framing fallback as automatic image generation is not permitted."
                            )

                    beat_shot_type = str(beat.get("shot_type") or sc_meta.get("shot_type") or "medium").lower()
                    z_expr, x_expr, y_expr, filter_chain, resolved_anim = build_ffmpeg_zoompan_filter(
                        plan=anim_plan,
                        scene_dur=b_dur,
                        target_fps=target_fps,
                        target_width=target_width,
                        target_height=target_height,
                        fallback_shot_type=beat_shot_type,
                        fallback_scene_idx=idx,
                    )
                    prev_anim_type = resolved_anim.get("animation_type")
                    rendered_animations.append(
                        {
                            "scene_index": idx,
                            "beat_id": b_id,
                            "beat_order": b_order,
                            "render_mode": "visual_beat",
                            "beat_fallback_reason": None,
                            "feasibility_status": status,
                            "image_file": b_img_rel,
                            "fallback_applied": fallback_applied,
                            "fallback_reason": fallback_reason_item,
                            "warnings": warnings,
                            **resolved_anim,
                            "start_time": round(b_start_frame / float(target_fps), 3),
                            "end_time": round((b_start_frame + b_frames) / float(target_fps), 3),
                            "frames": b_frames,
                        }
                    )

                    seg_cmd = [
                        "ffmpeg",
                        "-y",
                        "-i", str(b_img_abs),
                        "-filter_complex", f"[0:v]{filter_chain}[v]",
                        "-map", "[v]",
                        "-frames:v", str(b_frames),
                        "-c:v", "libx264",
                        "-preset", "fast",
                        "-crf", "19",
                        "-pix_fmt", "yuv420p",
                        "-an",
                        str(seg_path),
                    ]

                    try:
                        subprocess.run(
                            seg_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=True,
                            timeout=timeout_sec,
                        )
                    except subprocess.CalledProcessError as exc:
                        err_tail = exc.stderr[-500:] if exc.stderr else "Unknown error"
                        raise PipelineError(f"FFmpeg failed rendering beat {b_id} (scene {idx}): {err_tail.strip()}") from exc
                    except subprocess.TimeoutExpired as exc:
                        raise PipelineError(f"FFmpeg timed out rendering beat {b_id} (scene {idx}) after {timeout_sec}s") from exc

                    if not seg_path.is_file() or seg_path.stat().st_size == 0:
                        raise PipelineError(f"FFmpeg produced empty segment file for beat {b_id} (scene {idx})")

                    segment_paths.append(seg_path)

            else:
                # -------------------------------------------------------------
                # Legacy scene-level rendering path
                # -------------------------------------------------------------
                if fallback_reason is not None:
                    logger.warning(
                        "VisualBeats for scene %s bypassed (%s); falling back to legacy scene rendering",
                        idx,
                        fallback_reason,
                    )

                seg_path = temp_dir / f"scene_{idx:02d}.mp4"

                # Resolve AI-directed animation plan tailored to this scene's composition
                shot_type = str(sc_meta.get("shot_type") or "medium").lower()

                anim_plan = plan_scene_animation(
                    scene_data=sc_meta,
                    image_path=img_abs,
                    duration_seconds=scene_dur,
                    prev_anim_type=prev_anim_type,
                )

                z_expr, x_expr, y_expr, filter_chain, resolved_anim = build_ffmpeg_zoompan_filter(
                    plan=anim_plan,
                    scene_dur=scene_dur,
                    target_fps=target_fps,
                    target_width=target_width,
                    target_height=target_height,
                    fallback_shot_type=shot_type,
                    fallback_scene_idx=idx,
                )
                prev_anim_type = resolved_anim.get("animation_type")
                rendered_animations.append(
                    {
                        "scene_index": idx,
                        "render_mode": "scene",
                        "beat_fallback_reason": fallback_reason,
                        **resolved_anim,
                        "start_time": round(scene_start_frames[idx] / float(target_fps), 3),
                        "end_time": round((scene_start_frames[idx] + scene_frames[idx]) / float(target_fps), 3),
                        "frames": scene_frames[idx],
                    }
                )

                # Video-only segment (no audio stream encoded; continuous audio is muxed after concatenation)
                seg_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i", str(img_abs),
                    "-filter_complex", f"[0:v]{filter_chain}[v]",
                    "-map", "[v]",
                    "-frames:v", str(scene_frames[idx]),
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "19",
                    "-pix_fmt", "yuv420p",
                    "-an",
                    str(seg_path),
                ]

                try:
                    subprocess.run(
                        seg_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=True,
                        timeout=timeout_sec,
                    )
                except subprocess.CalledProcessError as exc:
                    err_tail = exc.stderr[-500:] if exc.stderr else "Unknown error"
                    raise PipelineError(f"FFmpeg failed rendering scene {idx}: {err_tail.strip()}") from exc
                except subprocess.TimeoutExpired as exc:
                    raise PipelineError(f"FFmpeg timed out rendering scene {idx} after {timeout_sec}s") from exc

                if not seg_path.is_file() or seg_path.stat().st_size == 0:
                    raise PipelineError(f"FFmpeg produced empty segment file for scene {idx}")

                segment_paths.append(seg_path)

        # 8. Concatenate segments and mux continuous narration audio
        concat_file = temp_dir / "concat_list.txt"
        concat_lines = [f"file '{p.resolve()}'" for p in segment_paths]
        concat_file.write_text("\n".join(concat_lines), encoding="utf-8")

        temp_video_mp4 = temp_dir / "temp_video.mp4"
        concat_cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(temp_video_mp4),
        ]
        try:
            subprocess.run(
                concat_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=timeout_sec,
            )
        except subprocess.CalledProcessError as exc:
            err_tail = exc.stderr[-500:] if exc.stderr else "Unknown error"
            raise PipelineError(f"FFmpeg video concatenation failed: {err_tail.strip()}") from exc
        except subprocess.TimeoutExpired as exc:
            raise PipelineError(f"FFmpeg timed out concatenating video segments after {timeout_sec}s") from exc

        if not temp_video_mp4.is_file() or temp_video_mp4.stat().st_size == 0:
            raise PipelineError("FFmpeg produced empty concatenated video")

        # Mux continuous narration audio WAV into final MP4 (with optional ducked background music).
        idle_vol = float(getattr(settings, "render_music_idle_volume", 0.18))
        duck_ratio = float(getattr(settings, "render_music_duck_ratio", 4.0))
        duck_thresh = float(getattr(settings, "render_music_duck_threshold", 0.03))
        attack_ms = float(getattr(settings, "render_music_attack_ms", 150.0))
        release_ms = float(getattr(settings, "render_music_release_ms", 350.0))

        mux_cmd = build_audio_mux_command(
            temp_video_mp4=temp_video_mp4,
            continuous_audio_abs=continuous_audio_abs,
            temp_final_mp4=temp_final_mp4,
            music_file_abs=resolved_music,
            idle_volume=idle_vol,
            duck_ratio=duck_ratio,
            duck_threshold=duck_thresh,
            attack_ms=attack_ms,
            release_ms=release_ms,
        )
        try:
            subprocess.run(
                mux_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=timeout_sec,
            )
        except subprocess.CalledProcessError as exc:
            err_tail = exc.stderr[-500:] if exc.stderr else "Unknown error"
            raise PipelineError(f"FFmpeg audio muxing failed: {err_tail.strip()}") from exc
        except subprocess.TimeoutExpired as exc:
            raise PipelineError(f"FFmpeg timed out muxing continuous audio after {timeout_sec}s") from exc

        if not temp_final_mp4.is_file() or temp_final_mp4.stat().st_size == 0:
            raise PipelineError("FFmpeg produced empty concatenated video output")

        # 9. Validate rendered MP4 using ffprobe
        probe_data = _probe_video_file(temp_final_mp4)

        streams = probe_data.get("streams", [])
        v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        if not v_stream:
            raise PipelineError("Rendered video contains no video stream")
        if not a_stream:
            raise PipelineError("Rendered video contains no audio stream")

        v_codec = v_stream.get("codec_name", "")
        if v_codec != "h264":
            raise PipelineError(f"Expected video codec 'h264', got '{v_codec}'")

        v_pix_fmt = v_stream.get("pix_fmt", "")
        if v_pix_fmt != "yuv420p":
            raise PipelineError(f"Expected pixel format 'yuv420p', got '{v_pix_fmt}'")

        v_width = v_stream.get("width")
        v_height = v_stream.get("height")
        if v_width != target_width or v_height != target_height:
            raise PipelineError(f"Expected resolution {target_width}x{target_height}, got {v_width}x{v_height}")

        a_codec = a_stream.get("codec_name", "")
        if a_codec != "aac":
            raise PipelineError(f"Expected audio codec 'aac', got '{a_codec}'")

        format_info = probe_data.get("format", {})
        actual_duration = round(float(format_info.get("duration", 0.0)), 3)
        file_size_bytes = int(format_info.get("size", temp_final_mp4.stat().st_size))

        if actual_duration <= 0.5:
            raise PipelineError(f"Rendered video duration ({actual_duration}s) is suspiciously short")

        # Duration integrity checks:
        # 1. Video duration must match expected duration derived from total frame count within 1 frame
        expected_video_duration = round(prev_cum_frames / float(target_fps), 3)
        v_stream_dur = round(float(v_stream.get("duration") or actual_duration), 3)
        if abs(v_stream_dur - expected_video_duration) > (1.0 / target_fps + 0.02):
            raise PipelineError(
                f"Video stream duration ({v_stream_dur}s) deviates from expected duration ({expected_video_duration}s)"
            )

        # 2. Audio stream must not be truncated below continuous narration duration
        a_stream_dur = round(float(a_stream.get("duration") or actual_duration), 3)
        if a_stream_dur < round(continuous_duration - 0.05, 3):
            raise PipelineError(
                f"Rendered audio stream duration ({a_stream_dur}s) was truncated below narration audio duration ({continuous_duration}s)"
            )

        # 10. Promote temp output to final destination atomically
        shutil.move(str(temp_final_mp4), str(final_mp4_path))

        # 11. Update Video model output_path and commit session if attached
        video.output_path = str(final_mp4_path)
        sess = None
        try:
            sess = object_session(video)
        except Exception:
            sess = None

        if sess is not None:
            try:
                sess.commit()
            except Exception:
                sess.rollback()

        render_latency = round(time.perf_counter() - render_start, 3)
        total_beats_rendered = sum(1 for a in rendered_animations if a.get("render_mode") == "visual_beat")
        render_mode = "visual_beats" if total_beats_rendered > 0 else "scene"
        beat_fallbacks = {
            a["scene_index"]: a["beat_fallback_reason"]
            for a in rendered_animations
            if a.get("beat_fallback_reason") is not None
        }

        # 12. Construct structured output
        return {
            "render": {
                "video_path": str(final_mp4_path),
                "duration_seconds": actual_duration,
                "resolution": f"{target_width}x{target_height}",
                "fps": target_fps,
                "codec": "h264",
                "audio_codec": "aac",
                "file_size_bytes": file_size_bytes,
                "scenes_rendered": len(all_indices),
                "beats_rendered": total_beats_rendered,
                "render_mode": render_mode,
                "beat_fallback_reasons": beat_fallbacks,
                "animations": rendered_animations,
                "audio_source": "continuous",
                "audio_file": cont_audio_rel,
                "timing_source": timing_source,
                "alignment_hash": alignment_hash,
                "music": {
                    "enabled": resolved_music is not None,
                    "file": str(resolved_music) if resolved_music else None,
                },
            },
            # Top-level video_path and duration_seconds for backward compatibility
            "video_path": str(final_mp4_path),
            "duration_seconds": actual_duration,
            "meta": {
                "task": "render",
                "scenes_count": len(all_indices),
                "beats_rendered": total_beats_rendered,
                "render_mode": render_mode,
                "beat_fallback_reasons": beat_fallbacks,
                "resolution": f"{target_width}x{target_height}",
                "fps": target_fps,
                "latency_seconds": render_latency,
                "file_size_bytes": file_size_bytes,
                "audio_source": "continuous",
                "audio_file": cont_audio_rel,
                "timing_source": timing_source,
                "alignment_hash": alignment_hash,
                "alignment_reused": alignment_reused,
                "music_enabled": resolved_music is not None,
                "music_file": str(resolved_music) if resolved_music else None,
            },
        }

    finally:
        # Clean up temporary segments directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
