"""Video rendering step implementation for Shorts Studio.

Composes vertical short videos (1080x1920, 30 fps, H.264 / AAC) by animating
each storyboard scene's vertical image according to its timing metadata and
muxing the complete continuous narration audio WAV using FFmpeg.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
import uuid

from sqlalchemy.orm import object_session

from app.animation import build_ffmpeg_zoompan_filter, plan_scene_animation
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

    scenes = [
        {"index": idx, "narration": str(it.get("narration") or "").strip()}
        for idx, it in zip(all_indices, items)
    ]
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

    bounds = {t["index"]: (float(t["start_time"]), float(t["end_time"])) for t in timings}
    return bounds, "aligned", alignment_hash, reused


def generate_render_step(
    video: Video,
    prev: dict[str, Any],
    step: VideoStep | None = None,
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

            seg_path = temp_dir / f"scene_{idx:02d}.mp4"

            # Resolve AI-directed animation plan tailored to this scene's composition
            sc_meta = scenes_by_idx.get(idx, {})
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

        # Mux continuous narration audio WAV into final MP4
        mux_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(temp_video_mp4),
            "-i", str(continuous_audio_abs),
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", "44100",
            "-ac", "2",
            "-shortest",
            "-movflags", "+faststart",
            str(temp_final_mp4),
        ]
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
                "animations": rendered_animations,
                "audio_source": "continuous",
                "audio_file": cont_audio_rel,
                "timing_source": timing_source,
                "alignment_hash": alignment_hash,
            },
            # Top-level video_path and duration_seconds for backward compatibility
            "video_path": str(final_mp4_path),
            "duration_seconds": actual_duration,
            "meta": {
                "task": "render",
                "scenes_count": len(all_indices),
                "resolution": f"{target_width}x{target_height}",
                "fps": target_fps,
                "latency_seconds": render_latency,
                "file_size_bytes": file_size_bytes,
                "audio_source": "continuous",
                "audio_file": cont_audio_rel,
                "timing_source": timing_source,
                "alignment_hash": alignment_hash,
                "alignment_reused": alignment_reused,
            },
        }

    finally:
        # Clean up temporary segments directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
