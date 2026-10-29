"""Subtitles pipeline step implementation for Shorts Studio.

Generates synchronized word-level and scene-level subtitle timing for the rendered video
using ground-truth narration audio and acoustic forced alignment (Vosk Kaldi HMM-TDNN on
continuous narration WAV, with compatible scene waveform fallback for legacy videos).
Caches alignment results by a deterministic SHA-256 hash of narration script, audio identity,
and aligner configuration.
Exports clean JSON, WebVTT, SRT, and ASS subtitle artifacts, and uses FFmpeg with libass
to burn animated word-level highlighted subtitles into a separate captioned MP4.
The original uncaptioned MP4 is preserved intact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
import uuid

from sqlalchemy.orm import object_session

from app.config import settings
from app.models import Video, VideoStep
from app.subtitles import (
    BaseSubtitleProvider,
    SubtitleError,
    SubtitleResult,
    get_subtitle_provider,
)


class PipelineError(Exception):
    """Raised when preconditions or subtitle generation fails."""


def _resolve_continuous_audio(
    vo_data: dict[str, Any], storage_root: Path
) -> tuple[str | None, Path | None, str | None, str | None]:
    """Extract continuous narration audio relative path, absolute path, script, and hash from voiceover output."""
    cont_rel: str | None = None
    narration_script: str | None = None
    audio_hash: str | None = None

    if isinstance(vo_data, dict):
        inner = vo_data.get("voiceover") if isinstance(vo_data.get("voiceover"), dict) else {}

        for candidate in (
            vo_data.get("audio_file"),
            inner.get("audio_file"),
            vo_data.get("file"),
            inner.get("file"),
        ):
            if isinstance(candidate, str) and candidate.strip().endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a")):
                cont_rel = candidate.strip()
                break

        narration_script = (
            vo_data.get("narration_script")
            or inner.get("narration_script")
        )
        audio_hash = (
            vo_data.get("audio_hash")
            or inner.get("audio_hash")
        )

    cont_abs: Path | None = None
    if cont_rel:
        cont_abs = (storage_root / cont_rel).resolve()

    return cont_rel, cont_abs, narration_script, audio_hash


def _compute_alignment_hash(
    sorted_scenes: list[dict[str, Any]],
    narration_script: str | None,
    continuous_audio_abs: Path | None,
    audio_hash: str | None,
    provider: BaseSubtitleProvider,
) -> str:
    """Compute a stable SHA-256 hash of narration text, audio content/identity, and aligner configuration."""
    scene_texts = "|".join(
        f"{s.get('index', i)}:{str(s.get('narration') or '').strip()}"
        for i, s in enumerate(sorted_scenes, 1)
    )
    script_part = (narration_script or scene_texts).strip()

    audio_identity = audio_hash or ""
    if continuous_audio_abs and continuous_audio_abs.is_file():
        stat = continuous_audio_abs.stat()
        audio_identity = f"{audio_identity}:{continuous_audio_abs.name}:{stat.st_size}"
    else:
        scene_files = "|".join(str(s.get("file") or "") for s in sorted_scenes)
        audio_identity = f"{audio_identity}:{scene_files}"

    config_part = f"{provider.provider_name}:{provider.model}:v2_caption_grouping"
    payload = f"{script_part}||{audio_identity}||{config_part}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def generate_subtitles_step(
    video: Video,
    prev: dict[str, Any],
    step: VideoStep | None = None,
    subtitle_provider: BaseSubtitleProvider | None = None,
) -> dict[str, Any]:
    """Execute the subtitle generation and video captioning step for a video.

    Args:
        video: Video ORM model instance.
        prev: Dictionary containing outputs of preceding pipeline steps.
        step: Current VideoStep instance (for status and error handling).
        subtitle_provider: Optional provider override (e.g. for testing).

    Returns:
        Structured dictionary conforming to the pipeline handler interface.
    """
    start_time = time.perf_counter()

    # 1. Validate render step precondition
    render_data = prev.get("render")
    video_path = getattr(video, "output_path", None)
    if not render_data and not video_path:
        raise PipelineError("Video render step must be completed before generating subtitles")

    # 2. Validate voiceover step precondition
    vo_data = prev.get("voiceover", {})
    if isinstance(vo_data, dict) and isinstance(vo_data.get("voiceover"), dict):
        vo_items = vo_data["voiceover"].get("items", [])
    elif isinstance(vo_data, dict) and isinstance(vo_data.get("items"), list):
        vo_items = vo_data["items"]
    else:
        vo_items = []

    valid_audio_items = [
        it for it in vo_items if isinstance(it, dict) and it.get("status") == "ok"
    ]
    if not valid_audio_items:
        raise PipelineError("Voiceover step must be completed with valid audio before generating subtitles")

    # 3. Sort scene audio items by index
    sorted_scenes = sorted(valid_audio_items, key=lambda it: it.get("index", 0))

    # 4. Resolve provider
    provider = subtitle_provider or get_subtitle_provider()

    # 5. Prepare filesystem storage directory
    storage_root = getattr(settings, "resolved_storage_dir", Path(settings.storage_dir).resolve())
    video_dir = storage_root / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)

    json_rel_path = f"{video.id}/subtitles.json"
    vtt_rel_path = f"{video.id}/subtitles.vtt"
    srt_rel_path = f"{video.id}/subtitles.srt"
    ass_rel_path = f"{video.id}/subtitles.ass"
    captioned_rel_path = f"{video.id}/final_video_captioned.mp4"

    json_abs_path = storage_root / json_rel_path
    vtt_abs_path = storage_root / vtt_rel_path
    srt_abs_path = storage_root / srt_rel_path
    ass_abs_path = storage_root / ass_rel_path
    captioned_abs_path = storage_root / captioned_rel_path

    # 6. Resolve continuous narration audio if present
    cont_rel, cont_abs, narration_script, audio_hash = _resolve_continuous_audio(
        vo_data if isinstance(vo_data, dict) else {}, storage_root
    )
    if cont_rel and cont_abs and not cont_abs.is_file():
        raise PipelineError(
            f"Continuous narration WAV referenced in voiceover step does not exist: {cont_abs}"
        )

    alignment_hash = _compute_alignment_hash(
        sorted_scenes=sorted_scenes,
        narration_script=narration_script,
        continuous_audio_abs=cont_abs,
        audio_hash=audio_hash,
        provider=provider,
    )
    cache_rel_path = f"{video.id}/audio/alignment_{alignment_hash[:12]}.json"
    cache_abs_path = storage_root / cache_rel_path

    # 7. Check alignment cache or execute subtitle alignment
    result: SubtitleResult | None = None
    reused_cache = False

    if cache_abs_path.is_file():
        try:
            cached_data = json.loads(cache_abs_path.read_text(encoding="utf-8"))
            if (
                isinstance(cached_data, dict)
                and cached_data.get("alignment_hash") == alignment_hash
                and cached_data.get("segments")
            ):
                result = SubtitleResult.from_dict(cached_data, reused=True)
                reused_cache = True
        except Exception:
            result = None

    if result is None:
        try:
            result = provider.generate(
                scenes=sorted_scenes,
                storage_root=storage_root,
                continuous_audio_path=cont_abs if (cont_abs and cont_abs.is_file()) else None,
                narration_script=narration_script,
            )
            result.alignment_hash = alignment_hash
            result.reused = False
        except Exception as exc:
            raise PipelineError(f"Subtitle generation failed: {exc}") from exc

        if cont_abs and cont_abs.parent.is_dir():
            try:
                cache_abs_path.write_text(
                    json.dumps(result.to_dict(), indent=2), encoding="utf-8"
                )
            except Exception:
                pass

    # 8. Write subtitle artifacts to storage
    subtitles_dict = result.to_dict()
    json_abs_path.write_text(json.dumps(subtitles_dict, indent=2), encoding="utf-8")
    vtt_abs_path.write_text(result.to_vtt(), encoding="utf-8")
    srt_abs_path.write_text(result.to_srt(), encoding="utf-8")
    ass_abs_path.write_text(result.to_ass(), encoding="utf-8")

    # 9. Burn subtitles into separate captioned video via FFmpeg if rendered video exists
    warnings: list[str] = []
    input_video_path = getattr(video, "output_path", None)
    if not input_video_path or not Path(input_video_path).is_file():
        candidate = video_dir / "final_video.mp4"
        if candidate.is_file():
            input_video_path = str(candidate)

    if input_video_path and Path(input_video_path).is_file():
        temp_captioned_path = video_dir / f"tmp_burn_{uuid.uuid4().hex[:8]}.mp4"
        timeout_sec = getattr(settings, "render_timeout_seconds", 300.0)

        burn_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(input_video_path),
            "-vf", f"ass={ass_abs_path.resolve()}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(temp_captioned_path),
        ]

        try:
            subprocess.run(
                burn_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=timeout_sec,
            )
            if temp_captioned_path.is_file() and temp_captioned_path.stat().st_size > 0:
                shutil.move(str(temp_captioned_path), str(captioned_abs_path))
            else:
                warnings.append("FFmpeg produced an empty captioned video output.")
        except subprocess.CalledProcessError as exc:
            err_tail = exc.stderr[-500:] if exc.stderr else "Unknown error"
            warnings.append(f"Subtitle burning failed: {err_tail.strip()}")
        except Exception as exc:
            warnings.append(f"Subtitle burning encountered error: {exc}")
        finally:
            if temp_captioned_path.exists():
                try:
                    temp_captioned_path.unlink()
                except OSError:
                    pass

    # 10. Check for user revision prompts (informational warning)
    if step and step.user_prompt:
        warnings.append("Subtitle timing was regenerated. To alter narration text, revise the script step.")

    total_latency = round(time.perf_counter() - start_time, 4)

    is_captioned_ready = captioned_abs_path.is_file() and captioned_abs_path.stat().st_size > 0

    # 11. Build output payload conforming to Shorts Studio conventions
    payload: dict[str, Any] = {
        "subtitles": {
            "segments": subtitles_dict["segments"],
            "format": "word_level",
            "timing_source": result.timing_source,
            "alignment_hash": alignment_hash,
            "reused": reused_cache,
            "audio_file": cont_rel,
            "total_words": result.total_words,
            "total_duration_seconds": result.total_duration_seconds,
            "file": json_rel_path,
            "vtt_file": vtt_rel_path,
            "srt_file": srt_rel_path,
            "ass_file": ass_rel_path,
            "captioned_file": captioned_rel_path if is_captioned_ready else None,
            "captioned_video_path": str(captioned_abs_path) if is_captioned_ready else None,
        },
        "subtitles_path": str(json_abs_path),
        "vtt_path": str(vtt_abs_path),
        "srt_path": str(srt_abs_path),
        "ass_path": str(ass_abs_path),
        "captioned_video_path": str(captioned_abs_path) if is_captioned_ready else None,
        "warnings": warnings,
        "meta": {
            "task": "subtitles",
            "provider": result.provider,
            "model": result.model,
            "timing_source": result.timing_source,
            "alignment_hash": alignment_hash,
            "reused": reused_cache,
            "scenes_count": len(result.segments),
            "words_count": result.total_words,
            "duration_seconds": result.total_duration_seconds,
            "captioned_burned": is_captioned_ready,
            "latency_seconds": total_latency,
        },
    }

    if provider.provider_name == "mock":
        payload["mock"] = True
        payload["meta"]["mock"] = True

    # 12. Persist step output and commit if session attached
    if step is not None:
        step.output = payload
        sess = None
        try:
            sess = object_session(step)
        except Exception:
            sess = None
        if sess is not None:
            try:
                sess.commit()
            except Exception:
                sess.rollback()

    return payload
