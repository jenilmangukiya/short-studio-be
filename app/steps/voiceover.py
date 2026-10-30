"""Voiceover step implementation for Shorts Studio.

Generates continuous video narration audio for the complete video as a single
logical audio asset using Kokoro-82M (or mock for automated tests), with
natural linguistic chunking, content-addressable caching/reuse, and scene-to-narration
timing mapping for downstream rendering and subtitle alignment.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from sqlalchemy.orm import object_session

from app.config import settings
from app.models import Video, VideoStep
from app.subtitles import (
    BaseSubtitleProvider,
    SubtitleError,
    apply_aligned_timings_to_items,
    compute_aligned_scene_timings,
    get_subtitle_provider,
    load_or_compute_alignment,
)
from app.voiceover import (
    BaseVoiceoverProvider,
    VoiceoverConfigError,
    VoiceoverError,
    collect_narration_script,
    estimate_scene_timings,
    generate_continuous_audio,
    get_voiceover_provider,
)


class PipelineError(Exception):
    """Raised when preconditions or step executions fail."""


def _build_output_payload(
    items: list[dict[str, Any]],
    continuous_file_rel: str | None,
    narration_script: str,
    continuous_hash: str,
    warnings: list[str],
    provider: BaseVoiceoverProvider,
    total_scenes: int,
    voice: str,
    speed: float,
    total_duration: float,
    total_latency: float,
    chunks_count: int = 1,
    inference_calls: int = 1,
    is_complete: bool = False,
    reused: bool = False,
    alignment_hash: str | None = None,
    alignment_reused: bool = False,
) -> dict[str, Any]:
    """Construct the standardized output payload for the continuous voiceover step."""
    generated_count = 0 if reused else (len(items) if is_complete else 0)
    reused_count = len(items) if reused else 0
    failed_count = sum(1 for it in items if it.get("status") == "failed")
    timing_type = (
        "aligned"
        if items and all(it.get("timing_type") == "aligned" for it in items)
        else "estimated"
    )

    payload: dict[str, Any] = {
        "voiceover": {
            "continuous": True,
            "audio_file": continuous_file_rel,
            "narration_script": narration_script,
            "audio_hash": continuous_hash,
            "items": [dict(it) for it in items],
            "complete": is_complete,
            "total_duration_seconds": round(total_duration, 3),
            "timing_type": timing_type,
            "alignment_hash": alignment_hash,
        },
        # Top-level fields for downstream rendering, subtitle alignment, and compatibility
        "audio_file": continuous_file_rel,
        "file": continuous_file_rel,
        "narration_script": narration_script,
        "audio_hash": continuous_hash,
        "duration_seconds": round(total_duration, 3),
        "warnings": list(warnings),
        "meta": {
            "provider": provider.provider_name,
            "model": provider.model,
            "task": "voiceover",
            "scenes_total": total_scenes,
            "scenes_generated": generated_count,
            "scenes_reused": reused_count,
            "scenes_failed": failed_count,
            "voice": voice,
            "speed": speed,
            "latency_seconds": round(total_latency, 4),
            "continuous": True,
            "chunks_count": chunks_count,
            "inference_calls": inference_calls,
            "audio_file": continuous_file_rel,
            "audio_hash": continuous_hash,
            "timing_type": timing_type,
            "alignment_hash": alignment_hash,
            "alignment_reused": alignment_reused,
        },
    }

    if provider.provider_name == "mock":
        payload["mock"] = True
        payload["meta"]["mock"] = True

    return payload


def _persist_partial(step: VideoStep | None, payload: dict[str, Any]) -> None:
    """Save step output progress without changing step status."""
    if step is not None:
        step.output = dict(payload)
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


def _persist_failure(step: VideoStep | None, error_message: str) -> None:
    """Save error state in step output upon failure."""
    if step is not None:
        step.output = {
            "voiceover": {
                "continuous": True,
                "items": [],
                "complete": False,
                "error": error_message,
            },
            "error": error_message,
        }
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


def _align_scene_timings(
    video: Video,
    storage_root: Path,
    validated_scenes: list[dict[str, Any]],
    full_script: str,
    full_hash: str,
    continuous_file_abs: Path,
    total_duration: float,
    voiceover_provider: BaseVoiceoverProvider,
    subtitle_provider: BaseSubtitleProvider | None,
    warnings: list[str],
) -> tuple[list[dict[str, Any]], str, bool] | None:
    """Run (or reuse) Vosk word alignment on the continuous narration and map scene boundaries.

    Returns (aligned_timings, alignment_hash, reused) or None when alignment is unavailable,
    in which case callers fall back to estimated timing.
    """
    if voiceover_provider.provider_name == "mock" and subtitle_provider is None:
        # Synthetic mock audio contains no speech; skip alignment in tests/dev mock mode.
        return None
    try:
        aligner = subtitle_provider or get_subtitle_provider()
    except Exception:
        return None
    if aligner.provider_name == "mock":
        return None

    try:
        alignment, alignment_hash, reused = load_or_compute_alignment(
            video_id=video.id,
            storage_root=storage_root,
            sorted_scenes=validated_scenes,
            narration_script=full_script,
            continuous_audio_abs=continuous_file_abs,
            audio_hash=full_hash,
            provider=aligner,
        )
        timings = compute_aligned_scene_timings(
            validated_scenes,
            alignment,
            total_duration if total_duration > 0 else alignment.total_duration_seconds,
        )
    except SubtitleError as exc:
        warnings.append(f"Word alignment unavailable; using estimated scene timing ({exc}).")
        return None
    except Exception as exc:  # pragma: no cover - defensive
        warnings.append(f"Word alignment failed; using estimated scene timing ({exc}).")
        return None
    return timings, alignment_hash, reused


def generate_voiceover_step(
    video: Video,
    prev: dict[str, Any],
    step: VideoStep | None = None,
    voiceover_provider: BaseVoiceoverProvider | None = None,
    subtitle_provider: BaseSubtitleProvider | None = None,
) -> dict[str, Any]:
    """Execute the voiceover step for a video using continuous narration.

    Collects narration text across all storyboard scenes in order, generates
    a single continuous narration audio asset, runs word-level forced alignment
    on it (persisted and reused by render/subtitles), and maps scene boundaries
    to actual narration word timestamps. Falls back to proportional estimated
    timing when alignment is unavailable.

    Args:
        video: Video ORM model instance.
        prev: Dictionary containing outputs of preceding pipeline steps.
        step: Current VideoStep instance (for progress persistence).
        voiceover_provider: Optional provider override (e.g. for testing).
        subtitle_provider: Optional word-aligner override (e.g. for testing).

    Returns:
        Structured output payload conforming to the pipeline handler interface.
    """
    # 1. Validate storyboard input
    sb_outer = prev.get("storyboard")
    if not sb_outer or not isinstance(sb_outer, dict):
        raise PipelineError("Storyboard step must be completed before generating voiceover")

    sb = sb_outer.get("storyboard") if isinstance(sb_outer.get("storyboard"), dict) else sb_outer

    scenes = sb.get("scenes") if isinstance(sb, dict) else None
    if not scenes or not isinstance(scenes, list) or len(scenes) == 0:
        raise PipelineError("Storyboard step must be completed before generating voiceover and contain scenes")

    # 2. Collect and validate narration text from all scenes in order
    try:
        full_script, validated_scenes = collect_narration_script(scenes)
    except ValueError as exc:
        _persist_failure(step, str(exc))
        raise PipelineError(str(exc)) from exc

    # 3. Extract voice settings from channel or fallback to defaults
    channel = getattr(video, "channel", None)
    channel_settings = getattr(channel, "settings", None) if channel else None

    default_voice = getattr(settings, "voiceover_default_voice", "af_heart") or "af_heart"
    default_speed = float(getattr(settings, "voiceover_default_speed", 1.0) or 1.0)

    voice = default_voice
    speed = default_speed
    if isinstance(channel_settings, dict):
        if channel_settings.get("voice") and isinstance(channel_settings["voice"], str):
            voice = channel_settings["voice"].strip()
        if channel_settings.get("voice_speed") is not None:
            try:
                speed = float(channel_settings["voice_speed"])
            except (ValueError, TypeError):
                pass

    # 4. Check for user revision prompt on this step
    warnings: list[str] = []
    if step and step.user_prompt:
        warnings.append(
            "Instructions are not used by the voiceover step. Edit the script or storyboard to change narration text."
        )

    # 5. Resolve provider
    provider = voiceover_provider or get_voiceover_provider()

    # 6. Prepare filesystem storage directory
    storage_root = getattr(settings, "resolved_storage_dir", Path(settings.storage_dir).resolve())
    audio_dir = storage_root / str(video.id) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # 7. Compute continuous narration hash
    hash_payload = f"{full_script}|{voice}|{speed}|{provider.model}".encode("utf-8")
    full_hash = hashlib.sha256(hash_payload).hexdigest()
    hash8 = full_hash[:8]

    continuous_file_rel = f"{video.id}/audio/narration_{hash8}.wav"
    continuous_file_abs = storage_root / continuous_file_rel

    # 8. Check for reuse of previously generated continuous audio
    if step and isinstance(step.output, dict):
        prev_hash = step.output.get("audio_hash") or step.output.get("voiceover", {}).get("audio_hash")
        prev_items = step.output.get("voiceover", {}).get("items", [])
        if (
            prev_hash == full_hash
            and continuous_file_abs.is_file()
            and continuous_file_abs.stat().st_size > 44
            and len(prev_items) == len(validated_scenes)
            and all(it.get("status") == "ok" for it in prev_items)
        ):
            reused_items = [
                {k: v for k, v in {**it, "reused": True, "error": None}.items() if k != "file"}
                for it in prev_items
            ]
            reused_duration = float(step.output.get("duration_seconds", 0.0))
            prev_inner = step.output.get("voiceover", {}) if isinstance(step.output.get("voiceover"), dict) else {}
            alignment_hash = prev_inner.get("alignment_hash")
            alignment_reused = True
            if not all(it.get("timing_type") == "aligned" for it in reused_items):
                # Upgrade legacy estimated timing to real word-aligned timing (audio is reused as-is).
                aligned = _align_scene_timings(
                    video=video,
                    storage_root=storage_root,
                    validated_scenes=validated_scenes,
                    full_script=full_script,
                    full_hash=full_hash,
                    continuous_file_abs=continuous_file_abs,
                    total_duration=reused_duration,
                    voiceover_provider=provider,
                    subtitle_provider=subtitle_provider,
                    warnings=warnings,
                )
                if aligned is not None:
                    timings, alignment_hash, alignment_reused = aligned
                    reused_items = apply_aligned_timings_to_items(reused_items, timings)
            reused_payload = _build_output_payload(
                items=reused_items,
                continuous_file_rel=continuous_file_rel,
                narration_script=full_script,
                continuous_hash=full_hash,
                warnings=warnings,
                provider=provider,
                total_scenes=len(validated_scenes),
                voice=voice,
                speed=speed,
                total_duration=reused_duration,
                total_latency=0.0,
                chunks_count=int(step.output.get("meta", {}).get("chunks_count", 1)),
                inference_calls=int(step.output.get("meta", {}).get("inference_calls", 1)),
                is_complete=True,
                reused=True,
                alignment_hash=alignment_hash,
                alignment_reused=alignment_reused,
            )
            _persist_partial(step, reused_payload)
            return reused_payload

    # 9. Calculate target duration from storyboard scene estimates if provided
    total_est = sum(float(s.get("est_seconds") or 0.0) for s in validated_scenes)
    if total_est <= 0.0 and isinstance(sb, dict):
        total_est = float(sb.get("total_seconds") or 0.0)
    target_duration = total_est if total_est > 0.0 else None

    # Generate continuous narration audio
    try:
        result, chunks_count = generate_continuous_audio(
            narration_script=full_script,
            provider=provider,
            voice=voice,
            speed=speed,
            target_duration=target_duration,
        )
    except VoiceoverConfigError as exc:
        _persist_failure(step, str(exc))
        raise PipelineError(str(exc)) from exc
    except Exception as exc:
        _persist_failure(step, str(exc))
        raise PipelineError(f"Continuous voiceover generation failed: {exc}") from exc

    # Validate audio output
    if len(result.audio_bytes) <= 44 or result.duration_seconds <= 0.0 or result.sample_rate <= 0:
        err_msg = "Generated continuous narration audio is empty or invalid"
        _persist_failure(step, err_msg)
        raise PipelineError(err_msg)

    # Save continuous WAV asset to disk
    try:
        continuous_file_abs.write_bytes(result.audio_bytes)
    except Exception as exc:
        err_msg = f"Failed to write continuous audio file to disk: {exc}"
        _persist_failure(step, err_msg)
        raise PipelineError(err_msg) from exc

    # 10. Compute scene-to-narration timing mapping for downstream rendering and subtitle alignment.
    # Prefer actual word timestamps from forced alignment on the continuous WAV; fall back to
    # proportional estimates only when alignment is unavailable.
    scene_timings = estimate_scene_timings(validated_scenes, result.duration_seconds)
    alignment_hash: str | None = None
    alignment_reused = False
    aligned = _align_scene_timings(
        video=video,
        storage_root=storage_root,
        validated_scenes=validated_scenes,
        full_script=full_script,
        full_hash=full_hash,
        continuous_file_abs=continuous_file_abs,
        total_duration=result.duration_seconds,
        voiceover_provider=provider,
        subtitle_provider=subtitle_provider,
        warnings=warnings,
    )
    if aligned is not None:
        scene_timings, alignment_hash, alignment_reused = aligned

    items: list[dict[str, Any]] = []
    for idx_i, sc in enumerate(validated_scenes):
        sc_idx = sc["index"]
        timing = scene_timings[idx_i]

        sc_latency = round(
            result.latency_seconds * (timing["duration_seconds"] / max(0.1, result.duration_seconds)), 4
        )

        item: dict[str, Any] = {
            "index": sc_idx,
            "scene_id": sc.get("id", sc_idx),
            "status": "ok",
            "narration": sc["narration"],
            "audio_hash": full_hash,
            "reused": False,
            "start_time": timing["start_time"],
            "end_time": timing["end_time"],
            "duration_seconds": timing["duration_seconds"],
            "timing_type": timing.get("timing_type", "estimated"),
            "sample_rate": result.sample_rate,
            "voice": voice,
            "speed": speed,
            "latency_seconds": sc_latency,
            "error": None,
        }
        if "speech_start_time" in timing:
            item["speech_start_time"] = timing["speech_start_time"]
            item["speech_end_time"] = timing["speech_end_time"]
        items.append(item)

    # 11. Complete standardized payload
    final_payload = _build_output_payload(
        items=items,
        continuous_file_rel=continuous_file_rel,
        narration_script=full_script,
        continuous_hash=full_hash,
        warnings=warnings,
        provider=provider,
        total_scenes=len(validated_scenes),
        voice=voice,
        speed=speed,
        total_duration=result.duration_seconds,
        total_latency=result.latency_seconds,
        chunks_count=chunks_count,
        inference_calls=result.inference_calls,
        is_complete=True,
        reused=False,
        alignment_hash=alignment_hash,
        alignment_reused=alignment_reused,
    )
    _persist_partial(step, final_payload)
    return final_payload
