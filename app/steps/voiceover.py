"""Voiceover step implementation for Shorts Studio.

Generates vertical short video narration audio for each scene in a storyboard
using Kokoro-82M (or mock for automated tests), with content-addressable
caching/reuse, per-scene progress persistence, and duration telemetry.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from sqlalchemy.orm import object_session

from app.config import settings
from app.models import Video, VideoStep
from app.voiceover import (
    BaseVoiceoverProvider,
    VoiceoverConfigError,
    VoiceoverError,
    get_voiceover_provider,
)


class PipelineError(Exception):
    """Raised when preconditions or step executions fail."""


def _build_output_payload(
    items: list[dict[str, Any]],
    warnings: list[str],
    provider: BaseVoiceoverProvider,
    total_scenes: int,
    voice: str,
    speed: float,
    is_complete: bool = False,
) -> dict[str, Any]:
    """Construct the standardized output payload for the voiceover step."""
    generated_count = sum(1 for it in items if it.get("status") == "ok" and not it.get("reused"))
    reused_count = sum(1 for it in items if it.get("status") == "ok" and it.get("reused"))
    failed_count = sum(1 for it in items if it.get("status") == "failed")

    total_duration = round(
        sum(float(it.get("duration_seconds", 0.0)) for it in items if it.get("status") == "ok"), 3
    )
    total_latency = round(sum(float(it.get("latency_seconds", 0.0)) for it in items), 4)

    payload: dict[str, Any] = {
        "voiceover": {
            "items": [dict(it) for it in items],
            "complete": is_complete,
            "total_duration_seconds": total_duration,
        },
        # Top-level duration_seconds for downstream rendering and backward compatibility
        "duration_seconds": total_duration,
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
            "latency_seconds": total_latency,
        },
    }

    if provider.provider_name == "mock":
        payload["mock"] = True
        payload["meta"]["mock"] = True

    return payload


def _persist_partial(step: VideoStep | None, payload: dict[str, Any]) -> None:
    """Save progress after each audio scene without changing step status."""
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


def generate_voiceover_step(
    video: Video,
    prev: dict[str, Any],
    step: VideoStep | None,
    voiceover_provider: BaseVoiceoverProvider | None = None,
) -> dict[str, Any]:
    """Execute the voiceover step for a video.

    Args:
        video: Video ORM model instance.
        prev: Dictionary containing outputs of preceding pipeline steps.
        step: Current VideoStep instance (for progress persistence).
        voiceover_provider: Optional provider override (e.g. for testing).
    """
    # 1. Validate storyboard input
    sb_outer = prev.get("storyboard")
    if not sb_outer or not isinstance(sb_outer, dict):
        raise PipelineError("Storyboard step must be completed before generating voiceover")

    # Support both wrapped {"storyboard": {...}} and raw {...} formats
    sb = sb_outer.get("storyboard") if isinstance(sb_outer.get("storyboard"), dict) else sb_outer

    scenes = sb.get("scenes") if isinstance(sb, dict) else None
    if not scenes or not isinstance(scenes, list) or len(scenes) == 0:
        raise PipelineError("Storyboard step must be completed before generating voiceover and contain scenes")

    # 2. Extract voice settings from channel or fallback to defaults
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

    # 3. Check for user revision prompt on this step
    warnings: list[str] = []
    if step and step.user_prompt:
        warnings.append(
            "Instructions are not used by the voiceover step. Edit the script or storyboard to change narration text."
        )

    # 4. Resolve provider
    provider = voiceover_provider or get_voiceover_provider()

    # 5. Prepare filesystem storage directory
    storage_root = Path(settings.storage_dir)
    audio_dir = storage_root / str(video.id) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    # 6. Map previously generated items in this step for potential reuse
    existing_items = (
        step.output.get("voiceover", {}).get("items", [])
        if step and isinstance(step.output, dict) and isinstance(step.output.get("voiceover"), dict)
        else []
    )
    existing_by_scene: dict[int, dict[str, Any]] = {
        it["index"]: it for it in existing_items if isinstance(it, dict) and "index" in it
    }

    items: list[dict[str, Any]] = []

    # 7. Sequential generation loop
    for scene in scenes:
        scene_idx = int(scene.get("index", len(items) + 1))
        narration = str(scene.get("narration") or "").strip()

        if not narration:
            items.append(
                {
                    "index": scene_idx,
                    "status": "failed",
                    "file": None,
                    "narration": "",
                    "audio_hash": "",
                    "reused": False,
                    "duration_seconds": 0.0,
                    "sample_rate": 24000,
                    "voice": voice,
                    "speed": speed,
                    "latency_seconds": 0.0,
                    "error": "Scene narration text is empty",
                }
            )
            _persist_partial(
                step,
                _build_output_payload(
                    items, warnings, provider, len(scenes), voice, speed, is_complete=False
                ),
            )
            continue

        # Content hash over narration text, voice, speed, and provider model
        hash_payload = f"{narration}|{voice}|{speed}|{provider.model}".encode("utf-8")
        full_hash = hashlib.sha256(hash_payload).hexdigest()
        hash8 = full_hash[:8]

        file_rel_path = f"{video.id}/audio/scene_{scene_idx:02d}_{hash8}.wav"
        file_abs_path = storage_root / file_rel_path

        # Check reuse criteria: previous item for same scene, status ok, audio_hash matches, file exists
        prev_item = existing_by_scene.get(scene_idx)
        if (
            prev_item
            and prev_item.get("status") == "ok"
            and prev_item.get("audio_hash") == full_hash
            and file_abs_path.is_file()
        ):
            items.append(
                {
                    "index": scene_idx,
                    "status": "ok",
                    "file": file_rel_path,
                    "narration": narration,
                    "audio_hash": full_hash,
                    "reused": True,
                    "duration_seconds": float(prev_item.get("duration_seconds", 0.0)),
                    "sample_rate": int(prev_item.get("sample_rate", 24000)),
                    "voice": prev_item.get("voice", voice),
                    "speed": float(prev_item.get("speed", speed)),
                    "latency_seconds": float(prev_item.get("latency_seconds", 0.0)),
                    "error": None,
                }
            )
            _persist_partial(
                step,
                _build_output_payload(
                    items, warnings, provider, len(scenes), voice, speed, is_complete=False
                ),
            )
            continue

        # Generate new audio
        try:
            est_sec = scene.get("est_seconds")
            result = provider.generate(
                text=narration, voice=voice, speed=speed, target_duration=est_sec
            )
            file_abs_path.write_bytes(result.audio_bytes)

            items.append(
                {
                    "index": scene_idx,
                    "status": "ok",
                    "file": file_rel_path,
                    "narration": narration,
                    "audio_hash": full_hash,
                    "reused": False,
                    "duration_seconds": result.duration_seconds,
                    "sample_rate": result.sample_rate,
                    "voice": voice,
                    "speed": speed,
                    "latency_seconds": result.latency_seconds,
                    "error": None,
                }
            )
        except VoiceoverConfigError as exc:
            # Fatal configuration or dependency error: save partial output and abort immediately
            _persist_partial(
                step,
                _build_output_payload(
                    items, warnings, provider, len(scenes), voice, speed, is_complete=False
                ),
            )
            raise PipelineError(str(exc)) from exc
        except Exception as exc:
            # Scene-specific generation error: record failure and continue with other scenes
            items.append(
                {
                    "index": scene_idx,
                    "status": "failed",
                    "file": None,
                    "narration": narration,
                    "audio_hash": full_hash,
                    "reused": False,
                    "duration_seconds": 0.0,
                    "sample_rate": 24000,
                    "voice": voice,
                    "speed": speed,
                    "latency_seconds": 0.0,
                    "error": str(exc),
                }
            )

        # Persist partial output after each scene
        _persist_partial(
            step,
            _build_output_payload(
                items, warnings, provider, len(scenes), voice, speed, is_complete=False
            ),
        )

    # 8. Verify if any scene failed
    failed_scenes = [it["index"] for it in items if it.get("status") == "failed"]
    if failed_scenes:
        final_payload = _build_output_payload(
            items, warnings, provider, len(scenes), voice, speed, is_complete=False
        )
        _persist_partial(step, final_payload)
        scenes_str = ", ".join(str(s) for s in failed_scenes)
        raise PipelineError(f"Voiceover generation failed for scenes: {scenes_str}")

    # 9. Complete output
    final_payload = _build_output_payload(
        items, warnings, provider, len(scenes), voice, speed, is_complete=True
    )
    _persist_partial(step, final_payload)
    return final_payload
