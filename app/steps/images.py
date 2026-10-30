"""Images step implementation for Shorts Studio.

Generates vertical images for each scene in a storyboard sequentially,
utilizing prompt composition, content-addressable caching/reuse,
per-image persistence for safety, and detailed telemetry metadata.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from sqlalchemy.orm import object_session

from app.config import settings
from app.images import (
    BaseImageProvider,
    ImageConfigError,
    ImageContentBlockedError,
    ImageError,
    ImageRateLimitError,
    get_image_provider,
)
from app.models import Video, VideoStep
from app.steps.prompt_compose import compose_final_prompt


class PipelineError(Exception):
    """Raised when preconditions or step executions fail."""


def _build_output_payload(
    items: list[dict[str, Any]],
    warnings: list[str],
    provider: BaseImageProvider,
    total_scenes: int,
    is_complete: bool = False,
) -> dict[str, Any]:
    """Construct the standardized output payload for the images step."""
    generated_count = sum(1 for it in items if it.get("status") == "ok" and not it.get("reused"))
    reused_count = sum(1 for it in items if it.get("status") == "ok" and it.get("reused"))
    failed_count = sum(1 for it in items if it.get("status") == "failed")

    # Sum estimated cost for newly generated images if configured
    cost_sum = 0.0
    has_cost = False
    for it in items:
        if it.get("status") == "ok" and not it.get("reused") and it.get("est_cost_usd") is not None:
            cost_sum += float(it["est_cost_usd"])
            has_cost = True
    total_cost = round(cost_sum, 4) if has_cost else None

    # Sum token metrics if returned by the provider
    has_tokens = False
    in_tokens_sum = 0
    out_tokens_sum = 0
    tot_tokens_sum = 0
    for it in items:
        if it.get("input_tokens") is not None:
            has_tokens = True
            in_tokens_sum += it.get("input_tokens") or 0
            out_tokens_sum += it.get("output_tokens") or 0
            tot_tokens_sum += it.get("total_tokens") or 0

    total_latency = round(sum(float(it.get("latency_seconds", 0.0)) for it in items), 4)

    payload: dict[str, Any] = {
        "images": {
            "items": [dict(it) for it in items],
            "complete": is_complete,
        },
        "warnings": list(warnings),
        "meta": {
            "provider": provider.provider_name,
            "model": provider.model,
            "task": "images",
            "images_total": total_scenes,
            "images_generated": generated_count,
            "images_reused": reused_count,
            "images_failed": failed_count,
            "est_cost_usd": total_cost,
            "input_tokens": in_tokens_sum if has_tokens else None,
            "output_tokens": out_tokens_sum if has_tokens else None,
            "total_tokens": tot_tokens_sum if has_tokens else None,
            "latency_seconds": total_latency,
        },
    }

    if provider.provider_name == "mock":
        payload["mock"] = True
        payload["meta"]["mock"] = True

    return payload



def _persist_partial(step: VideoStep | None, payload: dict[str, Any]) -> None:
    """Save progress after each image without changing step status."""
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



def generate_images_step(
    video: Video,
    prev: dict[str, Any],
    step: VideoStep | None,
    image_provider: BaseImageProvider | None = None,
) -> dict[str, Any]:
    """Execute the images step for a video.

    Args:
        video: Video ORM model instance.
        prev: Dictionary containing outputs of preceding pipeline steps.
        step: Current VideoStep instance (for progress persistence).
        image_provider: Optional provider override (e.g. for testing).
    """
    # 1. Validate storyboard input
    sb_outer = prev.get("storyboard")
    if not sb_outer or not isinstance(sb_outer, dict):
        raise PipelineError("Storyboard step must be completed before generating images")

    # Support both wrapped {"storyboard": {...}} and raw {...} formats
    sb = sb_outer.get("storyboard") if isinstance(sb_outer.get("storyboard"), dict) else sb_outer

    scenes = sb.get("scenes") if isinstance(sb, dict) else None
    if not scenes or not isinstance(scenes, list) or len(scenes) == 0:
        raise PipelineError("Storyboard step must be completed before generating images and contain scenes")

    # 2. Maximum scenes guard
    if len(scenes) > settings.image_max_per_video:
        raise PipelineError(
            f"Video contains {len(scenes)} scenes, exceeding the maximum allowed ({settings.image_max_per_video})"
        )

    # 3. Check for ignored user revision prompt on this step
    warnings: list[str] = []
    if step and step.user_prompt:
        warnings.append("Instructions are not used by the images step. Edit the storyboard prompt instead.")

    # 4. Resolve provider, size, quality
    provider = image_provider or get_image_provider()
    size = settings.openai_image_size or "1024x1536"
    quality = settings.openai_image_quality or "low"

    # 5. Extract style and anchor for prompt composition
    visual_style = sb.get("visual_style")
    subject_anchor = sb.get("subject_anchor")
    environment_anchor = sb.get("environment_anchor")
    video_topic = getattr(video, "topic", None)
    video_title = getattr(video, "title", None)

    # 6. Prepare filesystem storage directory
    storage_root = Path(settings.storage_dir)
    images_dir = storage_root / str(video.id) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 7. Map previously generated items in this step for potential reuse
    existing_items = (
        step.output.get("images", {}).get("items", [])
        if step and isinstance(step.output, dict) and isinstance(step.output.get("images"), dict)
        else []
    )
    existing_by_scene: dict[int, dict[str, Any]] = {
        it["index"]: it for it in existing_items if isinstance(it, dict) and "index" in it
    }

    items: list[dict[str, Any]] = []

    # 8. Sequential generation loop
    for idx, scene in enumerate(scenes):
        scene_idx = int(scene.get("index", len(items) + 1))
        prev_scene = scenes[idx - 1] if idx > 0 else None
        next_scene = scenes[idx + 1] if idx + 1 < len(scenes) else None

        # Dynamically compose exact final prompt at generation time
        prompt_text = compose_final_prompt(
            scene=scene,
            subject_anchor=subject_anchor,
            visual_style=visual_style,
            environment_anchor=environment_anchor,
            video_topic=video_topic,
            video_title=video_title,
            prev_scene=prev_scene,
            next_scene=next_scene,
        )

        # Full SHA256 prompt hash over exact prompt, provider model, size, and quality
        hash_payload = f"{prompt_text}|{provider.model}|{size}|{quality}".encode("utf-8")
        full_hash = hashlib.sha256(hash_payload).hexdigest()
        hash8 = full_hash[:8]

        file_rel_path = f"{video.id}/images/scene_{scene_idx:02d}_{hash8}.png"
        file_abs_path = storage_root / file_rel_path

        # Check reuse criteria: previous item for same scene, status ok, prompt_hash matches, file exists
        prev_item = existing_by_scene.get(scene_idx)
        if (
            prev_item
            and prev_item.get("status") == "ok"
            and prev_item.get("prompt_hash") == full_hash
            and file_abs_path.is_file()
        ):
            items.append(
                {
                    "index": scene_idx,
                    "status": "ok",
                    "file": file_rel_path,
                    "prompt": prompt_text,
                    "prompt_hash": full_hash,
                    "reused": True,
                    "model": prev_item.get("model", provider.model),
                    "size": prev_item.get("size", size),
                    "quality": prev_item.get("quality", quality),
                    "latency_seconds": float(prev_item.get("latency_seconds", 0.0)),
                    "est_cost_usd": prev_item.get("est_cost_usd"),
                    "input_tokens": prev_item.get("input_tokens"),
                    "output_tokens": prev_item.get("output_tokens"),
                    "total_tokens": prev_item.get("total_tokens"),
                    "error": None,
                }
            )
            # Persist progress
            _persist_partial(step, _build_output_payload(items, warnings, provider, len(scenes), is_complete=False))
            continue

        # Generate new image
        try:
            result = provider.generate(prompt=prompt_text, size=size, quality=quality)
            file_abs_path.write_bytes(result.image_bytes)

            items.append(
                {
                    "index": scene_idx,
                    "status": "ok",
                    "file": file_rel_path,
                    "prompt": prompt_text,
                    "prompt_hash": full_hash,
                    "reused": False,
                    "model": provider.model,
                    "size": size,
                    "quality": quality,
                    "latency_seconds": result.latency_seconds,
                    "est_cost_usd": result.est_cost_usd,
                    "input_tokens": result.input_tokens,
                    "output_tokens": result.output_tokens,
                    "total_tokens": result.total_tokens,
                    "error": None,
                }
            )
        except ImageConfigError as exc:
            # Fatal authentication or permission error: abort run immediately
            _persist_partial(step, _build_output_payload(items, warnings, provider, len(scenes), is_complete=False))
            raise PipelineError(str(exc)) from exc
        except ImageRateLimitError as exc:
            # Rate limit exceeded after retries exhausted: abort run and keep partial output
            _persist_partial(step, _build_output_payload(items, warnings, provider, len(scenes), is_complete=False))
            raise PipelineError(f"Image generation rate limit exceeded: {exc}") from exc
        except ImageContentBlockedError as exc:
            # Content blocked: scene-specific failure, continue other scenes
            items.append(
                {
                    "index": scene_idx,
                    "status": "failed",
                    "file": None,
                    "prompt": prompt_text,
                    "prompt_hash": full_hash,
                    "reused": False,
                    "model": provider.model,
                    "size": size,
                    "quality": quality,
                    "latency_seconds": 0.0,
                    "est_cost_usd": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "error": str(exc),
                }
            )
        except Exception as exc:
            # Other scene-specific error: mark scene failed and continue
            items.append(
                {
                    "index": scene_idx,
                    "status": "failed",
                    "file": None,
                    "prompt": prompt_text,
                    "prompt_hash": full_hash,
                    "reused": False,
                    "model": provider.model,
                    "size": size,
                    "quality": quality,
                    "latency_seconds": 0.0,
                    "est_cost_usd": None,
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                    "error": str(exc),
                }
            )

        # Persist partial output after each image
        _persist_partial(step, _build_output_payload(items, warnings, provider, len(scenes), is_complete=False))

    # 9. Verify if any scene failed
    failed_scenes = [it["index"] for it in items if it.get("status") == "failed"]
    if failed_scenes:
        final_payload = _build_output_payload(items, warnings, provider, len(scenes), is_complete=False)
        _persist_partial(step, final_payload)
        scenes_str = ", ".join(str(s) for s in failed_scenes)
        raise PipelineError(f"Image generation failed for scenes: {scenes_str}")

    # 10. Complete output
    final_payload = _build_output_payload(items, warnings, provider, len(scenes), is_complete=True)
    _persist_partial(step, final_payload)
    return final_payload
