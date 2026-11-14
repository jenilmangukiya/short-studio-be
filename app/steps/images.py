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
    planned_est_cost: float | None = None,
    planned_unique_images: int | None = None,
    generation_attempts: int = 0,
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
    total_cost = round(cost_sum, 4) if has_cost else (round(planned_est_cost, 4) if planned_est_cost is not None else None)

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
            "images_total": planned_unique_images if planned_unique_images is not None else total_scenes,
            "images_generated": generated_count,
            "images_reused": reused_count,
            "images_failed": failed_count,
            "est_cost_usd": total_cost,
            "planned_est_cost_usd": round(planned_est_cost, 4) if planned_est_cost is not None else None,
            "planned_unique_images": planned_unique_images if planned_unique_images is not None else total_scenes,
            "generation_attempts": generation_attempts,
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

    # 8. Pre-flight budget check: verify estimated cost against configurable budget
    unique_image_keys: set[str] = set()
    for sc in scenes:
        sc_idx = str(sc.get("index", ""))
        beats = sc.get("visual_beats") or []
        if beats:
            for b in beats:
                b_strat = b.get("image_strategy", "new_image")
                b_source = b.get("source_image_id")
                if b_strat == "new_image" or not b_source:
                    beat_key = b.get("id") or f"beat_{sc_idx}_{b.get('order', 1)}"
                    unique_image_keys.add(str(beat_key))
                else:
                    unique_image_keys.add(str(b_source).strip())
        else:
            sid = sc.get("source_image_id")
            if not sid or str(sid).strip() == sc_idx:
                unique_image_keys.add(sc_idx)
            else:
                unique_image_keys.add(str(sid).strip())

    unit_cost = (
        provider.est_cost_usd
        if provider.est_cost_usd is not None
        else getattr(settings, "openai_image_est_cost_usd", None) or 0.04
    )
    planned_est_cost = round(len(unique_image_keys) * unit_cost, 4)

    # Hard safety ceiling check (initial generous limit: 50 unique images per video)
    max_unique_ceiling = getattr(settings, "image_max_unique_per_video", 50)
    if len(unique_image_keys) > max_unique_ceiling:
        raise PipelineError(
            f"Video requires {len(unique_image_keys)} unique images, exceeding the hard safety ceiling of {max_unique_ceiling} unique images per video."
        )

    # Informational cost guidance: ordinary monetary budget does not block generation
    channel = getattr(video, "channel", None)
    channel_settings = getattr(channel, "settings", {}) or {}
    budget_limit = channel_settings.get("image_budget_usd") if isinstance(channel_settings, dict) else None
    if budget_limit is None:
        budget_limit = getattr(settings, "image_budget_usd", 0.50)

    if budget_limit is not None and planned_est_cost > float(budget_limit):
        warnings.append(
            f"Estimated image generation cost (${planned_est_cost:.2f} for {len(unique_image_keys)} unique images) "
            f"exceeds standard budget guideline (${float(budget_limit):.2f}). Proceeding under safety ceiling ({max_unique_ceiling} unique images)."
        )

    # Generation attempts tracking against retry safety limit
    max_attempts = getattr(settings, "image_max_generation_attempts", 75)
    generation_attempts = 0

    def _make_payload(complete: bool = False) -> dict[str, Any]:
        return _build_output_payload(
            items=items,
            warnings=warnings,
            provider=provider,
            total_scenes=len(scenes),
            is_complete=complete,
            planned_est_cost=planned_est_cost,
            planned_unique_images=len(unique_image_keys),
            generation_attempts=generation_attempts,
        )

    # Show estimated cost before generation commences
    _persist_partial(step, _make_payload(complete=False))

    # 9. Sequential generation loop
    for idx, scene in enumerate(scenes):
        scene_idx = int(scene.get("index", len(items) + 1))
        prev_scene = scenes[idx - 1] if idx > 0 else None
        next_scene = scenes[idx + 1] if idx + 1 < len(scenes) else None
        beats = scene.get("visual_beats") or []
        primary_beat_id = beats[0].get("id", f"beat_{scene_idx}_1") if beats else f"scene_{scene_idx}"

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

        # Check explicit cross-scene image reuse criteria (no automatic inference from beats)
        source_id = scene.get("source_image_id")

        if source_id and str(source_id).strip() != str(scene_idx):
            sid_str = str(source_id).strip()
            ref_item = next(
                (it for it in items if str(it.get("index")) == sid_str and it.get("status") == "ok" and it.get("file")),
                None,
            )
            if ref_item:
                items.append(
                    {
                        "index": scene_idx,
                        "id": primary_beat_id,
                        "beat_id": primary_beat_id,
                        "status": "ok",
                        "file": ref_item["file"],
                        "prompt": ref_item["prompt"],
                        "prompt_hash": ref_item["prompt_hash"],
                        "reused": True,
                        "source_image_id": sid_str,
                        "model": ref_item.get("model", provider.model),
                        "size": ref_item.get("size", size),
                        "quality": ref_item.get("quality", quality),
                        "latency_seconds": 0.0,
                        "est_cost_usd": 0.0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "total_tokens": 0,
                        "error": None,
                    }
                )
                _persist_partial(step, _make_payload(complete=False))
                continue
            else:
                raise PipelineError(f"Explicitly referenced source_image_id '{sid_str}' for scene {scene_idx} was not found.")

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
                    "id": primary_beat_id,
                    "beat_id": primary_beat_id,
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
            _persist_partial(step, _make_payload(complete=False))
        else:
            # Generate new image for primary scene / beat 1
            generation_attempts += 1
            if generation_attempts > max_attempts:
                _persist_partial(step, _make_payload(complete=False))
                raise PipelineError(
                    f"Image generation exceeded maximum attempt safety limit ({max_attempts}). Aborted to prevent infinite loops."
                )
            try:
                result = provider.generate(prompt=prompt_text, size=size, quality=quality)
                file_abs_path.write_bytes(result.image_bytes)

                items.append(
                    {
                        "index": scene_idx,
                        "id": primary_beat_id,
                        "beat_id": primary_beat_id,
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
                _persist_partial(step, _make_payload(complete=False))
                raise PipelineError(str(exc)) from exc
            except ImageRateLimitError as exc:
                _persist_partial(step, _make_payload(complete=False))
                raise PipelineError(f"Image generation rate limit exceeded: {exc}") from exc
            except ImageContentBlockedError as exc:
                items.append(
                    {
                        "index": scene_idx,
                        "id": primary_beat_id,
                        "beat_id": primary_beat_id,
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
                items.append(
                    {
                        "index": scene_idx,
                        "id": primary_beat_id,
                        "beat_id": primary_beat_id,
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

            _persist_partial(step, _make_payload(complete=False))

        # Generate unique images for any additional visual beats in this scene
        if len(beats) > 1:
            for b_idx, beat in enumerate(beats[1:], start=2):
                b_id = str(beat.get("id") or f"beat_{scene_idx}_{b_idx}")
                b_strat = beat.get("image_strategy", "new_image")
                b_source = beat.get("source_image_id")

                if b_strat != "new_image" and b_source and str(b_source).strip() != str(scene_idx):
                    ref_sid = str(b_source).strip()
                    ref_it = next(
                        (it for it in items if (str(it.get("index")) == ref_sid or str(it.get("id")) == ref_sid) and it.get("status") == "ok" and it.get("file")),
                        None,
                    )
                    if ref_it:
                        items.append(
                            {
                                "index": scene_idx,
                                "id": b_id,
                                "beat_id": b_id,
                                "order": b_idx,
                                "status": "ok",
                                "file": ref_it["file"],
                                "prompt": ref_it["prompt"],
                                "prompt_hash": ref_it["prompt_hash"],
                                "reused": True,
                                "source_image_id": ref_sid,
                                "model": ref_it.get("model", provider.model),
                                "size": ref_it.get("size", size),
                                "quality": ref_it.get("quality", quality),
                                "latency_seconds": 0.0,
                                "est_cost_usd": 0.0,
                                "input_tokens": 0,
                                "output_tokens": 0,
                                "total_tokens": 0,
                                "error": None,
                            }
                        )
                        _persist_partial(step, _make_payload(complete=False))
                        continue
                    else:
                        raise PipelineError(f"Image reference '{ref_sid}' for beat '{b_id}' not found.")

                # Unique image generation for this visual beat
                beat_ctx = {
                    **scene,
                    "image_prompt": beat.get("visual_description") or scene.get("image_prompt"),
                    "shot_type": beat.get("shot_type") or scene.get("shot_type"),
                    "camera_angle": beat.get("camera_angle") or scene.get("camera_angle"),
                    "focal_point": beat.get("focal_point") or scene.get("focal_point"),
                    "visual_technique": beat.get("visual_technique") or scene.get("visual_technique"),
                    "visual_purpose": beat.get("visual_purpose") or scene.get("visual_purpose"),
                }
                b_prompt = compose_final_prompt(
                    scene=beat_ctx,
                    subject_anchor=subject_anchor,
                    visual_style=visual_style,
                    environment_anchor=environment_anchor,
                    video_topic=video_topic,
                    video_title=video_title,
                )
                b_hash_payload = f"{b_prompt}|{provider.model}|{size}|{quality}".encode("utf-8")
                b_full_hash = hashlib.sha256(b_hash_payload).hexdigest()
                b_hash8 = b_full_hash[:8]

                b_file_rel = f"{video.id}/images/scene_{scene_idx:02d}_beat_{b_idx:02d}_{b_hash8}.png"
                b_file_abs = storage_root / b_file_rel

                # Check SHA256 reuse for this beat
                prev_beat_it = next(
                    (it for it in existing_items if it.get("id") == b_id and it.get("prompt_hash") == b_full_hash and (storage_root / it["file"]).is_file()),
                    None,
                )
                if prev_beat_it:
                    items.append(
                        {
                            "index": scene_idx,
                            "id": b_id,
                            "beat_id": b_id,
                            "order": b_idx,
                            "status": "ok",
                            "file": prev_beat_it["file"],
                            "prompt": b_prompt,
                            "prompt_hash": b_full_hash,
                            "reused": True,
                            "model": prev_beat_it.get("model", provider.model),
                            "size": prev_beat_it.get("size", size),
                            "quality": prev_beat_it.get("quality", quality),
                            "latency_seconds": float(prev_beat_it.get("latency_seconds", 0.0)),
                            "est_cost_usd": prev_beat_it.get("est_cost_usd"),
                            "input_tokens": prev_beat_it.get("input_tokens"),
                            "output_tokens": prev_beat_it.get("output_tokens"),
                            "total_tokens": prev_beat_it.get("total_tokens"),
                            "error": None,
                        }
                    )
                else:
                    generation_attempts += 1
                    if generation_attempts > max_attempts:
                        _persist_partial(step, _make_payload(complete=False))
                        raise PipelineError(
                            f"Image generation exceeded maximum attempt safety limit ({max_attempts}). Aborted to prevent infinite loops."
                        )
                    try:
                        b_result = provider.generate(prompt=b_prompt, size=size, quality=quality)
                        b_file_abs.write_bytes(b_result.image_bytes)
                        items.append(
                            {
                                "index": scene_idx,
                                "id": b_id,
                                "beat_id": b_id,
                                "order": b_idx,
                                "status": "ok",
                                "file": b_file_rel,
                                "prompt": b_prompt,
                                "prompt_hash": b_full_hash,
                                "reused": False,
                                "model": provider.model,
                                "size": size,
                                "quality": quality,
                                "latency_seconds": b_result.latency_seconds,
                                "est_cost_usd": b_result.est_cost_usd,
                                "input_tokens": b_result.input_tokens,
                                "output_tokens": b_result.output_tokens,
                                "total_tokens": b_result.total_tokens,
                                "error": None,
                            }
                        )
                    except (ImageConfigError, ImageRateLimitError):
                        raise
                    except Exception as exc:
                        items.append(
                            {
                                "index": scene_idx,
                                "id": b_id,
                                "beat_id": b_id,
                                "order": b_idx,
                                "status": "failed",
                                "file": None,
                                "prompt": b_prompt,
                                "prompt_hash": b_full_hash,
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

                _persist_partial(step, _make_payload(complete=False))

    # 9. Verify if any scene failed
    failed_scenes = [it["index"] for it in items if it.get("status") == "failed"]
    if failed_scenes:
        final_payload = _make_payload(complete=False)
        _persist_partial(step, final_payload)
        scenes_str = ", ".join(str(s) for s in failed_scenes)
        raise PipelineError(f"Image generation failed for scenes: {scenes_str}")

    # 10. Complete output
    final_payload = _make_payload(complete=True)
    _persist_partial(step, final_payload)
    return final_payload
