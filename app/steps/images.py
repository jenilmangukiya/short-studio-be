"""Images step implementation for Shorts Studio.

Generates vertical images for each scene in a storyboard sequentially,
utilizing prompt composition, content-addressable caching/reuse,
per-image persistence for safety, and detailed telemetry metadata.
"""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import uuid

from sqlalchemy import select
from sqlalchemy.orm import object_session

from app.config import settings
from app.db import acquire_video_publication_lock
from app.images import (
    BaseImageProvider,
    ImageConfigError,
    ImageContentBlockedError,
    ImageError,
    ImageRateLimitError,
    get_image_provider,
)
from app.models import PipelineJob, Video, VideoStep
from app.steps.prompt_compose import compose_final_prompt
from app.styles import (
    ReferenceMetadata,
    VisualStylePreset,
    canonical_snapshot_hash,
    get_style,
    resolve_visual_style_preset,
    select_scene_references,
)


class PipelineError(Exception):
    """Raised when preconditions or step executions fail."""


def hash_reference_images(reference_images: list[bytes] | None) -> str:
    """Return a deterministic 64-char hex SHA-256 hash of ordered reference image bytes.

    Produces '' if reference_images is empty or None.
    Preserves exact reference ordering and image byte contents.
    """
    if not reference_images:
        return ""
    hasher = hashlib.sha256()
    for img_bytes in reference_images:
        if isinstance(img_bytes, (bytes, bytearray)):
            item_hash = hashlib.sha256(img_bytes).hexdigest()
            hasher.update(f"{len(img_bytes)}:{item_hash};".encode("utf-8"))
    return hasher.hexdigest()


def compute_image_cache_key(
    prompt: str,
    provider_name: str,
    model: str,
    size: str,
    quality: str,
    visual_style_id: str | None = None,
    visual_style_version: int | None = None,
    visual_style_snapshot: dict[str, Any] | None = None,
    reference_images: list[bytes] | None = None,
) -> str:
    """Compute the authoritative 64-char hex SHA-256 image cache key.

    Accounts for all 9 canonical dimensions:
    1. Composed prompt text
    2. Visual style ID
    3. Visual style version
    4. Stable canonical hash of frozen style snapshot
    5. Provider name
    6. Provider model
    7. Image size
    8. Image quality
    9. Content hash of ordered selected reference images
    """
    prompt_clean = prompt.strip() if prompt else ""
    snap_hash = canonical_snapshot_hash(visual_style_snapshot)
    ref_hash = hash_reference_images(reference_images)
    style_id_str = str(visual_style_id or "")
    style_ver_str = str(visual_style_version if visual_style_version is not None else "")
    prov_name_str = str(provider_name or "")
    model_str = str(model or "")
    size_str = str(size or "")
    qual_str = str(quality or "")

    payload = (
        f"prompt={prompt_clean}|"
        f"style_id={style_id_str}|"
        f"style_ver={style_ver_str}|"
        f"snap_hash={snap_hash}|"
        f"provider={prov_name_str}|"
        f"model={model_str}|"
        f"size={size_str}|"
        f"quality={qual_str}|"
        f"ref_hash={ref_hash}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_output_payload(
    items: list[dict[str, Any]],
    warnings: list[str],
    provider: BaseImageProvider,
    total_scenes: int,
    is_complete: bool = False,
    planned_est_cost: float | None = None,
    planned_unique_images: int | None = None,
    generation_attempts: int = 0,
    visual_style_id: str | None = None,
    visual_style_version: int | None = None,
    visual_style_snapshot_hash: str | None = None,
    execution_id: str | None = None,
    job_id: str | None = None,
    lease_token: str | None = None,
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
            "visual_style_id": visual_style_id,
            "visual_style_version": visual_style_version,
            "visual_style_snapshot_hash": visual_style_snapshot_hash,
            "execution_id": execution_id,
            "job_id": str(job_id) if job_id else None,
            "lease_token": str(lease_token) if lease_token else None,
        },
    }

    if provider.provider_name == "mock":
        payload["mock"] = True
        payload["meta"]["mock"] = True

    return payload


def _is_image_step_stale(
    step: VideoStep | None,
    initial_snapshot_hash: str | None = None,
    initial_style_id: str | None = None,
    execution_id: str | None = None,
    job_id: uuid.UUID | str | None = None,
    lease_token: uuid.UUID | str | None = None,
) -> bool:
    """Check if the image generation step has become stale or superseded concurrently."""
    if step is None:
        return False
    if getattr(step, "status", None) == "stale":
        return True

    if job_id is None and hasattr(step, "_job_id"):
        job_id = getattr(step, "_job_id")
    if lease_token is None and hasattr(step, "_lease_token"):
        lease_token = getattr(step, "_lease_token")

    sess = None
    try:
        sess = object_session(step)
    except Exception:
        sess = None

    if sess is not None:
        try:
            vid = sess.get(Video, step.video_id)
            if vid is not None:
                curr_hash = canonical_snapshot_hash(getattr(vid, "visual_style_snapshot", None))
                curr_id = getattr(vid, "visual_style_id", None)
                if (initial_snapshot_hash is not None and curr_hash != initial_snapshot_hash) or (
                    initial_style_id is not None and curr_id != initial_style_id
                ):
                    return True

            curr_step = sess.get(VideoStep, step.id) if hasattr(step, "id") else None
            if curr_step:
                try:
                    sess.refresh(curr_step)
                except Exception:
                    pass
                if isinstance(curr_step.output, dict):
                    curr_meta = curr_step.output.get("meta", {})
                    curr_exec = curr_meta.get("execution_id")
                    if execution_id is not None and curr_exec is not None and curr_exec != str(execution_id):
                        return True
                    curr_lease = curr_meta.get("lease_token")
                    if lease_token is not None and curr_lease is not None and str(curr_lease) != str(lease_token):
                        return True
                    curr_job_id = curr_meta.get("job_id")
                    if job_id is not None and curr_job_id is not None and str(curr_job_id) != str(job_id):
                        return True

            # If job_id is tracked, verify job is still running and lease has not been revoked
            if job_id is not None:
                parsed_id = uuid.UUID(str(job_id)) if isinstance(job_id, str) else job_id
                job = sess.get(PipelineJob, parsed_id)
                if job is not None:
                    if job.status != "running":
                        return True
                    if lease_token is not None and str(job.lease_token) != str(lease_token):
                        return True
                    now = datetime.now(timezone.utc)
                    if job.lease_expires_at is not None:
                        exp = job.lease_expires_at
                        if exp.tzinfo is None:
                            exp = exp.replace(tzinfo=timezone.utc)
                        if exp < now:
                            return True
        except Exception:
            pass

    if isinstance(getattr(step, "output", None), dict):
        meta = step.output.get("meta", {})
        existing_hash = meta.get("visual_style_snapshot_hash")
        if existing_hash is not None and initial_snapshot_hash is not None and existing_hash != initial_snapshot_hash:
            return True
        existing_exec = meta.get("execution_id")
        if execution_id is not None and existing_exec is not None and existing_exec != str(execution_id):
            return True
        existing_lease = meta.get("lease_token")
        if lease_token is not None and existing_lease is not None and str(existing_lease) != str(lease_token):
            return True
        existing_job = meta.get("job_id")
        if job_id is not None and existing_job is not None and str(existing_job) != str(job_id):
            return True
    return False


def _promote_image_bytes(
    dest_path: Path,
    image_bytes: bytes,
    step: VideoStep | None,
    initial_snapshot_hash: str | None = None,
    initial_style_id: str | None = None,
    execution_id: str | None = None,
    job_id: uuid.UUID | str | None = None,
    lease_token: uuid.UUID | str | None = None,
    _pre_promote_hook: Any | None = None,
) -> bool:
    """Atomically write and promote generated image bytes under database publication lock."""
    if _is_image_step_stale(step, initial_snapshot_hash, initial_style_id, execution_id, job_id, lease_token):
        return False
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.parent / f"tmp_{uuid.uuid4().hex[:8]}.png"
    tmp_path.write_bytes(image_bytes)

    if _pre_promote_hook is not None:
        _pre_promote_hook()

    sess = None
    try:
        sess = object_session(step) if step is not None else None
    except Exception:
        sess = None

    if sess is not None:
        try:
            # 1. Lock ordering: Acquire Video publication lock FIRST
            locked_vid = acquire_video_publication_lock(sess, step.video_id)
            if locked_vid is None:
                tmp_path.unlink(missing_ok=True)
                return False

            # 2. Check staleness and lease ownership while holding lock
            if _is_image_step_stale(step, initial_snapshot_hash, initial_style_id, execution_id, job_id, lease_token):
                tmp_path.unlink(missing_ok=True)
                return False

            # 3. Promote staged generation file atomically under lock
            import os
            os.replace(str(tmp_path), str(dest_path))
            return True
        except Exception:
            tmp_path.unlink(missing_ok=True)
            return False

    # Standalone/unattached execution without database session
    if _is_image_step_stale(step, initial_snapshot_hash, initial_style_id, execution_id, job_id, lease_token):
        tmp_path.unlink(missing_ok=True)
        return False
    import os
    os.replace(str(tmp_path), str(dest_path))
    return True


def _persist_partial(
    step: VideoStep | None,
    payload: dict[str, Any],
    initial_snapshot_hash: str | None = None,
    initial_style_id: str | None = None,
    execution_id: str | None = None,
    job_id: uuid.UUID | str | None = None,
    lease_token: uuid.UUID | str | None = None,
) -> None:
    """Save progress after each image without changing step status.
    Aborts save if step has been marked stale, visual style has changed, or execution superseded.
    """
    if step is None or _is_image_step_stale(step, initial_snapshot_hash, initial_style_id, execution_id, job_id, lease_token):
        return

    sess = None
    try:
        sess = object_session(step)
    except Exception:
        sess = None

    if sess is not None:
        try:
            locked_vid = acquire_video_publication_lock(sess, step.video_id)
            if locked_vid is None:
                return

            if _is_image_step_stale(step, initial_snapshot_hash, initial_style_id, execution_id, job_id, lease_token):
                return

            curr_step = next((s for s in (locked_vid.steps or []) if s.name == "images"), None)
            if curr_step is not None:
                curr_step.output = dict(payload)
            step.output = dict(payload)
            sess.commit()
            return
        except Exception:
            sess.rollback()
            return

    if getattr(step, "status", None) == "stale":
        return

    # Check unattached step output for style / execution mismatch
    if isinstance(getattr(step, "output", None), dict):
        meta = step.output.get("meta", {})
        existing_hash = meta.get("visual_style_snapshot_hash")
        if existing_hash is not None and initial_snapshot_hash is not None and existing_hash != initial_snapshot_hash:
            return
        existing_exec = meta.get("execution_id")
        if execution_id is not None and existing_exec is not None and existing_exec != str(execution_id):
            return
        existing_lease = meta.get("lease_token")
        if lease_token is not None and existing_lease is not None and str(existing_lease) != str(lease_token):
            return

    step.output = dict(payload)


def _claim_step_ownership(
    step: VideoStep | None,
    payload: dict[str, Any],
    initial_snapshot_hash: str | None = None,
    initial_style_id: str | None = None,
    execution_id: str | None = None,
    job_id: uuid.UUID | str | None = None,
    lease_token: uuid.UUID | str | None = None,
) -> bool:
    """Durably record this execution as the current owner of the image step before generation starts.
    Verifies that the step is not stale due to visual style change, validates queue lease ownership,
    establishes the new execution identity, and commits to the database under lock so concurrent or older
    executions are immediately invalidated.
    """
    if step is None:
        return False
    if getattr(step, "status", None) == "stale":
        return False

    if job_id is None and hasattr(step, "_job_id"):
        job_id = getattr(step, "_job_id")
    if lease_token is None and hasattr(step, "_lease_token"):
        lease_token = getattr(step, "_lease_token")

    sess = None
    try:
        sess = object_session(step)
    except Exception:
        sess = None

    if sess is not None:
        try:
            # 1. Consistent lock ordering: Acquire Video publication lock FIRST
            locked_vid = acquire_video_publication_lock(sess, step.video_id)
            if locked_vid is None:
                return False

            # Verify visual style hasn't changed
            curr_hash = canonical_snapshot_hash(getattr(locked_vid, "visual_style_snapshot", None))
            curr_id = getattr(locked_vid, "visual_style_id", None)
            if (initial_snapshot_hash is not None and curr_hash != initial_snapshot_hash) or (
                initial_style_id is not None and curr_id != initial_style_id
            ):
                return False

            curr_step = next((s for s in (locked_vid.steps or []) if s.name == "images"), None)
            if curr_step is None or getattr(curr_step, "status", None) == "stale":
                return False

            # 2. Acquire PipelineJob lock SECOND and validate lease
            parsed_job_id = None
            if job_id is not None:
                parsed_job_id = uuid.UUID(str(job_id)) if isinstance(job_id, str) else job_id

            if parsed_job_id is not None:
                job = sess.scalar(
                    select(PipelineJob)
                    .where(PipelineJob.id == parsed_job_id)
                    .with_for_update()
                )
                if not job:
                    return False
                if job.status != "running":
                    return False
                if lease_token is not None and str(job.lease_token) != str(lease_token):
                    return False
                now = datetime.now(timezone.utc)
                if job.lease_expires_at is not None:
                    lease_exp = job.lease_expires_at
                    if lease_exp.tzinfo is None:
                        lease_exp = lease_exp.replace(tzinfo=timezone.utc)
                    if lease_exp < now:
                        return False
                # Reject if a newer job exists
                newer_job = sess.scalar(
                    select(PipelineJob).where(
                        PipelineJob.video_id == step.video_id,
                        PipelineJob.step_name == "images",
                        PipelineJob.id != job.id,
                        PipelineJob.scheduled_at >= job.scheduled_at,
                        PipelineJob.status.in_(["queued", "running", "completed"]),
                    )
                )
                if newer_job is not None:
                    return False
            else:
                # If job_id not explicitly provided, check if there is an active running job
                active_job = sess.scalar(
                    select(PipelineJob).where(
                        PipelineJob.video_id == step.video_id,
                        PipelineJob.step_name == "images",
                        PipelineJob.status == "running",
                    ).with_for_update()
                )
                if active_job is not None and lease_token is not None:
                    if str(active_job.lease_token) != str(lease_token):
                        return False

            # Check if existing step.output already has a claim from a newer/different job or execution
            if isinstance(curr_step.output, dict):
                existing_meta = curr_step.output.get("meta", {})
                existing_job = existing_meta.get("job_id")
                existing_lease = existing_meta.get("lease_token")
                existing_exec = existing_meta.get("execution_id")

                if parsed_job_id is not None and existing_job is not None and str(existing_job) != str(parsed_job_id):
                    # Check if the existing job is newer or currently holding an active lease
                    existing_job_row = sess.get(PipelineJob, uuid.UUID(str(existing_job)))
                    if existing_job_row is not None and parsed_job_id is not None:
                        current_job_row = sess.get(PipelineJob, parsed_job_id)
                        if existing_job_row and current_job_row:
                            if existing_job_row.scheduled_at > current_job_row.scheduled_at:
                                return False
                            # If existing job is still running and its lease is still unexpired:
                            now = datetime.now(timezone.utc)
                            exp = existing_job_row.lease_expires_at
                            if exp and exp.tzinfo is None:
                                exp = exp.replace(tzinfo=timezone.utc)
                            if existing_job_row.status == "running" and exp and exp > now:
                                return False

                elif lease_token is not None and existing_lease is not None and str(existing_lease) != str(lease_token):
                    # For executions without job_id (e.g. manual/direct), reject if another lease was stamped
                    if parsed_job_id is None:
                        return False

            # Stamp metadata into payload
            payload_to_save = dict(payload)
            if "meta" not in payload_to_save or not isinstance(payload_to_save["meta"], dict):
                payload_to_save["meta"] = {}
            payload_meta = dict(payload_to_save["meta"])
            if execution_id:
                payload_meta["execution_id"] = str(execution_id)
            if job_id:
                payload_meta["job_id"] = str(job_id)
            if lease_token:
                payload_meta["lease_token"] = str(lease_token)
            payload_to_save["meta"] = payload_meta

            curr_step.output = payload_to_save
            step.output = payload_to_save
            sess.commit()
            return True
        except Exception:
            sess.rollback()
            return False

    # Standalone/unattached execution
    payload_to_save = dict(payload)
    if "meta" not in payload_to_save or not isinstance(payload_to_save["meta"], dict):
        payload_to_save["meta"] = {}
    payload_meta = dict(payload_to_save["meta"])
    if execution_id:
        payload_meta["execution_id"] = str(execution_id)
    if job_id:
        payload_meta["job_id"] = str(job_id)
    if lease_token:
        payload_meta["lease_token"] = str(lease_token)
    payload_to_save["meta"] = payload_meta
    step.output = payload_to_save
    return True



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

    # Detect if storyboard is connected (8-step pipeline with Video Bible)
    is_connected_storyboard = bool(
        (isinstance(sb, dict) and sb.get("video_bible"))
        or (isinstance(prev, dict) and prev.get("video_bible"))
    )

    # Resolve active visual preset (fail-closed on invalid snapshot, unknown ID, or version mismatch)
    active_preset = resolve_visual_style_preset(
        snapshot=getattr(video, "visual_style_snapshot", None),
        style_id=getattr(video, "visual_style_id", None),
        style_version=getattr(video, "visual_style_version", None),
    )

    initial_style_id = active_preset.id if active_preset else getattr(video, "visual_style_id", None)
    initial_style_version = (
        active_preset.version if active_preset else getattr(video, "visual_style_version", None)
    )
    initial_snapshot_dict = getattr(video, "visual_style_snapshot", None) or (
        active_preset.model_dump() if active_preset else None
    )
    initial_snapshot_hash = canonical_snapshot_hash(initial_snapshot_dict)

    # 6. Prepare filesystem storage directory
    storage_root = Path(settings.storage_dir)
    images_dir = storage_root / str(video.id) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Collect available reference assets from Video Bible and active preset
    available_references: list[ReferenceMetadata] = []
    vb_source = prev.get("video_bible") or (sb.get("video_bible") if isinstance(sb, dict) else None)
    if isinstance(vb_source, dict):
        raw_refs = vb_source.get("reference_assets") or []
        for r in raw_refs:
            try:
                available_references.append(ReferenceMetadata.normalize(r))
            except Exception:
                pass
    elif hasattr(vb_source, "reference_assets") and getattr(vb_source, "reference_assets"):
        for r in vb_source.reference_assets:
            try:
                available_references.append(ReferenceMetadata.normalize(r))
            except Exception:
                pass

    if active_preset and active_preset.assets:
        if active_preset.assets.style_benchmark:
            available_references.append(
                ReferenceMetadata(
                    entity_id="style_benchmark",
                    role="style",
                    asset_path=active_preset.assets.style_benchmark,
                    visual_style_id=active_preset.id,
                    style_version=active_preset.version,
                    description="Preset global style benchmark",
                )
            )
        if active_preset.assets.anatomy_reference:
            available_references.append(
                ReferenceMetadata(
                    entity_id="anatomy_benchmark",
                    role="style",
                    asset_path=active_preset.assets.anatomy_reference,
                    visual_style_id=active_preset.id,
                    style_version=active_preset.version,
                    description="Preset character anatomy turnaround",
                )
            )
        if active_preset.assets.environment_reference:
            available_references.append(
                ReferenceMetadata(
                    entity_id="environment_benchmark",
                    role="style",
                    asset_path=active_preset.assets.environment_reference,
                    visual_style_id=active_preset.id,
                    style_version=active_preset.version,
                    description="Preset environment perspective benchmark",
                )
            )

    def _select_and_load_references(target_ctx: dict[str, Any]) -> tuple[list[bytes], dict[str, Any] | None, list[str]]:
        if not active_preset or not active_preset.capabilities.supports_reference_images:
            return [], None, []
        sel_result = select_scene_references(
            scene=target_ctx,
            preset=active_preset,
            available_references=available_references,
            check_file_exists=True,
            storage_root=storage_root,
        )
        loaded_bytes: list[bytes] = []
        selected_paths: list[str] = []
        for sel in sel_result.selected_references:
            p = Path(sel.asset_path)
            if not p.is_absolute():
                p = storage_root / p
            if p.is_file():
                try:
                    loaded_bytes.append(p.read_bytes())
                    selected_paths.append(sel.asset_path)
                except Exception:
                    pass
        return loaded_bytes, sel_result.model_dump(), selected_paths

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

    execution_id = getattr(step, "_execution_id", None) or uuid.uuid4().hex[:12]
    if step is not None:
        setattr(step, "_execution_id", execution_id)
    job_id = getattr(step, "_job_id", None)
    lease_token = getattr(step, "_lease_token", None)

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
            visual_style_id=initial_style_id,
            visual_style_version=initial_style_version,
            visual_style_snapshot_hash=initial_snapshot_hash,
            execution_id=execution_id,
            job_id=str(job_id) if job_id else None,
            lease_token=str(lease_token) if lease_token else None,
        )

    def _persist(complete: bool = False) -> None:
        _persist_partial(
            step=step,
            payload=_make_payload(complete=complete),
            initial_snapshot_hash=initial_snapshot_hash,
            initial_style_id=initial_style_id,
            execution_id=execution_id,
            job_id=job_id,
            lease_token=lease_token,
        )

    # Durably claim ownership before generation commences
    claimed = _claim_step_ownership(
        step=step,
        payload=_make_payload(complete=False),
        initial_snapshot_hash=initial_snapshot_hash,
        initial_style_id=initial_style_id,
        execution_id=execution_id,
        job_id=job_id,
        lease_token=lease_token,
    )
    if not claimed and step is not None:
        if getattr(step, "status", None) == "stale":
            raise PipelineError(f"Step '{step.name}' became stale due to concurrent visual style change.")
        raise PipelineError(f"Step '{step.name}' execution was superseded or lease expired.")

    # 9. Sequential generation loop
    for idx, scene in enumerate(scenes):
        scene_idx = int(scene.get("index", len(items) + 1))
        prev_scene = scenes[idx - 1] if idx > 0 else None
        next_scene = scenes[idx + 1] if idx + 1 < len(scenes) else None
        beats = scene.get("visual_beats") or []
        primary_beat = beats[0] if beats else None
        primary_beat_id = primary_beat.get("id", f"beat_{scene_idx}_1") if primary_beat else f"scene_{scene_idx}"

        # Consume precomposed final_prompt: beat final_prompt -> scene final_prompt -> legacy fallback
        prompt_text = (
            (primary_beat.get("final_prompt") if isinstance(primary_beat, dict) else getattr(primary_beat, "final_prompt", None))
            or scene.get("final_prompt")
        )

        if not prompt_text or not str(prompt_text).strip():
            if is_connected_storyboard:
                raise PipelineError(
                    f"Missing required final_prompt for scene {scene_idx} (beat '{primary_beat_id}') in connected storyboard."
                )
            # Legacy fallback for older 6-step videos without precomposed prompts
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

        # Select references for this scene BEFORE computing cache key
        ref_bytes, ref_diag, ref_paths = _select_and_load_references(scene)

        # Full SHA256 image cache key over all 9 canonical dimensions
        full_hash = compute_image_cache_key(
            prompt=prompt_text,
            provider_name=provider.provider_name,
            model=provider.model,
            size=size,
            quality=quality,
            visual_style_id=initial_style_id,
            visual_style_version=initial_style_version,
            visual_style_snapshot=initial_snapshot_dict,
            reference_images=ref_bytes,
        )
        hash8 = full_hash[:8]

        gen_filename = f"scene_{scene_idx:02d}_{hash8}_{execution_id[:8]}_{uuid.uuid4().hex[:8]}.png"
        gen_rel_path = f"{video.id}/images/generations/{gen_filename}"
        gen_abs_path = storage_root / gen_rel_path

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
                _persist(complete=False)
                continue
            else:
                raise PipelineError(f"Explicitly referenced source_image_id '{sid_str}' for scene {scene_idx} was not found.")

        # Check reuse criteria: previous item for same scene, status ok, prompt_hash matches, authoritative file exists
        prev_item = existing_by_scene.get(scene_idx)
        prev_file_exists = False
        prev_file_path = None
        if prev_item and prev_item.get("file"):
            if (storage_root / prev_item["file"]).is_file():
                prev_file_exists = True
                prev_file_path = prev_item["file"]

        if (
            prev_item
            and prev_item.get("status") == "ok"
            and prev_item.get("prompt_hash") == full_hash
            and prev_file_exists
        ):
            items.append(
                {
                    "index": scene_idx,
                    "id": primary_beat_id,
                    "beat_id": primary_beat_id,
                    "status": "ok",
                    "file": prev_file_path,
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
            _persist(complete=False)
        else:
            # Generate new image for primary scene / beat 1
            generation_attempts += 1
            if generation_attempts > max_attempts:
                _persist(complete=False)
                raise PipelineError(
                    f"Image generation exceeded maximum attempt safety limit ({max_attempts}). Aborted to prevent infinite loops."
                )
            try:
                result = provider.generate(
                    prompt=prompt_text,
                    size=size,
                    quality=quality,
                    reference_images=ref_bytes if ref_bytes else None,
                )
                promoted = _promote_image_bytes(
                    dest_path=gen_abs_path,
                    image_bytes=result.image_bytes,
                    step=step,
                    initial_snapshot_hash=initial_snapshot_hash,
                    initial_style_id=initial_style_id,
                    execution_id=execution_id,
                    job_id=job_id,
                    lease_token=lease_token,
                    _pre_promote_hook=getattr(step, "_pre_promote_hook", None),
                )
                if not promoted:
                    return _make_payload(complete=False)

                items.append(
                    {
                        "index": scene_idx,
                        "id": primary_beat_id,
                        "beat_id": primary_beat_id,
                        "status": "ok",
                        "file": gen_rel_path,
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
                        "reference_images": ref_paths,
                        "reference_selection": ref_diag,
                        "error": None,
                    }
                )
                _persist(complete=False)
            except ImageConfigError as exc:
                _persist(complete=False)
                raise PipelineError(str(exc)) from exc
            except ImageRateLimitError as exc:
                _persist(complete=False)
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

            _persist(complete=False)

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
                        _persist(complete=False)
                        continue
                    else:
                        raise PipelineError(f"Image reference '{ref_sid}' for beat '{b_id}' not found.")

                # Consume precomposed final_prompt: beat final_prompt -> scene final_prompt -> legacy fallback
                b_prompt = (
                    beat.get("final_prompt")
                    or scene.get("final_prompt")
                )
                if not b_prompt or not str(b_prompt).strip():
                    if is_connected_storyboard:
                        raise PipelineError(
                            f"Missing required final_prompt for beat '{b_id}' in connected storyboard."
                        )
                    # Legacy fallback for older 6-step videos without precomposed beat prompts
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

                # Select references for this beat BEFORE computing cache key
                b_ref_bytes, b_ref_diag, b_ref_paths = _select_and_load_references(beat)

                # Full SHA256 image cache key over all 9 canonical dimensions
                b_full_hash = compute_image_cache_key(
                    prompt=b_prompt,
                    provider_name=provider.provider_name,
                    model=provider.model,
                    size=size,
                    quality=quality,
                    visual_style_id=initial_style_id,
                    visual_style_version=initial_style_version,
                    visual_style_snapshot=initial_snapshot_dict,
                    reference_images=b_ref_bytes,
                )
                b_hash8 = b_full_hash[:8]

                b_gen_filename = f"scene_{scene_idx:02d}_beat_{b_idx:02d}_{b_hash8}_{execution_id[:8]}_{uuid.uuid4().hex[:8]}.png"
                b_gen_rel_path = f"{video.id}/images/generations/{b_gen_filename}"
                b_gen_abs_path = storage_root / b_gen_rel_path

                # Check SHA256 reuse for this beat: authoritative file must exist on disk
                prev_beat_it = next(
                    (
                        it
                        for it in existing_items
                        if it.get("id") == b_id
                        and it.get("prompt_hash") == b_full_hash
                        and it.get("file")
                        and (storage_root / it["file"]).is_file()
                    ),
                    None,
                )
                if prev_beat_it:
                    reused_beat_file = prev_beat_it["file"]
                    items.append(
                        {
                            "index": scene_idx,
                            "id": b_id,
                            "beat_id": b_id,
                            "order": b_idx,
                            "status": "ok",
                            "file": reused_beat_file,
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
                        _persist(complete=False)
                        raise PipelineError(
                            f"Image generation exceeded maximum attempt safety limit ({max_attempts}). Aborted to prevent infinite loops."
                        )
                    try:
                        b_result = provider.generate(
                            prompt=b_prompt,
                            size=size,
                            quality=quality,
                            reference_images=b_ref_bytes if b_ref_bytes else None,
                        )
                        b_promoted = _promote_image_bytes(
                            dest_path=b_gen_abs_path,
                            image_bytes=b_result.image_bytes,
                            step=step,
                            initial_snapshot_hash=initial_snapshot_hash,
                            initial_style_id=initial_style_id,
                            execution_id=execution_id,
                            job_id=job_id,
                            lease_token=lease_token,
                            _pre_promote_hook=getattr(step, "_pre_promote_hook", None),
                        )
                        if not b_promoted:
                            return _make_payload(complete=False)
                        items.append(
                            {
                                "index": scene_idx,
                                "id": b_id,
                                "beat_id": b_id,
                                "order": b_idx,
                                "status": "ok",
                                "file": b_gen_rel_path,
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
                                "reference_images": b_ref_paths,
                                "reference_selection": b_ref_diag,
                                "error": None,
                            }
                        )
                        _persist(complete=False)
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

                _persist(complete=False)

    # 9. Verify if any scene failed
    failed_scenes = [it["index"] for it in items if it.get("status") == "failed"]
    if failed_scenes:
        final_payload = _make_payload(complete=False)
        _persist(complete=False)
        scenes_str = ", ".join(str(s) for s in failed_scenes)
        raise PipelineError(f"Image generation failed for scenes: {scenes_str}")

    # Phase 4B Concurrency Protection: verify step did not become stale or visual style changed
    if step is not None:
        sess = None
        try:
            sess = object_session(step)
        except Exception:
            sess = None
        if sess is not None:
            try:
                vid = sess.get(Video, step.video_id)
                if vid is not None:
                    curr_h = canonical_snapshot_hash(getattr(vid, "visual_style_snapshot", None))
                    curr_i = getattr(vid, "visual_style_id", None)
                    if (
                        step.status == "stale"
                        or (initial_snapshot_hash and curr_h != initial_snapshot_hash)
                        or (initial_style_id and curr_i != initial_style_id)
                    ):
                        raise PipelineError(
                            f"Step '{step.name}' became stale due to concurrent visual style change."
                        )
            except PipelineError:
                raise
            except Exception:
                pass

    # 10. Complete output
    final_payload = _make_payload(complete=True)
    _persist(complete=True)
    return final_payload
