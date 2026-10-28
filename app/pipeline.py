"""Pipeline engine. Handlers are MOCKS for now (zero AI cost).
To go real, replace a handler with one that calls your LLM/image/TTS provider
and returns a JSON-serialisable dict. Signature: (video, prev_outputs, step) -> dict
"""
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import SessionLocal
from .models import PipelineJob, Video, VideoStep, VideoStepRevision

STEPS = ["script", "storyboard", "images", "voiceover", "render", "subtitles"]


class PipelineError(Exception):
    pass


def _now():
    return datetime.now(timezone.utc)


def _dir(video: Video) -> Path:
    p = Path(settings.storage_dir) / str(video.id)
    p.mkdir(parents=True, exist_ok=True)
    return p


from .steps.script import generate_script_step
from .steps.storyboard import generate_storyboard_step
from .steps.images import generate_images_step
from .steps.voiceover import generate_voiceover_step
from .steps.render import generate_render_step
from .steps.subtitles import generate_subtitles_step


# ---------- handlers ----------
def h_script(video, prev, step):
    return generate_script_step(video, prev, step)


def h_storyboard(video, prev, step):
    return generate_storyboard_step(video, prev, step)


def h_images(video, prev, step):
    return generate_images_step(video, prev, step)


def h_voiceover(video, prev, step):
    return generate_voiceover_step(video, prev, step)


def h_render(video, prev, step):
    return generate_render_step(video, prev, step)


def h_subtitles(video, prev, step):
    return generate_subtitles_step(video, prev, step)


HANDLERS = {
    "script": h_script, "storyboard": h_storyboard, "images": h_images,
    "voiceover": h_voiceover, "render": h_render, "subtitles": h_subtitles,
    "subtitle": h_subtitles,
}


# ---------- engine ----------
def record_revision(
    db: Session,
    step: VideoStep,
    output: dict | None,
    change_summary: str | None = None,
    user_prompt: str | None = None,
) -> VideoStepRevision | None:
    """Save an immutable revision snapshot for a step's output."""
    if not output:
        return None
    current_versions = [r.version for r in (step.revisions or [])]
    next_ver = max(current_versions, default=0) + 1
    rev = VideoStepRevision(
        step_id=step.id,
        version=next_ver,
        output=output,
        user_prompt=user_prompt,
        change_summary=change_summary,
    )
    db.add(rev)
    db.flush()
    return rev


def init_steps(db: Session, video: Video):
    for i, name in enumerate(STEPS):
        db.add(VideoStep(video_id=video.id, name=name, position=i, status="pending"))
    video.status = "in_progress"
    db.commit()
    db.refresh(video)


def claim_next_step(db: Session, video_id) -> VideoStep:
    """Lock the video row, pick the first non-done step, mark it running.
    Prevents advancing if any step is currently awaiting approval."""
    video = db.scalar(
        select(Video).where(Video.id == video_id).with_for_update().execution_options(populate_existing=True)
    )
    if not video.steps:
        raise PipelineError("Video not started yet")
    if any(s.status == "running" for s in video.steps):
        raise PipelineError("A step is already running")
    for s in video.steps:
        if s.status == "awaiting_approval":
            raise PipelineError(f"Step '{s.name}' is awaiting approval before advancing")
    step = next((s for s in video.steps if s.status != "done"), None)
    if not step:
        raise PipelineError("All steps already completed")
    step.status, step.error, step.started_at = "running", None, _now()
    video.status = "in_progress"
    db.commit()
    return step


def execute(video_id, step_name: str, run_all: bool = False):
    """Runs in a background task with its own session (later: move to a worker queue)."""
    with SessionLocal() as db:
        name = step_name
        while name:
            video = db.scalar(
                select(Video).where(Video.id == video_id).execution_options(populate_existing=True)
            )
            step = next(s for s in video.steps if s.name == name)
            prev = {s.name: s.output for s in video.steps if s.status in ("done", "awaiting_approval") and s.output}
            try:
                step.output = HANDLERS[name](video, prev, step)
                step.error = None
            except Exception as exc:  # noqa: BLE001
                step.status, step.error, step.finished_at = "failed", str(exc)[:2000], _now()
                video.status = "failed"
                db.commit()
                return

            step.finished_at = _now()

            # Record version snapshot for this generation
            summary = f"Generation with prompt: {step.user_prompt}" if step.user_prompt else "Initial generation"
            record_revision(
                db=db,
                step=step,
                output=step.output,
                change_summary=summary,
                user_prompt=step.user_prompt,
            )

            # Gatekeeping policy check
            channel_settings = getattr(video.channel, "settings", {}) or {}
            require_approval = channel_settings.get("require_approval", True)
            pause_for_approval = require_approval and (not run_all or channel_settings.get("strict_approval", False))

            if pause_for_approval:
                step.status = "awaiting_approval"
                video.status = "in_progress"
                db.commit()
                break

            step.status = "done"
            nxt = next((s for s in video.steps if s.status != "done"), None)
            video.status = "in_progress" if nxt else "completed"
            if nxt and run_all:
                nxt.status, nxt.error, nxt.started_at = "running", None, _now()
                name = nxt.name
            else:
                name = None
            db.commit()


def approve_step(db: Session, video: Video, step_name: str) -> VideoStep:
    """Approve a step currently awaiting approval and transition it to done."""
    step = next((s for s in video.steps if s.name == step_name), None)
    if not step:
        raise PipelineError(f"Step '{step_name}' not found")
    if step.status != "awaiting_approval":
        raise PipelineError(f"Step '{step_name}' is not awaiting approval (status: {step.status})")

    step.status = "done"
    all_done = all(s.status == "done" for s in video.steps)
    video.status = "completed" if all_done else "in_progress"
    db.commit()
    db.refresh(step)
    return step


def rerun_step(db: Session, video: Video, step_name: str, prompt: str | None):
    """Reset one step (with optional revision prompt) and mark downstream steps stale.
    Cancels any queued jobs and ensures no worker is actively executing."""
    if any(s.status == "running" for s in video.steps):
        raise PipelineError("A step is running; wait for it to finish")

    # Check for active worker jobs
    active_jobs = db.scalars(
        select(PipelineJob).where(
            PipelineJob.video_id == video.id,
            PipelineJob.status.in_(["queued", "running"]),
        )
    ).all()
    if any(j.status == "running" for j in active_jobs):
        raise PipelineError("A step is actively executing; wait for worker to finish before rerunning")
    for j in active_jobs:
        j.status = "cancelled"

    target = next((s for s in video.steps if s.name == step_name), None)
    if not target:
        raise PipelineError("Video not started yet")

    if target.output:
        record_revision(
            db=db,
            step=target,
            output=target.output,
            change_summary=f"Archived before rerun with prompt: '{prompt}'" if prompt else "Archived before rerun",
            user_prompt=target.user_prompt,
        )

    for s in video.steps:
        if s.position == target.position:
            s.status, s.user_prompt, s.error = "pending", prompt, None
        elif s.position > target.position and s.status in ("done", "awaiting_approval"):
            s.status = "stale"
    video.status = "in_progress"
    if target.name not in ("subtitles", "subtitle"):
        video.output_path = None
    db.commit()


def update_step_content(
    db: Session,
    video: Video,
    step_name: str,
    updated_payload: dict,
    change_summary: str | None = None,
) -> VideoStep:
    """Inline revision of a step's output payload (e.g. script narration, scene image prompts).
    Automatically saves revision history, updates output, and marks downstream steps stale."""
    if any(s.status == "running" for s in video.steps):
        raise PipelineError("A step is running; wait for it to finish")

    active_jobs = db.scalars(
        select(PipelineJob).where(
            PipelineJob.video_id == video.id,
            PipelineJob.status.in_(["queued", "running"]),
        )
    ).all()
    if active_jobs:
        raise PipelineError("Cannot modify step while background jobs are queued or executing")

    target = next((s for s in video.steps if s.name == step_name), None)
    if not target:
        raise PipelineError(f"Step '{step_name}' not found")
    if not target.output:
        raise PipelineError("Cannot edit step without existing output")

    # 1. Snapshot the current output into revisions before overwriting
    record_revision(
        db=db,
        step=target,
        output=target.output,
        change_summary="Previous version before inline edit",
        user_prompt=target.user_prompt,
    )

    # 2. Merge payload cleanly
    new_output = dict(target.output)
    for k, v in updated_payload.items():
        if isinstance(v, dict) and isinstance(new_output.get(k), dict):
            new_output[k] = {**new_output[k], **v}
        else:
            new_output[k] = v

    target.output = new_output

    # 3. Snapshot new output
    record_revision(
        db=db,
        step=target,
        output=target.output,
        change_summary=change_summary or "Inline content revision",
        user_prompt=target.user_prompt,
    )

    # 4. Invalidate downstream steps to stale
    for s in video.steps:
        if s.position > target.position and s.status in ("done", "awaiting_approval"):
            s.status = "stale"

    if target.name not in ("subtitles", "subtitle"):
        video.output_path = None

    video.status = "in_progress"
    db.commit()
    db.refresh(target)
    return target


def rollback_step(db: Session, video: Video, step_name: str, target_version: int) -> VideoStep:
    """Roll back a step's output to a previously recorded revision version."""
    if any(s.status == "running" for s in video.steps):
        raise PipelineError("A step is running; wait for it to finish")

    active_jobs = db.scalars(
        select(PipelineJob).where(
            PipelineJob.video_id == video.id,
            PipelineJob.status.in_(["queued", "running"]),
        )
    ).all()
    if active_jobs:
        raise PipelineError("Cannot roll back while background jobs are queued or executing")

    target = next((s for s in video.steps if s.name == step_name), None)
    if not target:
        raise PipelineError(f"Step '{step_name}' not found")

    revision = next((r for r in target.revisions if r.version == target_version), None)
    if not revision:
        raise PipelineError(f"Revision version {target_version} not found for step '{step_name}'")

    # 1. Snapshot current before rollback
    if target.output:
        record_revision(
            db=db,
            step=target,
            output=target.output,
            change_summary=f"Snapshot before rollback to version {target_version}",
            user_prompt=target.user_prompt,
        )

    # 2. Restore revision
    target.output = revision.output
    target.user_prompt = revision.user_prompt

    # 3. Snapshot rollback action
    record_revision(
        db=db,
        step=target,
        output=target.output,
        change_summary=f"Rolled back to version {target_version}",
        user_prompt=target.user_prompt,
    )

    # 4. Invalidate downstream steps
    for s in video.steps:
        if s.position > target.position and s.status in ("done", "awaiting_approval"):
            s.status = "stale"

    if target.name not in ("subtitles", "subtitle"):
        video.output_path = None

    video.status = "in_progress"
    db.commit()
    db.refresh(target)
    return target


