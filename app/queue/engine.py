"""PostgreSQL-backed transactional queue engine using SKIP LOCKED.
Guarantees atomic enqueuing, exactly-once claiming, heartbeat lease tracking,
and dead-letter crash recovery.
"""
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.models import PipelineJob, Video, VideoStep
from app.pipeline import HANDLERS, PipelineError, record_revision


class QueueError(Exception):
    pass


class JobConflictError(QueueError):
    pass


def _now() -> datetime:
    return datetime.now(timezone.utc)


def enqueue_step_job(
    db: Session,
    video_id: uuid.UUID,
    step_name: str,
    run_all: bool = False,
    max_attempts: int = 3,
) -> PipelineJob:
    """Atomically claim a step and enqueue a background execution job.
    Ensures that step status updates and job persistence commit in the exact same transaction."""
    video = db.scalar(
        select(Video).where(Video.id == video_id).with_for_update().execution_options(populate_existing=True)
    )
    if not video or not video.steps:
        raise PipelineError("Video not started yet")

    # Gatekeeping check: cannot enqueue next step if any preceding step is awaiting approval
    for s in video.steps:
        if s.status == "awaiting_approval":
            raise PipelineError(f"Step '{s.name}' is awaiting approval before advancing")

    # Check for active running or queued job for this step or video
    existing_job = db.scalar(
        select(PipelineJob).where(
            PipelineJob.video_id == video_id,
            PipelineJob.step_name == step_name,
            PipelineJob.status.in_(["queued", "running"]),
        )
    )
    if existing_job:
        raise JobConflictError(f"An active job already exists for step '{step_name}' (status: {existing_job.status})")

    target_step = next((s for s in video.steps if s.name == step_name), None)
    if not target_step:
        raise PipelineError(f"Step '{step_name}' not found on video")

    # Atomic state transition
    target_step.status = "running"
    target_step.error = None
    target_step.started_at = _now()
    video.status = "in_progress"

    job = PipelineJob(
        video_id=video_id,
        step_name=step_name,
        status="queued",
        run_all=run_all,
        attempts=0,
        max_attempts=max_attempts,
        scheduled_at=_now(),
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def claim_next_job(
    db: Session,
    worker_id: str,
    lease_duration_seconds: int = 60,
) -> tuple[PipelineJob, uuid.UUID] | None:
    """Atomically claim the next queued job ready for execution using SELECT FOR UPDATE SKIP LOCKED.
    Returns (job, lease_token) where lease_token serves as a fencing token preventing expired
    workers from writing output after their lease has been revoked."""
    now = _now()

    # Query with skip_locked for PostgreSQL; standard with_for_update for SQLite fallback
    is_sqlite = db.bind is not None and db.bind.dialect.name == "sqlite"
    select_stmt = (
        select(PipelineJob)
        .where(
            PipelineJob.status == "queued",
            PipelineJob.scheduled_at <= now,
        )
        .order_by(PipelineJob.scheduled_at.asc())
        .limit(1)
    )

    if not is_sqlite:
        select_stmt = select_stmt.with_for_update(skip_locked=True)
    else:
        select_stmt = select_stmt.with_for_update()

    job = db.scalar(select_stmt)
    if not job:
        return None

    # Issue unique fencing token and extend lease
    lease_token = uuid.uuid4()
    job.status = "running"
    job.attempts += 1
    job.started_at = now
    job.last_heartbeat_at = now
    job.lease_expires_at = now + timedelta(seconds=lease_duration_seconds)
    job.worker_id = worker_id
    job.lease_token = lease_token

    db.commit()
    db.refresh(job)
    return job, lease_token


def renew_job_heartbeat(
    db: Session,
    job_id: uuid.UUID,
    lease_token: uuid.UUID,
    lease_duration_seconds: int = 60,
) -> bool:
    """Atomic heartbeat renewal. Verifies the fencing lease token.
    Returns True if extended; False if the lease expired and was revoked."""
    now = _now()
    stmt = (
        update(PipelineJob)
        .where(
            PipelineJob.id == job_id,
            PipelineJob.lease_token == lease_token,
            PipelineJob.status == "running",
        )
        .values(
            last_heartbeat_at=now,
            lease_expires_at=now + timedelta(seconds=lease_duration_seconds),
        )
    )
    res = db.execute(stmt)
    db.commit()
    return bool(res.rowcount and res.rowcount > 0)


def complete_job(
    db: Session,
    job_id: uuid.UUID,
    lease_token: uuid.UUID,
    output: dict,
) -> bool:
    """Complete a job and apply step status transitions with Phase 4 gatekeeping.
    Fails safely if the worker's lease expired before completion."""
    now = _now()
    job = db.scalar(
        select(PipelineJob)
        .where(
            PipelineJob.id == job_id,
            PipelineJob.lease_token == lease_token,
            PipelineJob.status == "running",
        )
        .with_for_update()
    )
    if not job:
        # Fencing token revoked! An expired lease was recovered by another worker
        return False

    video = db.scalar(
        select(Video).where(Video.id == job.video_id).with_for_update().execution_options(populate_existing=True)
    )
    if not video:
        return False

    step = next((s for s in video.steps if s.name == job.step_name), None)
    if not step:
        return False

    # Apply step output and save version revision
    step.output = output
    step.error = None
    step.finished_at = now

    summary = f"Generation with prompt: {step.user_prompt}" if step.user_prompt else "Initial generation"
    record_revision(
        db=db,
        step=step,
        output=step.output,
        change_summary=summary,
        user_prompt=step.user_prompt,
    )

    # Check Phase 4 gatekeeping policy
    channel_settings = getattr(video.channel, "settings", {}) or {}
    require_approval = channel_settings.get("require_approval", True)
    pause_for_approval = require_approval and (not job.run_all or channel_settings.get("strict_approval", False))

    job.status = "completed"
    job.finished_at = now
    job.lease_token = None
    job.lease_expires_at = None

    if pause_for_approval:
        step.status = "awaiting_approval"
        video.status = "in_progress"
    else:
        step.status = "done"
        nxt = next((s for s in video.steps if s.status != "done"), None)
        video.status = "in_progress" if nxt else "completed"

        # Chain next step if run_all was requested and no approval is needed
        if nxt and job.run_all:
            nxt.status = "running"
            nxt.started_at = now
            nxt.error = None
            next_job = PipelineJob(
                video_id=video.id,
                step_name=nxt.name,
                status="queued",
                run_all=True,
                attempts=0,
                max_attempts=job.max_attempts,
                scheduled_at=now,
            )
            db.add(next_job)

    db.commit()
    return True


def fail_job(
    db: Session,
    job_id: uuid.UUID,
    lease_token: uuid.UUID,
    error_message: str,
    base_retry_delay_seconds: int = 5,
) -> bool:
    """Record job failure with exponential backoff retry or dead-letter transition."""
    now = _now()
    job = db.scalar(
        select(PipelineJob)
        .where(
            PipelineJob.id == job_id,
            PipelineJob.lease_token == lease_token,
            PipelineJob.status == "running",
        )
        .with_for_update()
    )
    if not job:
        return False

    video = db.scalar(select(Video).where(Video.id == job.video_id).with_for_update())
    step = next((s for s in (video.steps if video else []) if s.name == job.step_name), None)

    # Clean error summary: truncate excessive stack traces and sanitize
    clean_error = str(error_message).strip()[:1000]

    if job.attempts < job.max_attempts:
        # Exponential backoff: base_delay * 2^(attempts-1)
        delay = base_retry_delay_seconds * (2 ** (job.attempts - 1))
        job.status = "queued"
        job.scheduled_at = now + timedelta(seconds=delay)
        job.lease_token = None
        job.lease_expires_at = None
        job.error = clean_error
        if step:
            step.error = f"[Attempt {job.attempts}/{job.max_attempts} failed, retrying in {delay}s]: {clean_error}"
    else:
        # Move to Dead Letter Queue
        job.status = "dead_letter"
        job.finished_at = now
        job.lease_token = None
        job.lease_expires_at = None
        job.error = clean_error
        if step:
            step.status = "failed"
            step.error = f"[Dead letter after {job.attempts} attempts]: {clean_error}"
            step.finished_at = now
        if video:
            video.status = "failed"

    db.commit()
    return True


def reap_dead_jobs(
    db: Session,
    grace_period_seconds: int = 15,
) -> int:
    """Reaper loop: find running jobs whose heartbeat lease has expired (e.g. worker was SIGKILLed or OOMed).
    Recovers the job for retry or marks as dead letter."""
    now = _now()
    cutoff = now - timedelta(seconds=grace_period_seconds)

    is_sqlite = db.bind is not None and db.bind.dialect.name == "sqlite"
    select_stmt = select(PipelineJob).where(
        PipelineJob.status == "running",
        PipelineJob.lease_expires_at < cutoff,
    )
    if not is_sqlite:
        select_stmt = select_stmt.with_for_update(skip_locked=True)
    else:
        select_stmt = select_stmt.with_for_update()

    dead_jobs = db.scalars(select_stmt).all()
    reaped_count = 0

    for job in dead_jobs:
        video = db.scalar(select(Video).where(Video.id == job.video_id).with_for_update())
        step = next((s for s in (video.steps if video else []) if s.name == job.step_name), None)
        worker_info = job.worker_id or "unknown worker"
        diagnostic = f"Lease expired (worker {worker_info} stopped heartbeats at {job.last_heartbeat_at})"

        if job.attempts < job.max_attempts:
            job.status = "queued"
            job.scheduled_at = now  # Requeue immediately
            job.lease_token = None
            job.lease_expires_at = None
            job.error = diagnostic
            if step:
                step.error = f"[Reaped & Requeued]: {diagnostic}"
        else:
            job.status = "dead_letter"
            job.finished_at = now
            job.lease_token = None
            job.lease_expires_at = None
            job.error = diagnostic
            if step:
                step.status = "failed"
                step.error = f"[Dead Letter]: {diagnostic}"
                step.finished_at = now
            if video:
                video.status = "failed"

        reaped_count += 1

    db.commit()
    return reaped_count
