"""Unit and lifecycle tests for Phase 5 PostgreSQL-backed transactional queue."""
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.models import Channel, PipelineJob, User, Video, VideoStep
from app.pipeline import init_steps
from app.queue.engine import (
    JobConflictError,
    PipelineError,
    claim_next_job,
    complete_job,
    enqueue_step_job,
    fail_job,
    reap_dead_jobs,
    renew_job_heartbeat,
)
from app.queue.worker import QueueWorker


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _create_test_video(session: Session, require_approval: bool = True) -> Video:
    user = User(email=f"user_{uuid.uuid4().hex[:8]}@example.com", password_hash="hash")
    session.add(user)
    session.flush()

    channel = Channel(
        user_id=user.id,
        name=f"Channel {uuid.uuid4().hex[:6]}",
        settings={"require_approval": require_approval},
    )
    session.add(channel)
    session.flush()

    video = Video(channel_id=channel.id, topic="Testing Queue System")
    session.add(video)
    session.flush()

    init_steps(session, video)
    return video


def test_enqueue_step_job_atomic_success(db_session):
    """Enqueueing a step job atomically updates step to running and creates a queued job."""
    video = _create_test_video(db_session, require_approval=True)

    job = enqueue_step_job(db_session, video.id, "script", run_all=False)
    assert job.status == "queued"
    assert job.step_name == "script"
    assert job.attempts == 0

    db_session.refresh(video)
    assert video.steps[0].status == "running"
    assert video.steps[0].started_at is not None
    assert video.status == "in_progress"


def test_enqueue_duplicate_active_job_prevention(db_session):
    """Attempting to enqueue another job while one is already queued or running raises JobConflictError."""
    video = _create_test_video(db_session, require_approval=True)

    # First enqueue succeeds
    enqueue_step_job(db_session, video.id, "script", run_all=False)

    # Second enqueue for same step raises conflict
    with pytest.raises(JobConflictError, match="active job already exists"):
        enqueue_step_job(db_session, video.id, "script", run_all=False)


def test_claim_next_job_with_fencing_token(db_session):
    """Claiming a job issues a fresh fencing lease token and sets status to running."""
    video = _create_test_video(db_session, require_approval=True)
    enqueue_step_job(db_session, video.id, "script", run_all=False)

    claim_result = claim_next_job(db_session, worker_id="worker-test-1", lease_duration_seconds=60)
    assert claim_result is not None
    job, lease_token = claim_result

    assert job.status == "running"
    assert job.worker_id == "worker-test-1"
    assert job.attempts == 1
    assert job.lease_token == lease_token
    assert job.lease_expires_at is not None

    # When queue is empty, claim returns None
    assert claim_next_job(db_session, worker_id="worker-test-1") is None


def test_heartbeat_renewal_and_revocation(db_session):
    """Heartbeat extends lease if token matches, and rejects if lease was revoked."""
    video = _create_test_video(db_session, require_approval=True)
    enqueue_step_job(db_session, video.id, "script", run_all=False)
    job, lease_token = claim_next_job(db_session, worker_id="worker-test-1", lease_duration_seconds=30)

    # 1. Valid renewal
    renewed = renew_job_heartbeat(db_session, job.id, lease_token, lease_duration_seconds=60)
    assert renewed is True

    # 2. Invalid renewal with bad fencing token
    bad_token = uuid.uuid4()
    renewed_bad = renew_job_heartbeat(db_session, job.id, bad_token, lease_duration_seconds=60)
    assert renewed_bad is False


def test_complete_job_with_phase4_gatekeeping(db_session):
    """Completing a job with require_approval=True pauses step at awaiting_approval."""
    video = _create_test_video(db_session, require_approval=True)
    enqueue_step_job(db_session, video.id, "script", run_all=False)
    job, lease_token = claim_next_job(db_session, worker_id="worker-test-1")

    output_payload = {"script": {"title": "Hook", "narration": "Script text"}}
    completed = complete_job(db_session, job.id, lease_token, output=output_payload)
    assert completed is True

    db_session.refresh(video)
    step = video.steps[0]
    assert step.status == "awaiting_approval"
    assert step.output == output_payload
    assert step.revisions_count >= 1

    db_session.refresh(job)
    assert job.status == "completed"
    assert job.finished_at is not None


def test_complete_job_auto_advance_when_approval_disabled(db_session):
    """When require_approval=False and run_all=True, completing step 0 enqueues step 1."""
    video = _create_test_video(db_session, require_approval=False)
    enqueue_step_job(db_session, video.id, "script", run_all=True)
    job, lease_token = claim_next_job(db_session, worker_id="worker-test-1")

    output_payload = {"script": {"title": "Hook", "narration": "Script text"}}
    completed = complete_job(db_session, job.id, lease_token, output=output_payload)
    assert completed is True

    db_session.refresh(video)
    assert video.steps[0].status == "done"

    # Step 1 (storyboard) should have been chained and queued
    assert video.steps[1].status == "running"
    next_job = db_session.query(PipelineJob).filter(PipelineJob.step_name == "storyboard").first()
    assert next_job is not None
    assert next_job.status == "queued"
    assert next_job.run_all is True


def test_failure_retry_with_exponential_backoff(db_session):
    """Failure increments attempts and schedules retry with backoff."""
    video = _create_test_video(db_session, require_approval=True)
    enqueue_step_job(db_session, video.id, "script", run_all=False, max_attempts=3)
    job, lease_token = claim_next_job(db_session, worker_id="worker-test-1")

    # Fail attempt 1
    fail_job(db_session, job.id, lease_token, error_message="LLM rate limit reached", base_retry_delay_seconds=10)

    db_session.refresh(job)
    assert job.status == "queued"
    assert job.attempts == 1
    assert "LLM rate limit reached" in job.error
    # Next schedule should be in the future
    scheduled_at = job.scheduled_at
    if scheduled_at.tzinfo is None:
        scheduled_at = scheduled_at.replace(tzinfo=timezone.utc)
    assert scheduled_at > datetime.now(timezone.utc)


def test_dead_letter_queue_after_max_attempts(db_session):
    """Job moves to dead_letter when max attempts is exhausted."""
    video = _create_test_video(db_session, require_approval=True)
    enqueue_step_job(db_session, video.id, "script", run_all=False, max_attempts=2)

    # Attempt 1
    job, token1 = claim_next_job(db_session, worker_id="worker-test-1")
    fail_job(db_session, job.id, token1, "Transient error 1", base_retry_delay_seconds=0)

    # Fast forward schedule for attempt 2
    job.scheduled_at = datetime.now(timezone.utc) - timedelta(seconds=1)
    db_session.commit()

    # Attempt 2 (exhausts max_attempts=2)
    job, token2 = claim_next_job(db_session, worker_id="worker-test-1")
    fail_job(db_session, job.id, token2, "Permanent fatal error", base_retry_delay_seconds=0)

    db_session.refresh(job)
    assert job.status == "dead_letter"
    assert job.finished_at is not None

    db_session.refresh(video)
    assert video.steps[0].status == "failed"
    assert "Dead letter" in video.steps[0].error
    assert video.status == "failed"


def test_crash_recovery_reaper(db_session):
    """Reaper discovers abandoned jobs with expired leases and requeues them."""
    video = _create_test_video(db_session, require_approval=True)
    enqueue_step_job(db_session, video.id, "script", run_all=False, max_attempts=3)
    job, _ = claim_next_job(db_session, worker_id="crashed-worker", lease_duration_seconds=10)

    # Simulate worker crash: lease expired 30 seconds ago
    job.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=30)
    db_session.commit()

    reaped_count = reap_dead_jobs(db_session, grace_period_seconds=5)
    assert reaped_count == 1

    db_session.refresh(job)
    assert job.status == "queued"
    assert job.lease_token is None
    assert "stopped heartbeats" in job.error


def test_queue_worker_run_once_integration(db_session, monkeypatch):
    """QueueWorker.run_once claims, executes handler outside transaction, and completes job."""
    video = _create_test_video(db_session, require_approval=True)
    enqueue_step_job(db_session, video.id, "script", run_all=False)

    def mock_script(v, prev, s):
        return {"script": {"title": "Bridges", "narration": "Arch bridges distribute load."}}

    monkeypatch.setitem(__import__("app.pipeline", fromlist=["HANDLERS"]).HANDLERS, "script", mock_script)
    monkeypatch.setattr("app.queue.worker.SessionLocal", sessionmaker(bind=db_session.get_bind()))

    worker = QueueWorker(worker_id="unit-test-worker", lease_duration_seconds=30)
    processed = worker.run_once()
    assert processed is True

    db_session.refresh(video)
    assert video.steps[0].status == "awaiting_approval"
    assert video.steps[0].output["script"]["title"] == "Bridges"


def test_api_next_and_jobs_endpoints(db_session):
    """FastAPI POST /videos/{id}/next and GET /videos/{id}/jobs integration."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.deps import get_db, get_owned_video

    video = _create_test_video(db_session, require_approval=True)
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_owned_video] = lambda: video

    client = TestClient(app)
    try:
        # Enqueue next step
        resp = client.post(f"/videos/{video.id}/next")
        assert resp.status_code == 202
        data = resp.json()
        step = next(s for s in data["steps"] if s["name"] == "script")
        assert step["status"] == "running"

        # List jobs
        jobs_resp = client.get(f"/videos/{video.id}/jobs")
        assert jobs_resp.status_code == 200
        jobs = jobs_resp.json()
        assert len(jobs) == 1
        assert jobs[0]["step_name"] == "script"
        assert jobs[0]["status"] == "queued"
    finally:
        app.dependency_overrides.clear()
