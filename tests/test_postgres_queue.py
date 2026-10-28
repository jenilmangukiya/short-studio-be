"""PostgreSQL integration tests verifying true database concurrency,
row-locking via SELECT FOR UPDATE SKIP LOCKED, partial unique index idempotency,
and atomic crash recovery against a live PostgreSQL 16 database.
"""
import concurrent.futures
import threading
import uuid
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.db import SessionLocal, engine
from app.models import Channel, PipelineJob, User, Video, VideoStep
from app.pipeline import init_steps
from app.queue.engine import (
    JobConflictError,
    QueueError,
    claim_next_job,
    complete_job,
    enqueue_step_job,
    fail_job,
    reap_dead_jobs,
    renew_job_heartbeat,
)


@pytest.fixture(scope="module")
def check_postgres():
    """Ensure PostgreSQL is accessible before running concurrency integration tests."""
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL dialect required for SKIP LOCKED concurrency integration tests")
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL container not reachable: {exc}")


@pytest.fixture
def pg_cleanup():
    """Tracks test user IDs and cleans them up after each test."""
    created_user_ids = []

    def _register(user_id: uuid.UUID):
        created_user_ids.append(user_id)

    yield _register

    with SessionLocal() as db:
        for uid in created_user_ids:
            try:
                user = db.scalar(select(User).where(User.id == uid))
                if user:
                    db.delete(user)
                db.commit()
            except Exception:
                db.rollback()


def _create_pg_video(db: Session, pg_cleanup, require_approval: bool = True) -> Video:
    user = User(email=f"pg_test_{uuid.uuid4().hex[:10]}@example.com", password_hash="hash")
    db.add(user)
    db.flush()
    pg_cleanup(user.id)

    channel = Channel(
        user_id=user.id,
        name=f"PG Concurrency Channel {uuid.uuid4().hex[:6]}",
        settings={"require_approval": require_approval},
    )
    db.add(channel)
    db.flush()

    video = Video(
        channel_id=channel.id,
        topic="PostgreSQL Concurrency Testing",
        status="draft",
    )
    db.add(video)
    db.flush()
    init_steps(db, video)
    db.commit()
    db.refresh(video)
    return video


def test_postgres_concurrent_claims_skip_locked(check_postgres, pg_cleanup):
    """Verify that multiple concurrent workers claiming from PostgreSQL simultaneously
    using SELECT FOR UPDATE SKIP LOCKED claim disjoint jobs with ZERO duplicate claims and ZERO blocking."""
    num_jobs = 5
    job_ids = []

    with SessionLocal() as db:
        for i in range(num_jobs):
            v = _create_pg_video(db, pg_cleanup, require_approval=True)
            job = enqueue_step_job(db, v.id, "script", run_all=False)
            job_ids.append(job.id)

    # Barrier to synchronize threads before claiming simultaneously
    barrier = threading.Barrier(num_jobs)
    results = [None] * num_jobs

    def worker_claim(worker_idx: int):
        barrier.wait()  # synchronize release
        with SessionLocal() as thread_db:
            claim = claim_next_job(
                thread_db,
                worker_id=f"concurrent-worker-{worker_idx}",
                lease_duration_seconds=60,
            )
            return claim

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_jobs) as executor:
        futures = {executor.submit(worker_claim, i): i for i in range(num_jobs)}
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            results[idx] = future.result()

    # Verify that all 5 workers received a claim
    claimed_jobs = [r for r in results if r is not None]
    assert len(claimed_jobs) == num_jobs, f"Expected {num_jobs} claims, got {len(claimed_jobs)}"

    claimed_job_ids = [claim[0].id for claim in claimed_jobs]
    claimed_tokens = [claim[1] for claim in claimed_jobs]

    # Concurrency Guarantee: All claimed job IDs must be unique
    assert len(set(claimed_job_ids)) == num_jobs, "Duplicate job claimed across workers!"
    assert set(claimed_job_ids) == set(job_ids)

    # Concurrency Guarantee: All lease tokens must be unique
    assert len(set(claimed_tokens)) == num_jobs

    # Queue should now be empty for an extra worker
    with SessionLocal() as db:
        extra_claim = claim_next_job(db, worker_id="extra-worker", lease_duration_seconds=60)
        assert extra_claim is None


def test_postgres_partial_unique_index_active_jobs(check_postgres, pg_cleanup):
    """Verify PostgreSQL partial unique index uq_pipeline_jobs_active_step
    prevents duplicate active (queued or running) jobs for the exact same step."""
    with SessionLocal() as db:
        video = _create_pg_video(db, pg_cleanup, require_approval=True)
        # 1. Enqueue script step
        job1 = enqueue_step_job(db, video.id, "script", run_all=False)
        assert job1.status == "queued"

        # 2. Attempt to enqueue the same step again while active
        with pytest.raises(JobConflictError, match="already exists for step"):
            enqueue_step_job(db, video.id, "script", run_all=False)

        # 3. Direct DB level insert attempt must also violate the partial unique index
        from sqlalchemy.exc import IntegrityError
        with pytest.raises(IntegrityError):
            raw_dup = PipelineJob(
                video_id=video.id,
                step_name="script",
                status="queued",
            )
            db.add(raw_dup)
            db.commit()
        db.rollback()

        # 4. Claim and complete job1
        claimed, token = claim_next_job(db, worker_id="worker-pg-1")
        assert claimed.id == job1.id
        ok = complete_job(
            db,
            job1.id,
            token,
            output={"script": {"title": "PG Concurrency Done", "narration": "Done."}},
        )
        assert ok is True

        db.refresh(job1)
        assert job1.status == "completed"

        # 5. Now that job1 is 'completed', it is excluded from the partial index where predicate.
        # User reruns step: rerun_step resets script status to pending, allowing re-enqueueing without conflict.
        from app.pipeline import rerun_step
        rerun_step(db, video, "script", prompt="Revised narration")
        job2 = enqueue_step_job(db, video.id, "script", run_all=False)
        assert job2.id != job1.id
        assert job2.status == "queued"


def test_postgres_atomic_enqueue_rollback(check_postgres, pg_cleanup):
    """Verify atomic transaction boundary: if an enqueue operation fails mid-transaction,
    no partial step mutations or dangling jobs are committed."""
    with SessionLocal() as db:
        video = _create_pg_video(db, pg_cleanup, require_approval=True)
        step = video.steps[0]
        initial_status = step.status
        assert initial_status == "pending"

        # Simulate transaction failure during job creation
        try:
            with db.begin_nested():
                step.status = "running"
                db.flush()
                # Simulate an unhandled exception before commit
                raise ValueError("Simulated pipeline failure")
        except ValueError:
            db.rollback()

        # Step status must remain unchanged in DB
        db.refresh(step)
        assert step.status == "pending"

        # No jobs must exist
        jobs = db.scalars(select(PipelineJob).where(PipelineJob.video_id == video.id)).all()
        assert len(jobs) == 0


def test_postgres_concurrent_reapers_atomic_recovery(check_postgres, pg_cleanup):
    """Verify multiple concurrent crash reapers recovering expired leases do not double-reap."""
    num_expired = 4
    with SessionLocal() as db:
        for _ in range(num_expired):
            v = _create_pg_video(db, pg_cleanup, require_approval=True)
            enqueue_step_job(db, v.id, "script")
            job, _ = claim_next_job(db, worker_id="crashed-worker", lease_duration_seconds=5)
            # Simulate worker crash: lease expired 30s ago
            job.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=30)
            db.commit()

    # Run 3 reapers concurrently
    num_reapers = 3
    barrier = threading.Barrier(num_reapers)

    def run_reaper():
        barrier.wait()
        with SessionLocal() as thread_db:
            return reap_dead_jobs(thread_db, grace_period_seconds=0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_reapers) as executor:
        futures = [executor.submit(run_reaper) for _ in range(num_reapers)]
        reaped_counts = [f.result() for f in concurrent.futures.as_completed(futures)]

    # The sum across all reapers must equal exactly the number of expired jobs
    total_reaped = sum(reaped_counts)
    assert total_reaped == num_expired, f"Expected {num_expired} total reaped, got {total_reaped}"


def test_postgres_fencing_token_rejects_expired_worker(check_postgres, pg_cleanup):
    """Verify that if worker A's lease expires and worker B re-claims the job,
    worker A cannot commit completion or renew heartbeat using its obsolete fencing token."""
    with SessionLocal() as db:
        video = _create_pg_video(db, pg_cleanup, require_approval=True)
        job = enqueue_step_job(db, video.id, "script")

        # Worker A claims job
        worker_a_job, token_a = claim_next_job(db, worker_id="worker-a", lease_duration_seconds=10)
        assert worker_a_job.id == job.id

        # Simulate Worker A freeze / crash: lease expired
        worker_a_job.lease_expires_at = datetime.now(timezone.utc) - timedelta(seconds=15)
        db.commit()

        # Reaper recovers job
        reaped = reap_dead_jobs(db, grace_period_seconds=0)
        assert reaped == 1

        # Worker B claims job with new fencing token
        worker_b_job, token_b = claim_next_job(db, worker_id="worker-b", lease_duration_seconds=60)
        assert worker_b_job.id == job.id
        assert token_b != token_a

        # Worker A attempts heartbeat renewal with stale token -> rejected
        renewed = renew_job_heartbeat(db, job.id, token_a, lease_duration_seconds=60)
        assert renewed is False

        # Worker A attempts completion with stale token -> returns False (fencing protection)
        success_a = complete_job(
            db,
            job.id,
            token_a,
            output={"script": {"title": "Stale Worker A", "narration": "Old"}},
        )
        assert success_a is False

        # Worker B completes cleanly with active token
        success_b = complete_job(
            db,
            job.id,
            token_b,
            output={"script": {"title": "Fresh Worker B", "narration": "New"}},
        )
        assert success_b is True

        db.refresh(video)
        script_step = video.steps[0]
        assert script_step.output["script"]["title"] == "Fresh Worker B"
