"""Durable background worker for pipeline job execution with heartbeats,
crash recovery, and signal handling.
"""
import logging
import os
import signal
import socket
import threading
import time
import uuid
from typing import Callable

from sqlalchemy import select

from app.db import SessionLocal
from app.models import PipelineJob, Video
from app.pipeline import HANDLERS
from .engine import claim_next_job, complete_job, fail_job, reap_dead_jobs, renew_job_heartbeat

logger = logging.getLogger("shorts_studio.worker")


class HeartbeatThread(threading.Thread):
    """Background thread that periodically sends heartbeats to renew lease duration."""

    def __init__(
        self,
        job_id: uuid.UUID,
        lease_token: uuid.UUID,
        lease_duration: int = 60,
        interval: float = 15.0,
    ):
        super().__init__(daemon=True)
        self.job_id = job_id
        self.lease_token = lease_token
        self.lease_duration = lease_duration
        self.interval = interval
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def run(self):
        while not self._stop_event.wait(self.interval):
            try:
                with SessionLocal() as db:
                    renewed = renew_job_heartbeat(
                        db=db,
                        job_id=self.job_id,
                        lease_token=self.lease_token,
                        lease_duration_seconds=self.lease_duration,
                    )
                    if not renewed:
                        logger.warning(
                            "Heartbeat rejected for job %s: lease was lost or revoked", self.job_id
                        )
                        break
            except Exception as e:
                logger.error("Error extending heartbeat for job %s: %s", self.job_id, e)


class QueueWorker:
    """Production worker claiming jobs transactionally via SKIP LOCKED, executing pipeline
    handlers outside DB transactions, and providing crash recovery."""

    def __init__(
        self,
        worker_id: str | None = None,
        lease_duration_seconds: int = 60,
        poll_interval_seconds: float = 1.0,
        reaper_interval_seconds: float = 30.0,
    ):
        self.worker_id = worker_id or f"worker-{socket.gethostname()}-{os.getpid()}"
        self.lease_duration = lease_duration_seconds
        self.poll_interval = poll_interval_seconds
        self.reaper_interval = reaper_interval_seconds
        self._stop_event = threading.Event()
        self._last_reap = 0.0

    def stop(self):
        """Signal worker to gracefully stop after completing current in-flight task."""
        logger.info("Worker %s stopping gracefully...", self.worker_id)
        self._stop_event.set()

    def install_signal_handlers(self):
        """Install SIGINT/SIGTERM handlers for graceful shutdown."""
        def _handle_sig(sig, frame):
            logger.info("Received shutdown signal %s", sig)
            self.stop()

        try:
            signal.signal(signal.SIGINT, _handle_sig)
            signal.signal(signal.SIGTERM, _handle_sig)
        except (ValueError, AttributeError):
            # Not in main thread or unsupported platform
            pass

    def run_once(self) -> bool:
        """Attempt to claim and execute one queued job.
        Returns True if a job was processed; False if queue was empty."""
        claim_result = None
        with SessionLocal() as db:
            claim_result = claim_next_job(
                db=db,
                worker_id=self.worker_id,
                lease_duration_seconds=self.lease_duration,
            )

        if not claim_result:
            return False

        job, lease_token = claim_result
        job_id = job.id
        step_name = job.step_name
        video_id = job.video_id
        logger.info("Worker %s claimed job %s (step: %s, attempt: %d)", self.worker_id, job_id, step_name, job.attempts)

        # 1. Start heartbeat thread to extend lease during long-running tasks
        heartbeat = HeartbeatThread(
            job_id=job_id,
            lease_token=lease_token,
            lease_duration=self.lease_duration,
            interval=max(5.0, self.lease_duration / 4),
        )
        heartbeat.start()

        # 2. Execute step handler outside the DB transaction
        output = None
        error_msg = None
        try:
            with SessionLocal() as db:
                video = db.scalar(
                    select(Video).where(Video.id == video_id).execution_options(populate_existing=True)
                )
                step = next((s for s in (video.steps if video else []) if s.name == step_name), None)
                if not video or not step:
                    raise RuntimeError(f"Video {video_id} or step {step_name} not found")

                prev_outputs = {
                    s.name: s.output for s in video.steps if s.status in ("done", "awaiting_approval") and s.output
                }

                handler = HANDLERS.get(step_name)
                if not handler:
                    raise RuntimeError(f"No handler registered for step '{step_name}'")

                output = handler(video, prev_outputs, step)
        except Exception as exc:  # noqa: BLE001
            error_msg = str(exc)
            logger.exception("Step handler failed for job %s: %s", job_id, exc)
        finally:
            heartbeat.stop()
            heartbeat.join(timeout=2.0)

        # 3. Finalize in short atomic transaction
        with SessionLocal() as db:
            if error_msg is not None:
                fail_job(
                    db=db,
                    job_id=job_id,
                    lease_token=lease_token,
                    error_message=error_msg,
                )
            else:
                completed = complete_job(
                    db=db,
                    job_id=job_id,
                    lease_token=lease_token,
                    output=output or {},
                )
                if not completed:
                    logger.warning("Job %s could not be marked completed: lease expired or revoked", job_id)

        return True

    def run(self):
        """Continuous execution loop."""
        logger.info("Worker %s started. Polling for jobs...", self.worker_id)
        self.install_signal_handlers()

        while not self._stop_event.is_set():
            # Periodic dead-job reaper check
            now = time.time()
            if now - self._last_reap >= self.reaper_interval:
                self._last_reap = now
                try:
                    with SessionLocal() as db:
                        reaped = reap_dead_jobs(db)
                        if reaped > 0:
                            logger.info("Reaper recovered %d dead jobs", reaped)
                except Exception as e:
                    logger.error("Error running dead-job reaper: %s", e)

            # Process job
            try:
                processed = self.run_once()
                if not processed:
                    # Queue is empty, pause briefly
                    self._stop_event.wait(self.poll_interval)
            except Exception as e:
                logger.error("Worker loop exception: %s", e)
                self._stop_event.wait(self.poll_interval)

        logger.info("Worker %s stopped cleanly.", self.worker_id)
