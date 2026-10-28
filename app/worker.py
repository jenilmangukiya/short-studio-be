"""CLI runner for Shorts Studio background queue worker.
Usage:
    python -m app.worker [--once] [--poll-interval 1.0] [--lease-duration 60]
    python scripts/run_worker.py
"""
import argparse
import logging
import sys

from app.db import SessionLocal
from app.queue.engine import reap_dead_jobs
from app.queue.worker import QueueWorker


def parse_args():
    parser = argparse.ArgumentParser(
        description="Shorts Studio PostgreSQL-backed Queue Worker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Unique worker identifier (defaults to hostname-pid)",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=1.0,
        help="Queue polling interval in seconds when idle",
    )
    parser.add_argument(
        "--lease-duration",
        type=int,
        default=60,
        help="Worker lease duration in seconds",
    )
    parser.add_argument(
        "--reaper-interval",
        type=float,
        default=30.0,
        help="Dead-job reaper scan interval in seconds",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single claim-execute cycle if a job is available, then exit",
    )
    parser.add_argument(
        "--reap-only",
        action="store_true",
        help="Execute a single crash-recovery reaper pass and exit",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Console logging level",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        stream=sys.stdout,
    )
    logger = logging.getLogger("shorts_studio.worker_cli")

    if args.reap_only:
        logger.info("Running standalone dead-job crash recovery pass...")
        with SessionLocal() as db:
            reaped = reap_dead_jobs(db)
            logger.info("Crash recovery complete. Reaped %d dead jobs.", reaped)
        return

    worker = QueueWorker(
        worker_id=args.worker_id,
        lease_duration_seconds=args.lease_duration,
        poll_interval_seconds=args.poll_interval,
        reaper_interval_seconds=args.reaper_interval,
    )

    if args.once:
        logger.info("Worker running in single-pass mode (--once)...")
        processed = worker.run_once()
        if processed:
            logger.info("Successfully executed one queued job.")
        else:
            logger.info("No queued jobs ready for execution.")
        return

    worker.run()


if __name__ == "__main__":
    main()
