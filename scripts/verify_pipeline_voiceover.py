"""Verification script for Phase A: Real Kokoro Voiceover Pipeline Integration.

Runs the actual pipeline voiceover step on video 386aaf7e-32fd-4c40-bf49-c349e4f2ee9c,
validates generated 24 kHz WAV files and DB metadata, and tests audio reuse.

Usage:
    .venv/bin/python scripts/verify_pipeline_voiceover.py
"""

from pathlib import Path
import sys
import time
import uuid
import wave

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import select
from app.config import settings
from app.db import SessionLocal
from app.models import Video, VideoStep
import app.pipeline as pipeline

TARGET_VIDEO_ID = uuid.UUID("386aaf7e-32fd-4c40-bf49-c349e4f2ee9c")


def main():
    print("=" * 70)
    print("  PHASE A: KOKORO-82M PIPELINE INTEGRATION VERIFICATION")
    print("=" * 70)
    print(f"Target Video ID: {TARGET_VIDEO_ID}")
    print(f"Configured Provider: {settings.voiceover_provider}")
    print(f"Default Voice:       {settings.voiceover_default_voice}\n")

    # -------------------------------------------------------------------------
    # 1. Pre-Run Inspection
    # -------------------------------------------------------------------------
    with SessionLocal() as db:
        video = db.scalar(select(Video).where(Video.id == TARGET_VIDEO_ID))
        if not video:
            print(f"ERROR: Video {TARGET_VIDEO_ID} not found in database.")
            sys.exit(1)

        print(f"Video Topic:   \"{video.topic}\"")
        print(f"Video Status:  {video.status}")
        print("Existing Steps:")
        step_dict = {s.name: s for s in video.steps}
        for s in video.steps:
            print(f"  - {s.name} (pos {s.position}): status={s.status}, error={s.error}")

        # Safety assertions
        assert step_dict["script"].status == "done", "Script step must be done"
        assert step_dict["storyboard"].status == "done", "Storyboard step must be done"
        assert step_dict["images"].status == "done", "Images step must be done"
        assert step_dict["voiceover"].status in ("pending", "done"), f"Voiceover status was {step_dict['voiceover'].status}"
        assert step_dict["render"].status == "pending", f"Render step must be pending, was {step_dict['render'].status}"

        # If voiceover was already done from prior manual testing, reset it cleanly
        if step_dict["voiceover"].status == "done":
            print("\nNotice: Voiceover was already done. Resetting to pending for fresh test...")
            pipeline.rerun_step(db, video, "voiceover", prompt=None)
            db.refresh(video)

    # -------------------------------------------------------------------------
    # 2. Run 1: Real Kokoro Pipeline Generation
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 1: Claiming and executing voiceover via pipeline.claim_next_step & execute...")
    print("-" * 70)

    with SessionLocal() as db:
        claimed_step = pipeline.claim_next_step(db, TARGET_VIDEO_ID)
        assert claimed_step.name == "voiceover", f"Expected to claim 'voiceover', but claimed '{claimed_step.name}'"
        print(f"Claimed step: '{claimed_step.name}' (status: {claimed_step.status})")

    gen_start = time.perf_counter()
    pipeline.execute(TARGET_VIDEO_ID, "voiceover", run_all=False)
    gen_elapsed = time.perf_counter() - gen_start

    print(f"Pipeline voiceover execution finished in {gen_elapsed:.2f} seconds.")

    # -------------------------------------------------------------------------
    # 3. Verify Database Output & Status after Run 1
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 2: Verifying database step output & metadata...")
    print("-" * 70)

    with SessionLocal() as db:
        video = db.scalar(select(Video).where(Video.id == TARGET_VIDEO_ID))
        vo_step = next(s for s in video.steps if s.name == "voiceover")
        render_step = next(s for s in video.steps if s.name == "render")

        assert vo_step.status == "done", f"Expected voiceover status 'done', got '{vo_step.status}'"
        assert vo_step.error is None, f"Expected no error, got '{vo_step.error}'"
        assert render_step.status == "pending", f"Render step must remain 'pending', got '{render_step.status}'"
        assert video.status == "in_progress", f"Expected video status 'in_progress', got '{video.status}'"

        output = vo_step.output
        assert output is not None, "Step output is None"
        assert "voiceover" in output, "Missing 'voiceover' key in output"
        vo_data = output["voiceover"]
        meta = output.get("meta", {})

        print(f"Voiceover Complete:   {vo_data.get('complete')}")
        print(f"Total Duration:       {vo_data.get('total_duration_seconds'):.2f} s")
        print(f"Top-Level Duration:   {output.get('duration_seconds'):.2f} s")
        print(f"Provider in Meta:     {meta.get('provider')}")
        print(f"Model in Meta:        {meta.get('model')}")
        print(f"Voice in Meta:        {meta.get('voice')}")
        print(f"Speed in Meta:        {meta.get('speed')}")
        print(f"Scenes Total:         {meta.get('scenes_total')}")
        print(f"Scenes Generated:     {meta.get('scenes_generated')}")
        print(f"Scenes Reused:        {meta.get('scenes_reused')}")
        print(f"Scenes Failed:        {meta.get('scenes_failed')}")
        print(f"Step Inference Time:  {meta.get('latency_seconds'):.2f} s")

        assert vo_data.get("complete") is True, "Expected complete == True"
        assert meta.get("provider") == "kokoro", f"Expected provider 'kokoro', got '{meta.get('provider')}'"
        assert meta.get("scenes_total") == 11, f"Expected 11 scenes, got {meta.get('scenes_total')}"
        assert meta.get("scenes_generated") == 11, f"Expected 11 generated, got {meta.get('scenes_generated')}"
        assert meta.get("scenes_reused") == 0, f"Expected 0 reused, got {meta.get('scenes_reused')}"
        assert meta.get("scenes_failed") == 0, f"Expected 0 failed, got {meta.get('scenes_failed')}"
        assert len(vo_data.get("items", [])) == 11, f"Expected 11 items, got {len(vo_data.get('items', []))}"

        items = vo_data["items"]

    # -------------------------------------------------------------------------
    # 4. Validate Audio Files on Disk
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 3: Inspecting and validating generated WAV files on disk...")
    print("-" * 70)

    storage_root = Path(settings.storage_dir).resolve()
    initial_mtimes = {}

    for item in items:
        idx = item["index"]
        rel_file = item["file"]
        abs_file = storage_root / rel_file

        assert abs_file.is_file(), f"File does not exist: {abs_file}"
        file_size = abs_file.stat().st_size
        assert file_size > 44, f"File {abs_file} too small: {file_size} bytes"

        # Record exact modification time for reuse check
        initial_mtimes[idx] = abs_file.stat().st_mtime_ns

        # Validate WAV stream
        with wave.open(str(abs_file), "rb") as wf:
            channels = wf.getnchannels()
            samp_width = wf.getsampwidth()
            framerate = wf.getframerate()
            frames = wf.getnframes()
            dur = frames / float(framerate)

        assert channels == 1, f"Scene {idx}: Expected 1 channel, got {channels}"
        assert samp_width == 2, f"Scene {idx}: Expected 16-bit PCM (2 bytes), got {samp_width}"
        assert framerate == 24000, f"Scene {idx}: Expected 24000 Hz, got {framerate}"
        assert frames > 0, f"Scene {idx}: Frame count is 0"
        assert abs(dur - item["duration_seconds"]) < 0.05, f"Duration mismatch: {dur} vs {item['duration_seconds']}"

        print(f"Scene {idx:02d}: {abs_file.name} | {dur:.2f}s | {frames:,} frames | {file_size:,} bytes | 24kHz 16-bit Mono [OK]")

    # -------------------------------------------------------------------------
    # 5. Run 2: Audio Reuse Verification
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 4: Rerunning voiceover step to verify cache reuse...")
    print("-" * 70)

    with SessionLocal() as db:
        video = db.scalar(select(Video).where(Video.id == TARGET_VIDEO_ID))
        print("Calling pipeline.rerun_step(video, 'voiceover', prompt=None)...")
        pipeline.rerun_step(db, video, "voiceover", prompt=None)

    with SessionLocal() as db:
        claimed_rerun = pipeline.claim_next_step(db, TARGET_VIDEO_ID)
        assert claimed_rerun.name == "voiceover", f"Expected 'voiceover', got '{claimed_rerun.name}'"
        print(f"Claimed rerun step: '{claimed_rerun.name}' (status: {claimed_rerun.status})")

    reuse_start = time.perf_counter()
    pipeline.execute(TARGET_VIDEO_ID, "voiceover", run_all=False)
    reuse_elapsed = time.perf_counter() - reuse_start

    print(f"Rerun finished in {reuse_elapsed:.3f} seconds (compared to {gen_elapsed:.2f}s initial).")

    # -------------------------------------------------------------------------
    # 6. Verify Reuse Metadata & Unchanged File Timestamps
    # -------------------------------------------------------------------------
    print("\n" + "-" * 70)
    print("STEP 5: Verifying reuse metadata and file modification timestamps...")
    print("-" * 70)

    with SessionLocal() as db:
        video = db.scalar(select(Video).where(Video.id == TARGET_VIDEO_ID))
        vo_step = next(s for s in video.steps if s.name == "voiceover")
        render_step = next(s for s in video.steps if s.name == "render")

        assert vo_step.status == "done", f"Expected status 'done', got '{vo_step.status}'"
        assert render_step.status == "pending", f"Render step must remain 'pending', got '{render_step.status}'"
        assert video.status == "in_progress", f"Expected video status 'in_progress', got '{video.status}'"

        meta = vo_step.output["meta"]
        items_rerun = vo_step.output["voiceover"]["items"]

        print(f"Scenes Total:     {meta.get('scenes_total')}")
        print(f"Scenes Reused:    {meta.get('scenes_reused')}")
        print(f"Scenes Generated: {meta.get('scenes_generated')}")

        assert meta.get("scenes_reused") == 11, f"Expected 11 reused, got {meta.get('scenes_reused')}"
        assert meta.get("scenes_generated") == 0, f"Expected 0 generated, got {meta.get('scenes_generated')}"

        for item in items_rerun:
            idx = item["index"]
            assert item["reused"] is True, f"Scene {idx} reused is False"
            assert item["status"] == "ok", f"Scene {idx} status is not ok"

            # Check file mtime has NOT changed
            abs_file = storage_root / item["file"]
            curr_mtime = abs_file.stat().st_mtime_ns
            assert curr_mtime == initial_mtimes[idx], f"Scene {idx} file was modified during reuse run!"

    print("\nAll 11 scene audio files verified: identical file mtimes confirmed, 0 files regenerated.")
    print("=" * 70)
    print("  PHASE A VERIFICATION: 100% SUCCESS")
    print("=" * 70)


if __name__ == "__main__":
    main()
