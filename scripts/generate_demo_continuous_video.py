"""Generate a complete end-to-end demo video using Continuous Narration and Render integration.

Reuses the 12 high-quality vertical AI images from the Yawning topic, synthesizes
a unified continuous narration using the local Kokoro-82M ONNX model, renders
the complete 1080x1920 video with continuous audio muxing, and generates animated
subtitles. Saves the video in both the PostgreSQL database and storage directory.

Usage:
    .venv/bin/python scripts/generate_demo_continuous_video.py
"""

from __future__ import annotations

import datetime
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.db import SessionLocal
from app.models import Channel, Video, VideoStep
from app.pipeline import STEPS
from app.steps.render import generate_render_step
from app.steps.subtitles import generate_subtitles_step
from app.steps.voiceover import generate_voiceover_step
from app.voiceover.kokoro import KokoroVoiceoverProvider


def run_ffprobe(file_path: Path) -> dict:
    """Inspect media file with ffprobe and return parsed metadata."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=codec_name,codec_type,width,height,r_frame_rate,channels,sample_rate,pix_fmt,duration",
        "-show_entries", "format=duration,size,bit_rate",
        "-of", "json",
        str(file_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return json.loads(res.stdout)


def main():
    print("=" * 70)
    print("Generating Complete Continuous Narration Test Video")
    print("=" * 70)

    storage_root = getattr(settings, "resolved_storage_dir", Path(settings.storage_dir).resolve())
    source_video_id = "393ccae9-7ed5-41bb-b74b-5bba992ec5ab"
    source_dir = storage_root / source_video_id

    if not source_dir.is_dir():
        print(f"Error: Source video directory not found: {source_dir}")
        sys.exit(1)

    with SessionLocal() as db:
        src_video = db.query(Video).filter(Video.id == source_video_id).first()
        if not src_video:
            print(f"Error: Source video not found in DB: {source_video_id}")
            sys.exit(1)

        src_script_step = next(s for s in src_video.steps if s.name == "script")
        src_sb_step = next(s for s in src_video.steps if s.name == "storyboard")
        src_img_step = next(s for s in src_video.steps if s.name == "images")

        # 1. Create new Video entity
        new_video_id = uuid.uuid4()
        new_dir = storage_root / str(new_video_id)
        new_dir.mkdir(parents=True, exist_ok=True)
        images_dir = new_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        audio_dir = new_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[1] Created new video entity: {new_video_id}")
        print(f"    Topic: Why We Yawn When We See Others Yawn (Continuous Narration)")
        print(f"    Storage: {new_dir}")

        # 2. Copy existing image assets to new video folder
        print("[2] Copying 12 existing vertical AI images into new video storage...")
        img_items = []
        for it in src_img_step.output.get("images", {}).get("items", []):
            idx = it["index"]
            src_file_rel = it["file"]
            src_file_abs = storage_root / src_file_rel
            dest_filename = f"scene_{idx:02d}.png"
            dest_file_rel = f"{new_video_id}/images/{dest_filename}"
            dest_file_abs = new_dir / "images" / dest_filename
            shutil.copy2(str(src_file_abs), str(dest_file_abs))
            img_items.append({
                **it,
                "file": dest_file_rel,
            })

        new_images_output = {
            "images": {
                "items": img_items,
            },
            "items": img_items,
        }

        # 3. Create Video & VideoStep rows in DB
        now = datetime.datetime.now(datetime.timezone.utc)
        new_video = Video(
            id=new_video_id,
            channel_id=src_video.channel_id,
            topic="Why We Yawn When We See Others Yawn (Continuous Narration)",
            status="in_progress",
        )
        db.add(new_video)

        step_models = {}
        for pos, s_name in enumerate(STEPS):
            step_obj = VideoStep(
                video_id=new_video_id,
                name=s_name,
                position=pos,
                status="pending",
                started_at=now,
            )
            db.add(step_obj)
            step_models[s_name] = step_obj
        db.flush()

        step_models["script"].output = src_script_step.output
        step_models["script"].status = "done"
        step_models["script"].finished_at = now

        step_models["storyboard"].output = src_sb_step.output
        step_models["storyboard"].status = "done"
        step_models["storyboard"].finished_at = now

        step_models["images"].output = new_images_output
        step_models["images"].status = "done"
        step_models["images"].finished_at = now
        db.commit()

        # 4. Generate Continuous Narration using local Kokoro-82M ONNX
        print("\n[3] Synthesizing continuous narration with local Kokoro-82M ONNX...")
        prev_vo = {
            "script": src_script_step.output,
            "storyboard": src_sb_step.output,
            "images": new_images_output,
        }

        t_vo_start = time.perf_counter()
        vo_output = generate_voiceover_step(
            video=new_video,
            prev=prev_vo,
            step=step_models["voiceover"],
        )
        vo_duration = round(time.perf_counter() - t_vo_start, 2)
        print(f"    Continuous narration generated in {vo_duration}s!")
        print(f"    Narration WAV: {vo_output['audio_file']}")
        print(f"    Audio duration: {vo_output['duration_seconds']}s")
        print(f"    Inference calls / batches: {vo_output['meta']['inference_calls']}")

        step_models["voiceover"].output = vo_output
        step_models["voiceover"].status = "done"
        step_models["voiceover"].finished_at = datetime.datetime.now(datetime.timezone.utc)
        db.commit()

        # 5. Render Video with Continuous Audio
        print("\n[4] Rendering video with continuous narration audio muxing...")
        prev_render = {
            **prev_vo,
            "voiceover": vo_output,
        }

        t_render_start = time.perf_counter()
        render_output = generate_render_step(
            video=new_video,
            prev=prev_render,
            step=step_models["render"],
        )
        render_duration = round(time.perf_counter() - t_render_start, 2)
        print(f"    Video rendered in {render_duration}s!")
        print(f"    Rendered MP4: {render_output['video_path']}")
        print(f"    Reported duration: {render_output['duration_seconds']}s")
        print(f"    Audio source used: {render_output['render']['audio_source']}")

        step_models["render"].output = render_output
        step_models["render"].status = "done"
        step_models["render"].finished_at = datetime.datetime.now(datetime.timezone.utc)
        new_video.output_path = render_output["video_path"]
        db.commit()

        # 6. Generate Subtitles and Burn Captions
        print("\n[5] Generating synchronized subtitles and burning captions...")
        prev_subs = {
            **prev_render,
            "render": render_output,
        }

        t_subs_start = time.perf_counter()
        subs_output = generate_subtitles_step(
            video=new_video,
            prev=prev_subs,
            step=step_models["subtitles"],
        )
        subs_duration = round(time.perf_counter() - t_subs_start, 2)
        print(f"    Subtitles completed in {subs_duration}s!")
        captioned_path = Path(subs_output["subtitles"]["captioned_video_path"])
        print(f"    Captioned MP4: {captioned_path}")

        step_models["subtitles"].output = subs_output
        step_models["subtitles"].status = "done"
        step_models["subtitles"].finished_at = datetime.datetime.now(datetime.timezone.utc)
        new_video.status = "completed"
        db.commit()

        # 7. Final FFprobe Inspection
        print("\n[6] Final FFprobe Verification on Captioned Video:")
        probe = run_ffprobe(captioned_path)
        v_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
        a_stream = next(s for s in probe["streams"] if s["codec_type"] == "audio")
        fmt = probe["format"]

        print(f"    Resolution: {v_stream['width']}x{v_stream['height']} (Vertical 9:16)")
        print(f"    Video Codec: {v_stream['codec_name']} @ {v_stream['r_frame_rate']} fps")
        print(f"    Audio Codec: {a_stream['codec_name']}, {a_stream['sample_rate']} Hz, {a_stream['channels']} channels")
        print(f"    Audio Stream Duration: {float(a_stream['duration']):.3f}s")
        print(f"    Video Stream Duration: {float(v_stream['duration']):.3f}s")
        print(f"    Container Duration: {float(fmt['duration']):.3f}s")
        print(f"    File Size: {int(fmt['size']) / 1024 / 1024:.2f} MB")

        print("\n" + "=" * 70)
        print("NEW VIDEO GENERATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print(f"Video ID: {new_video_id}")
        print(f"Uncaptioned MP4: {render_output['video_path']}")
        print(f"Captioned MP4:   {captioned_path}")
        print(f"Continuous WAV:  {storage_root / vo_output['audio_file']}")
        print("=" * 70)


if __name__ == "__main__":
    main()
