"""Render AI-directed animation test video using existing project assets without modifying originals.

Usage:
    .venv/bin/python scripts/render_ai_animation_test.py [--video-id 393ccae9-7ed5-41bb-b74b-5bba992ec5ab]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.animation.analyzer import analyze_image_composition
from app.animation.planner import plan_scene_animation
from app.animation.renderer import build_ffmpeg_zoompan_filter
from app.config import settings
from app.db import SessionLocal
from app.models import Video, VideoStep

DEFAULT_VIDEO_ID = "393ccae9-7ed5-41bb-b74b-5bba992ec5ab"


def get_audio_duration(file_path: Path) -> float:
    """Probe audio duration accurately with ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(res.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description="Render AI-directed animation test video.")
    parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID, help="Target video UUID")
    args = parser.parse_args()

    video_id = args.video_id
    print("=" * 80)
    print(f"  AI-DIRECTED ANIMATION TEST RENDER: {video_id}")
    print("=" * 80)

    storage_root = Path(settings.storage_dir).resolve()
    video_dir = storage_root / video_id
    if not video_dir.is_dir():
        print(f"ERROR: Video directory not found: {video_dir}")
        sys.exit(1)

    # 1. Fetch storyboard and step data from DB (read-only)
    with SessionLocal() as db:
        video = db.query(Video).filter(Video.id == video_id).first()
        if not video:
            print(f"ERROR: Video not found in DB: {video_id}")
            sys.exit(1)
        print(f"Topic: {video.topic}")

        steps = db.query(VideoStep).filter(VideoStep.video_id == video_id).all()
        step_dict = {s.name: s for s in steps}

    sb_output = step_dict.get("storyboard", VideoStep()).output or {}
    sb_scenes = sb_output.get("storyboard", {}).get("scenes", [])
    scenes_by_idx = {s.get("index"): s for s in sb_scenes if isinstance(s, dict) and s.get("index")}

    images_dir = video_dir / "images"
    audio_dir = video_dir / "audio"
    subtitles_ass = video_dir / "subtitles.ass"

    image_files = sorted(images_dir.glob("scene_*.png"))
    audio_files = sorted(audio_dir.glob("scene_*.wav"))

    num_scenes = len(image_files)
    print(f"Found {num_scenes} image files and {len(audio_files)} audio files.")
    assert num_scenes > 0, "No image files found!"
    assert len(audio_files) == num_scenes, "Image and audio count mismatch!"

    temp_dir = video_dir / f"tmp_anim_test_{int(time.time())}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    raw_output_path = video_dir / "test_ai_animation_raw.mp4"
    captioned_output_path = video_dir / "test_ai_animation_captioned.mp4"

    target_w = 1080
    target_h = 1920
    target_fps = 30

    segment_paths: list[Path] = []
    scene_reports: list[dict] = []
    prev_anim_type: str | None = None
    total_audio_duration = 0.0

    print("\nPlanning and rendering scene segments...")
    start_time = time.perf_counter()

    for idx in range(1, num_scenes + 1):
        img_path = image_files[idx - 1]
        vo_path = audio_files[idx - 1]
        scene_dur = get_audio_duration(vo_path)
        total_audio_duration += scene_dur

        sc_meta = scenes_by_idx.get(idx, {})
        shot_type = sc_meta.get("shot_type") or "medium"
        focal_desc = sc_meta.get("focal_point") or ""
        narration = sc_meta.get("narration") or ""

        # Analyze image composition
        img_analysis = analyze_image_composition(img_path)

        # Plan AI-directed animation
        anim_plan = plan_scene_animation(
            scene_data=sc_meta,
            image_path=img_path,
            duration_seconds=scene_dur,
            prev_anim_type=prev_anim_type,
        )

        # Build FFmpeg filter
        z_expr, x_expr, y_expr, filter_chain, resolved = build_ffmpeg_zoompan_filter(
            plan=anim_plan,
            scene_dur=scene_dur,
            target_fps=target_fps,
            target_width=target_w,
            target_height=target_h,
            fallback_shot_type=shot_type,
            fallback_scene_idx=idx,
        )
        prev_anim_type = resolved.get("animation_type")

        # Record report
        report_entry = {
            "scene": idx,
            "shot_type": shot_type,
            "focal_point": focal_desc[:38],
            "comp_type": img_analysis["composition_type"],
            "centroid": f"({img_analysis['centroid_x']:.2f}, {img_analysis['centroid_y']:.2f})",
            "animation_type": resolved.get("animation_type"),
            "scale_range": f"{resolved.get('start_scale'):.2f} -> {resolved.get('end_scale'):.2f}",
            "coords_path": f"({resolved.get('start_coords')[0]:.2f}, {resolved.get('start_coords')[1]:.2f}) -> ({resolved.get('end_coords')[0]:.2f}, {resolved.get('end_coords')[1]:.2f})",
            "duration": f"{scene_dur:.2f}s",
            "reasoning": resolved.get("reasoning", ""),
        }
        scene_reports.append(report_entry)

        # Render segment
        seg_path = temp_dir / f"scene_{idx:02d}.mp4"
        seg_cmd = [
            "ffmpeg",
            "-y",
            "-loop", "1",
            "-t", str(scene_dur),
            "-i", str(img_path),
            "-i", str(vo_path),
            "-vf", filter_chain,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(seg_path),
        ]
        subprocess.run(seg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        segment_paths.append(seg_path)
        print(f"  [✓] Scene {idx:02d}: {resolved.get('animation_type'):<12} | {shot_type:<14} | dur: {scene_dur:.2f}s")

    # Concatenate segments
    print("\nConcatenating scene segments into raw MP4...")
    concat_list = temp_dir / "concat.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for p in segment_paths:
            f.write(f"file '{p.resolve()}'\n")

    concat_cmd = [
        "ffmpeg",
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        "-movflags", "+faststart",
        str(raw_output_path),
    ]
    subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
    print(f"  [✓] Raw MP4 created: {raw_output_path}")

    # Burn in subtitles
    if subtitles_ass.is_file():
        print("\nBurning in word-level animated subtitles...")
        burn_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(raw_output_path),
            "-vf", f"ass={subtitles_ass.resolve()}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(captioned_output_path),
        ]
        subprocess.run(burn_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        print(f"  [✓] Captioned MP4 created: {captioned_output_path}")
    else:
        print(f"  [!] Subtitles file not found at {subtitles_ass}, skipping burn-in.")

    # Cleanup temp directory
    shutil.rmtree(temp_dir, ignore_errors=True)
    render_time = time.perf_counter() - start_time
    print(f"\nRender finished in {render_time:.2f}s.")

    # Print Per-Scene Report Table
    print("\n" + "=" * 120)
    print("  PER-SCENE AI-DIRECTED ANIMATION REPORT")
    print("=" * 120)
    header = f"{'#':<3} | {'Shot Type':<15} | {'Composition':<16} | {'Centroid':<12} | {'Anim Type':<13} | {'Zoom Range':<14} | {'Path (X, Y)':<26} | {'Dur':<6}"
    print(header)
    print("-" * 120)
    for r in scene_reports:
        row = f"{r['scene']:<3} | {r['shot_type']:<15} | {r['comp_type']:<16} | {r['centroid']:<12} | {r['animation_type']:<13} | {r['scale_range']:<14} | {r['coords_path']:<26} | {r['duration']:<6}"
        print(row)
        print(f"     Reasoning: {r['reasoning']}")
    print("=" * 120)

    # Verification via ffprobe
    probe_target = captioned_output_path if captioned_output_path.is_file() else raw_output_path
    print(f"\nVerifying output file: {probe_target}")
    probe_cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration,size:stream=width,height,codec_name,r_frame_rate",
        "-of", "json",
        str(probe_target),
    ]
    probe_res = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    meta = json.loads(probe_res.stdout)

    v_stream = next(s for s in meta["streams"] if s["codec_name"] == "h264")
    a_stream = next((s for s in meta["streams"] if s["codec_name"] == "aac"), None)
    total_dur = float(meta["format"]["duration"])
    file_size_mb = int(meta["format"]["size"]) / (1024 * 1024)

    print("-" * 60)
    print(f"Resolution:      {v_stream['width']}x{v_stream['height']} (Aspect Ratio 9:16)")
    print(f"Video Codec:     {v_stream['codec_name']} (YUV420p)")
    print(f"Audio Codec:     {a_stream['codec_name'] if a_stream else 'None'}")
    print(f"Framerate:       {v_stream['r_frame_rate']} fps")
    print(f"Total Duration:  {total_dur:.3f}s (Audio Duration: {total_audio_duration:.3f}s, delta: {abs(total_dur - total_audio_duration):.3f}s)")
    print(f"File Size:       {file_size_mb:.2f} MB")
    print("-" * 60)

    print("\nOUTPUT PATHS:")
    print(f"  Raw Animation:      {raw_output_path}")
    print(f"  Captioned with ASS: {captioned_output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
