"""Render 3 full video comparisons for Kokoro voices using existing visual assets and pipeline."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

# Ensure backend root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import soundfile as sf

from app.animation.renderer import build_ffmpeg_zoompan_filter
from app.config import settings
from app.db import SessionLocal
from app.models import Video
from app.subtitles.acoustic import align_script_with_vosk, _extract_display_tokens
from app.subtitles.types import SubtitleResult, SubtitleSegment

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("comparison_renderer")

VIDEO_ID = "e8342d48-2150-4b46-b05e-88c454f1fc87"

SCRIPT = """Have you ever been just about to fall asleep, and suddenly felt like you were falling?

It happens when your body starts to relax as you fall asleep.

Sometimes, your muscles suddenly twitch.

One possible explanation is that your brain gets confused by that sudden change. It reacts almost like you're losing your balance.

So your body suddenly jerks, and just like that, you're awake again.

That strange falling feeling has a name: a hypnic jerk.

And no, it doesn't mean something is wrong with your heart. It's a common thing that happens as you fall asleep."""

VOICES = [
    {
        "key": "af_heart",
        "audio_path": Path("test_voices/voice_1_af_heart.wav"),
        "non_captioned_name": "comparison_af_heart.mp4",
        "captioned_name": "comparison_af_heart_captioned.mp4",
        "aliases": ["comparison_voice_1_captioned.mp4"],
    },
    {
        "key": "af_bella",
        "audio_path": Path("test_voices/voice_2_af_bella.wav"),
        "non_captioned_name": "comparison_voice_2.mp4",
        "captioned_name": "comparison_voice_2_captioned.mp4",
        "aliases": ["comparison_af_bella_captioned.mp4", "comparison_af_bella.mp4"],
    },
    {
        "key": "am_adam",
        "audio_path": Path("test_voices/voice_3_am_adam.wav"),
        "non_captioned_name": "comparison_voice_3.mp4",
        "captioned_name": "comparison_voice_3_captioned.mp4",
        "aliases": ["comparison_am_adam_captioned.mp4", "comparison_am_adam.mp4"],
    },
]


def load_original_video_data() -> dict:
    """Load render animations and image references from DB (read-only)."""
    with SessionLocal() as db:
        video = db.query(Video).filter(Video.id == VIDEO_ID).first()
        if not video:
            raise RuntimeError(f"Video {VIDEO_ID} not found in database.")
        
        render_step = next((s for s in video.steps if s.name == "render"), None)
        if not render_step or not isinstance(render_step.output, dict):
            raise RuntimeError("Original render step output not found.")
        
        animations = render_step.output["render"]["animations"]
        logger.info("Loaded %d animations from original render output", len(animations))
        return animations


def render_single_voice(
    voice_info: dict,
    animations: list[dict],
    storage_root: Path,
    out_dir: Path,
    tokens: list[str],
) -> dict:
    """Render non-captioned and captioned videos for a single voice."""
    v_key = voice_info["key"]
    wav_path = storage_root / voice_info["audio_path"]
    if not wav_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    audio_info = sf.info(str(wav_path))
    wav_duration = float(audio_info.duration)
    target_fps = 30
    target_total_frames = int(round(wav_duration * target_fps))

    logger.info("[%s] Duration: %.3fs -> Target frames: %d", v_key, wav_duration, target_total_frames)

    # Calculate proportional beat frames based on original 1119 frames
    orig_total_frames = sum(a["frames"] for a in animations)
    beat_frames = []
    for a in animations:
        f = max(1, int(round((a["frames"] / float(orig_total_frames)) * target_total_frames)))
        beat_frames.append(f)
    
    # Adjust final beat so sum exactly equals target_total_frames
    diff = target_total_frames - sum(beat_frames)
    beat_frames[-1] = max(1, beat_frames[-1] + diff)
    assert sum(beat_frames) == target_total_frames, f"Frame sum mismatch: {sum(beat_frames)} != {target_total_frames}"

    tmp_dir = out_dir / f"tmp_{v_key}_{uuid.uuid4().hex[:6]}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    segment_paths = []
    try:
        # Step 1: Render 13 video segments with zoompan
        logger.info("[%s] Rendering %d video segments...", v_key, len(animations))
        curr_frame = 0
        for i, (a, b_frames) in enumerate(zip(animations, beat_frames), 1):
            b_id = a.get("beat_id", f"beat_{i}")
            b_dur = round(b_frames / float(target_fps), 4)
            img_rel = a["image_file"]
            img_abs = storage_root / img_rel
            if not img_abs.is_file():
                raise FileNotFoundError(f"Image not found: {img_abs}")

            seg_path = tmp_dir / f"seg_{i:02d}_{b_id}.mp4"

            # Build FFmpeg zoompan filter
            z_expr, x_expr, y_expr, filter_chain, resolved_anim = build_ffmpeg_zoompan_filter(
                plan=a,
                scene_dur=b_dur,
                target_fps=target_fps,
                target_width=1080,
                target_height=1920,
            )

            cmd = [
                "ffmpeg",
                "-y",
                "-i", str(img_abs),
                "-filter_complex", f"[0:v]{filter_chain}[v]",
                "-map", "[v]",
                "-frames:v", str(b_frames),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "19",
                "-pix_fmt", "yuv420p",
                "-an",
                str(seg_path),
            ]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            segment_paths.append(seg_path)
            curr_frame += b_frames

        # Step 2: Concatenate segments
        logger.info("[%s] Concatenating segments...", v_key)
        concat_txt = tmp_dir / "concat_list.txt"
        concat_txt.write_text("\n".join(f"file '{p.resolve()}'" for p in segment_paths), encoding="utf-8")
        temp_video_mp4 = tmp_dir / "temp_video.mp4"
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", str(temp_video_mp4)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )

        # Step 3: Mux with continuous narration audio
        logger.info("[%s] Muxing narration audio...", v_key)
        non_captioned_path = out_dir / voice_info["non_captioned_name"]
        mux_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(temp_video_mp4),
            "-i", str(wav_path),
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            str(non_captioned_path),
        ]
        subprocess.run(mux_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Step 4: Subtitle Forced Alignment with Vosk
        logger.info("[%s] Performing Vosk forced alignment...", v_key)
        dur, aligned_words = align_script_with_vosk(wav_path, tokens)
        
        sub_result = SubtitleResult(
            segments=[
                SubtitleSegment(
                    index=1,
                    text=SCRIPT,
                    start=0.0,
                    end=dur,
                    words=aligned_words,
                    timing_source="forced_alignment",
                )
            ],
            total_words=len(aligned_words),
            total_duration_seconds=dur,
            provider="acoustic",
            model="vosk",
            timing_source="forced_alignment",
        )
        ass_path = out_dir / f"comparison_{v_key}.ass"
        ass_path.write_text(sub_result.to_ass(), encoding="utf-8")

        # Step 5: Burn captions
        logger.info("[%s] Burning captions into video...", v_key)
        captioned_path = out_dir / voice_info["captioned_name"]
        burn_cmd = [
            "ffmpeg",
            "-y",
            "-i", str(non_captioned_path),
            "-vf", f"ass={ass_path.resolve()}",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            str(captioned_path),
        ]
        subprocess.run(burn_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

        # Create alias copies for convenience
        for alias in voice_info.get("aliases", []):
            alias_path = out_dir / alias
            if not alias_path.exists():
                shutil.copy2(str(captioned_path if "captioned" in alias else non_captioned_path), str(alias_path))

        # Probe final files
        probe_non_cap = json.loads(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration,size:stream=width,height,r_frame_rate,codec_name", "-of", "json", str(non_captioned_path)]
        ))
        probe_cap = json.loads(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration,size:stream=width,height,r_frame_rate,codec_name", "-of", "json", str(captioned_path)]
        ))

        return {
            "key": v_key,
            "non_captioned_path": str(non_captioned_path),
            "captioned_path": str(captioned_path),
            "ass_path": str(ass_path),
            "duration": float(probe_cap["format"]["duration"]),
            "size_bytes": int(probe_cap["format"]["size"]),
            "width": probe_cap["streams"][0]["width"],
            "height": probe_cap["streams"][0]["height"],
            "fps": probe_cap["streams"][0]["r_frame_rate"],
            "words_aligned": len(aligned_words),
        }

    finally:
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir, ignore_errors=True)


def main():
    start_total = time.perf_counter()
    backend_root = Path(__file__).resolve().parent.parent
    storage_root = backend_root / "storage"
    out_dir = storage_root / "comparison_videos"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading original video animations...")
    animations = load_original_video_data()

    tokens = _extract_display_tokens(SCRIPT)
    logger.info("Script tokens count: %d", len(tokens))

    results = []
    for v_info in VOICES:
        logger.info("========================================")
        logger.info("Starting render for voice: %s", v_info["key"])
        t0 = time.perf_counter()
        res = render_single_voice(v_info, animations, storage_root, out_dir, tokens)
        elapsed = time.perf_counter() - t0
        res["render_time"] = round(elapsed, 2)
        results.append(res)
        logger.info("Finished %s in %.2fs. Captioned size: %.2f MB", v_info["key"], elapsed, res["size_bytes"] / 1024**2)

    total_elapsed = round(time.perf_counter() - start_total, 2)
    logger.info("========================================")
    logger.info("ALL RENDERS COMPLETE in %.2fs!", total_elapsed)
    print("\nRESULTS_JSON:" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
