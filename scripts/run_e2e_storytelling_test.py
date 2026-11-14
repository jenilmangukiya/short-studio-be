"""Real End-to-End Visual Storytelling Test Script.

Orchestrates a complete real video production run:
1. script (LLM: gpt-6-luna) - target 40-45s spoken narration
2. storyboard (LLM: gpt-6-luna) - emotion-aware dynamic visual beats (~2-5s)
3. images (OpenAI: gpt-image-2) - unique images per visual beat
4. voiceover (Kokoro: af_heart, continuous narration WAV)
5. render (FFmpeg: Ken Burns camera motion & continuous audio muxing)
6. subtitles (Vosk: forced acoustic word alignment & ASS burn-in)

Usage:
    .venv/bin/python scripts/run_e2e_storytelling_test.py
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.db import SessionLocal
from app.models import Channel, Video, VideoStep
from app.pipeline import approve_step, claim_next_step, execute, init_steps


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
    print("=" * 80)
    print("SHORTS STUDIO - REAL END-TO-END VISUAL STORYTELLING TEST RUN")
    print("=" * 80)

    topic = "Why Do We Sometimes Feel Like We're Falling Just Before Falling Asleep?"
    description = (
        "A faceless educational video explaining the hypnic jerk: why we feel a sudden sensation "
        "of falling just before falling asleep. Target 40–45 seconds of natural spoken narration. "
        "Open with curiosity and the universal sensation of jolting awake. Contrast the common "
        "misconception (that your heart stopped or your soul left your body) with the real "
        "neurological explanation: the brain misinterpreting muscle relaxation as a freefall. "
        "Dynamic visual storytelling with curiosity, surprise, misconception, explanation, "
        "and a strong conclusion."
    )

    channel_id = "ecc7a8f2-b71d-4a35-bbea-3c4446caaa8a"
    storage_root = getattr(settings, "resolved_storage_dir", Path(settings.storage_dir).resolve())

    with SessionLocal() as db:
        channel = db.query(Channel).filter(Channel.id == channel_id).first()
        if not channel:
            print(f"Error: Channel {channel_id} not found in database.")
            sys.exit(1)

        print(f"Channel:      {channel.name} ({channel.platform})")
        print(f"Voice Speed:  {channel.settings.get('voice_speed', 1.0)}")
        print(f"LLM Provider: {settings.llm_provider} ({settings.openai_model})")
        print(f"Img Provider: {settings.image_provider} ({settings.openai_image_model}, quality: {settings.openai_image_quality})")
        print(f"TTS Provider: {settings.voiceover_provider} ({settings.voiceover_default_voice})")
        print(f"Sub Provider: {settings.subtitles_provider} ({settings.vosk_model_path})")
        print(f"Safety Limits: max_unique={settings.image_max_unique_per_video}, max_attempts={settings.image_max_generation_attempts}")
        print("-" * 80)

        # 1. Create Video
        video = Video(
            channel_id=channel.id,
            topic=topic,
            description=description,
        )
        db.add(video)
        db.commit()
        db.refresh(video)

        video_id = str(video.id)
        print(f"\n[INIT] Created Video {video_id}")
        print(f"       Topic: {video.topic}")

        # 2. Init Steps
        init_steps(db, video)
        print(f"       Initialized {len(video.steps)} pipeline steps.")

    # Execute step-by-step
    steps_to_run = ["script", "storyboard", "images", "voiceover", "render", "subtitles"]

    for step_name in steps_to_run:
        print("\n" + "=" * 80)
        print(f"RUNNING STEP: {step_name.upper()}")
        print("=" * 80)
        step_t0 = time.perf_counter()

        with SessionLocal() as db:
            claimed = claim_next_step(db, video_id)
            print(f"Claimed step: {claimed.name} (status: {claimed.status})")

        # Execute
        execute(video_id, step_name, run_all=False)
        step_elapsed = time.perf_counter() - step_t0

        with SessionLocal() as db:
            video = db.query(Video).filter(Video.id == video_id).first()
            step_record = next(s for s in video.steps if s.name == step_name)

            if step_record.status == "failed":
                print(f"\n[ERROR] Step '{step_name}' failed after {step_elapsed:.2f}s:")
                print(f"        {step_record.error}")
                sys.exit(1)

            if step_record.status == "awaiting_approval":
                print(f"Step '{step_name}' finished and is awaiting approval. Approving...")
                approve_step(db, video, step_name)
                db.refresh(step_record)

            print(f"Step '{step_name}' completed with status '{step_record.status}' in {step_elapsed:.2f}s.")

            # Step-specific telemetry printing
            out = step_record.output or {}
            if step_name == "script":
                scr = out.get("script", {})
                meta = out.get("meta", {})
                narration = scr.get("narration", "")
                word_count = len(narration.split())
                est_dur = meta.get("estimated_duration_seconds")
                print(f"  Hook:     {scr.get('hook')}")
                print(f"  Angle:    {scr.get('angle')}")
                print(f"  Words:    {word_count} words")
                print(f"  Est Dur:  {est_dur}s")
                print(f"  Narration:\n  \"{narration}\"")

            elif step_name == "storyboard":
                sb = out.get("storyboard", {})
                meta = out.get("meta", {})
                scenes = sb.get("scenes", [])
                total_beats = sum(len(s.get("visual_beats", [])) for s in scenes)
                print(f"  Scenes:        {len(scenes)}")
                print(f"  Visual Beats:  {total_beats}")
                print(f"  Anchor:        {sb.get('subject_anchor')}")
                for s in scenes:
                    s_idx = s.get("index")
                    s_beats = s.get("visual_beats", [])
                    print(f"    Scene {s_idx} ({len(s_beats)} beats, {s.get('shot_type')}, {s.get('camera_angle')}): \"{s.get('narration')[:50]}...\"")
                    for b in s_beats:
                        b_ord = b.get("order")
                        print(f"      Beat {s_idx}.{b_ord} ({b.get('shot_type')}, {b.get('camera_angle')}): {b.get('visual_purpose') or b.get('visual_description')[:60]}")

            elif step_name == "images":
                img_data = out.get("images", {})
                items = img_data.get("items", [])
                meta = out.get("meta", {})
                print(f"  Total Items:       {len(items)}")
                print(f"  Generated:         {meta.get('images_generated')}")
                print(f"  Reused:            {meta.get('images_reused')}")
                print(f"  Failed:            {meta.get('images_failed')}")
                print(f"  Attempts:          {meta.get('generation_attempts')}")
                print(f"  Planned Cost:      ${meta.get('planned_est_cost_usd')}")
                print(f"  Total Cost:        ${meta.get('est_cost_usd')}")
                print(f"  Tokens:            in={meta.get('input_tokens')}, out={meta.get('output_tokens')}, tot={meta.get('total_tokens')}")
                for it in items:
                    print(f"    Item {it.get('index')}/{it.get('id')}: status={it.get('status')}, reused={it.get('reused')}, file={it.get('file')}")

            elif step_name == "voiceover":
                vo_data = out.get("voiceover", {})
                meta = out.get("meta", {})
                items = vo_data.get("items", [])
                print(f"  Audio File:        {vo_data.get('audio_file')}")
                print(f"  Duration:          {meta.get('total_duration')}s")
                print(f"  Voice:             {meta.get('voice')} at {meta.get('speed')}x")
                print(f"  Scene Segments:    {len(items)}")

            elif step_name == "render":
                rend_data = out.get("render", {})
                meta = out.get("meta", {})
                print(f"  Render Mode:       {meta.get('render_mode')}")
                print(f"  Video File:        {rend_data.get('video_file')}")
                print(f"  Resolution:        {rend_data.get('width')}x{rend_data.get('height')} @ {rend_data.get('fps')}fps")
                print(f"  Duration:          {meta.get('video_duration')}s")
                print(f"  Render Latency:    {meta.get('render_latency_seconds')}s")

            elif step_name == "subtitles":
                sub_data = out.get("subtitles", {})
                meta = out.get("meta", {})
                print(f"  Timing Source:     {meta.get('timing_source')}")
                print(f"  Total Words:       {meta.get('word_count')}")
                print(f"  Captioned Video:   {sub_data.get('captioned_video_file')}")
                print(f"  Match Ratio:       {meta.get('match_ratio')}")

    # Inspect final captioned video
    print("\n" + "=" * 80)
    print("FINAL VERIFICATION OF CAPTIONED VIDEO")
    print("=" * 80)

    with SessionLocal() as db:
        video = db.query(Video).filter(Video.id == video_id).first()
        sub_step = next(s for s in video.steps if s.name == "subtitles")
        cap_file_rel = sub_step.output.get("subtitles", {}).get("captioned_video_file")
        cap_file_abs = storage_root / cap_file_rel

        if not cap_file_abs.is_file():
            print(f"Error: Captioned video not found at {cap_file_abs}")
            sys.exit(1)

        probe = run_ffprobe(cap_file_abs)
        fmt = probe.get("format", {})
        streams = probe.get("streams", [])
        v_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
        a_stream = next((s for s in streams if s.get("codec_type") == "audio"), {})

        duration_sec = float(fmt.get("duration", 0.0))
        size_bytes = int(fmt.get("size", 0))

        print(f"Final Video File: {cap_file_abs}")
        print(f"File Size:        {size_bytes / (1024 * 1024):.2f} MB ({size_bytes:,} bytes)")
        print(f"Duration:         {duration_sec:.2f} seconds")
        print(f"Video Stream:     {v_stream.get('codec_name')}, {v_stream.get('width')}x{v_stream.get('height')}, {v_stream.get('r_frame_rate')} fps")
        print(f"Audio Stream:     {a_stream.get('codec_name')}, {a_stream.get('channels')}ch, {a_stream.get('sample_rate')} Hz")
        print("=" * 80)
        print("SUCCESSFULLY COMPLETED END-TO-END RUN")
        print(f"Video ID: {video_id}")
        print("=" * 80)


if __name__ == "__main__":
    main()
