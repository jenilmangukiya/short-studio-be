"""Eight-Step Real End-to-End Pipeline Test Script.

Orchestrates a complete real video production run using the 8-step pipeline:
1. creative_direction (LLM: gpt-6-luna) - narrative format, visual genre, protagonist config
2. script (LLM: gpt-6-luna) - target 40-45s spoken narration with epistemic nuance
3. video_bible (LLM: gpt-6-luna) - entities catalog (chars, envs, objects, concepts) & art direction
4. storyboard (LLM: gpt-6-luna + code prompt composition + validation) - connected directorial beats
5. images (OpenAI: gpt-image-2) - vertical 9:16 images generated via composed prompts
6. voiceover (Kokoro: am_adam) - continuous narration WAV
7. render (FFmpeg) - Ken Burns camera motion & continuous audio muxing
8. subtitles (Vosk) - forced acoustic word alignment & ASS burn-in

Usage:
    .venv/bin/python scripts/run_e2e_eight_step_test.py
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
    print("SHORTS STUDIO - EIGHT-STEP PIPELINE REAL END-TO-END TEST")
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

        # 2. Init Steps (Eight-Step sequence)
        init_steps(db, video)
        print(f"       Initialized {len(video.steps)} pipeline steps: {[s.name for s in video.steps]}")

    steps_to_run = [
        "creative_direction",
        "script",
        "video_bible",
        "storyboard",
        "images",
        "voiceover",
        "render",
        "subtitles",
    ]

    outputs: dict[str, dict] = {}

    for step_name in steps_to_run:
        print("\n" + "=" * 80)
        print(f"RUNNING STEP: {step_name.upper()}")
        print("=" * 80)
        step_t0 = time.perf_counter()

        with SessionLocal() as db:
            claimed = claim_next_step(db, video_id)
            print(f"Claimed step: {claimed.name} (position: {claimed.position}, status: {claimed.status})")

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

            out = step_record.output or {}
            outputs[step_name] = out

            # Telemetry logging per stage
            if step_name == "creative_direction":
                cd = out.get("creative_direction", {})
                print(f"  Format Archetype:    {cd.get('format_archetype')}")
                print(f"  Storytelling Angle:  {cd.get('storytelling_angle')}")
                print(f"  Central Thesis:      {cd.get('central_thesis')}")
                print(f"  Tone:                {cd.get('tone')}")
                print(f"  Visual Genre:        {cd.get('visual_genre')}")
                prot = cd.get("protagonist_config", {})
                print(f"  Protagonist Config:  human_presence={prot.get('human_presence')}, summary={prot.get('archetype_summary')}")
                print(f"  Constraints:         {cd.get('creative_constraints')}")

            elif step_name == "script":
                scr = out.get("script", {})
                meta = out.get("meta", {})
                narration = scr.get("narration", "")
                word_count = len(narration.split())
                print(f"  Title:     {scr.get('title')}")
                print(f"  Hook:      {scr.get('hook')}")
                print(f"  Angle:     {scr.get('angle')}")
                print(f"  Words:     {word_count} words")
                print(f"  Duration:  {meta.get('estimated_duration_seconds')}s")
                print(f"  Narration:\n  \"{narration}\"")

            elif step_name == "video_bible":
                vb = out.get("video_bible", {})
                entities = vb.get("entities", {})
                art = vb.get("art_direction", {})
                chars = entities.get("characters", [])
                envs = entities.get("environments", [])
                objs = entities.get("objects_and_props", []) or entities.get("objects", [])
                cpts = entities.get("concepts_and_diagrams", []) or entities.get("concepts", [])
                print(f"  Topic Summary:       {vb.get('topic_summary')}")
                print(f"  Art Visual Genre:    {art.get('visual_genre')}")
                print(f"  Lighting Philosophy: {art.get('lighting_philosophy')}")
                print(f"  Texture:             {art.get('rendering_texture')}")
                print(f"  Negative Anchors:    {art.get('negative_anchors')}")
                print(f"  Entities Summary:")
                print(f"    - Characters ({len(chars)}): {[c.get('name') for c in chars]}")
                print(f"    - Environments ({len(envs)}): {[e.get('name') for e in envs]}")
                print(f"    - Objects ({len(objs)}): {[o.get('name') for o in objs]}")
                print(f"    - Concepts ({len(cpts)}): {[cp.get('name') for cp in cpts]}")

            elif step_name == "storyboard":
                sb = out.get("storyboard", {})
                meta = out.get("meta", {})
                scenes = sb.get("scenes", [])
                beats = sb.get("beats", [])
                warnings = meta.get("warnings", [])
                print(f"  Scenes Count:        {len(scenes)}")
                print(f"  Beats Count:         {len(beats)}")
                print(f"  Total Duration:      {sb.get('total_seconds')}s")
                print(f"  Validation Warnings: {len(warnings)}")
                for w in warnings:
                    print(f"    [WARN] {w}")
                print("\n  Directorial Storyboard Beats:")
                for b_idx, b in enumerate(beats, start=1):
                    eb = b.get("entity_bindings", {})
                    cam = b.get("camera", {})
                    cs = b.get("character_state")
                    cs_str = f"({cs.get('emotion')}, {cs.get('action_in_progress')})" if cs else "None"
                    print(f"    Beat {b_idx} [Scene {b.get('scene_index')}.{b.get('order')} | {b.get('duration_est_seconds')}s]:")
                    print(f"      Narration:    \"{b.get('narration_slice')[:60]}...\"")
                    print(f"      Transition:   {b.get('transition_from_previous')}")
                    print(f"      Entities:     char={eb.get('character_id')}, env={eb.get('environment_id')}, concept={eb.get('concept_id')}, objs={eb.get('object_ids')}")
                    print(f"      Camera:       {cam.get('shot_type')} / {cam.get('camera_angle')} -> {cam.get('focal_point_description')[:50]}")
                    print(f"      Char State:   {cs_str}")

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
                for it in items:
                    print(f"    Item {it.get('index')}/{it.get('id')}: status={it.get('status')}, reused={it.get('reused')}, file={it.get('file')}")

            elif step_name == "voiceover":
                vo_data = out.get("voiceover", {}) or {}
                meta = out.get("meta", {}) or {}
                audio_f = out.get("audio_file") or vo_data.get("audio_file")
                dur = out.get("duration_seconds") or vo_data.get("duration_seconds")
                voice_name = meta.get("voice") or vo_data.get("voice") or getattr(settings, "voiceover_default_voice", "am_adam")
                speed_val = meta.get("speed") or vo_data.get("speed") or 0.94
                print(f"  Audio File:        {audio_f}")
                print(f"  Duration:          {dur}s")
                print(f"  Voice:             {voice_name} at {speed_val}x")

            elif step_name == "render":
                rend_data = out.get("render", {}) or {}
                meta = out.get("meta", {}) or {}
                v_path = out.get("video_path") or rend_data.get("video_path")
                res = rend_data.get("resolution") or (f"{rend_data.get('width')}x{rend_data.get('height')}" if rend_data.get('width') else "1080x1920")
                dur = out.get("duration_seconds") or rend_data.get("duration_seconds")
                print(f"  Render Mode:       {meta.get('render_mode', rend_data.get('render_mode'))}")
                print(f"  Video File:        {v_path}")
                print(f"  Resolution:        {res} @ {rend_data.get('fps', 30)}fps")
                print(f"  Duration:          {dur}s")

            elif step_name == "subtitles":
                sub_data = out.get("subtitles", {}) or {}
                meta = out.get("meta", {}) or {}
                cap_f = out.get("captioned_video_path") or sub_data.get("captioned_file") or sub_data.get("captioned_video_path")
                print(f"  Timing Source:     {meta.get('timing_source', sub_data.get('timing_source'))}")
                print(f"  Total Words:       {meta.get('word_count', sub_data.get('total_words'))}")
                print(f"  Captioned Video:   {cap_f}")
                print(f"  Match Ratio:       {meta.get('match_ratio', 1.0)}")

    # Final Video Analysis
    print("\n" + "=" * 80)
    print("FINAL VERIFICATION OF CAPTIONED VIDEO")
    print("=" * 80)

    with SessionLocal() as db:
        video = db.query(Video).filter(Video.id == video_id).first()
        sub_step = next(s for s in video.steps if s.name == "subtitles")
        sub_out = sub_step.output or {}
        sub_data = sub_out.get("subtitles", {}) or {}
        cap_path_str = sub_out.get("captioned_video_path") or sub_data.get("captioned_file") or sub_data.get("captioned_video_path")
        if not cap_path_str:
            print("Error: Captioned video path not found in subtitle output.")
            sys.exit(1)

        cap_file_abs = Path(cap_path_str) if Path(cap_path_str).is_absolute() else (storage_root / cap_path_str)

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
        print("EIGHT-STEP PIPELINE RUN COMPLETE")
        print(f"Video ID: {video_id}")
        print("=" * 80)

        # Save summary report to JSON
        report = {
            "video_id": video_id,
            "topic": video.topic,
            "final_video_path": str(cap_file_abs),
            "file_size_mb": round(size_bytes / (1024 * 1024), 2),
            "duration_seconds": round(duration_sec, 2),
            "video_stream": {
                "codec": v_stream.get("codec_name"),
                "resolution": f"{v_stream.get('width')}x{v_stream.get('height')}",
                "fps": v_stream.get("r_frame_rate"),
            },
            "audio_stream": {
                "codec": a_stream.get("codec_name"),
                "sample_rate": a_stream.get("sample_rate"),
                "channels": a_stream.get("channels"),
            },
            "outputs": outputs,
        }

        report_path = storage_root / video_id / "e2e_eight_step_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Summary JSON saved to {report_path}")


if __name__ == "__main__":
    main()
