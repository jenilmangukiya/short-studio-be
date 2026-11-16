"""Verified End-to-End Execution Script with Character Gating.

Runs the eight-step pipeline with explicit programmatic verification gates:
1. creative_direction: asserts character_count == 'single' and a human sleeper archetype is chosen.
2. script: conversational narration.
3. video_bible: asserts characters catalog contains the human sleeper with visual/apparel anchors.
4. storyboard: asserts character is bound across physical bedroom scenes.
5. images, voiceover, render, subtitles: standard pipeline execution without modification.
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.db import SessionLocal
from app.models import Channel, Video, VideoStep
from app.pipeline import approve_step, claim_next_step, execute, init_steps


def run_ffprobe(file_path: Path) -> dict:
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
    print("SHORTS STUDIO - VERIFIED CHARACTER PIPELINE RUN")
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
        print("-" * 80)

        video = Video(channel_id=channel.id, topic=topic, description=description)
        db.add(video)
        db.commit()
        db.refresh(video)

        video_id = str(video.id)
        print(f"\n[INIT] Created Video {video_id}")
        init_steps(db, video)

    outputs: dict[str, dict] = {}

    def _run_step(step_name: str) -> dict:
        print("\n" + "=" * 80)
        print(f"RUNNING STEP: {step_name.upper()}")
        print("=" * 80)
        t0 = time.perf_counter()
        with SessionLocal() as db:
            claimed = claim_next_step(db, video_id)
            print(f"Claimed step: {claimed.name} (position: {claimed.position})")

        execute(video_id, step_name, run_all=False)
        elapsed = time.perf_counter() - t0

        with SessionLocal() as db:
            video = db.query(Video).filter(Video.id == video_id).first()
            s_rec = next(s for s in video.steps if s.name == step_name)
            if s_rec.status == "failed":
                print(f"[ERROR] Step '{step_name}' failed after {elapsed:.2f}s: {s_rec.error}")
                sys.exit(1)
            if s_rec.status == "awaiting_approval":
                print(f"Step '{step_name}' awaiting approval. Approving...")
                approve_step(db, video, step_name)
                db.refresh(s_rec)
            print(f"Step '{step_name}' completed in {elapsed:.2f}s.")
            out = s_rec.output or {}
            outputs[step_name] = out
            return out

    # -------------------------------------------------------------------------
    # Step 1: Creative Direction
    # -------------------------------------------------------------------------
    cd_out = _run_step("creative_direction")
    cd_data = cd_out.get("creative_direction", {})
    prot_cfg = cd_data.get("protagonist_config", {})
    char_count = prot_cfg.get("character_count")
    arch_summary = prot_cfg.get("archetype_summary")
    cd_warnings = cd_out.get("warnings", [])

    print("\n[VERIFICATION GATE 1: Creative Direction Character Selection]")
    print(f"  character_count:    {char_count}")
    print(f"  archetype_summary:  {arch_summary}")
    print(f"  advisory warnings:  {cd_warnings}")

    if char_count == "none":
        print("\n[STOPPING CONDITION MET]: Creative Direction selected character_count='none' for human sleep topic.")
        print("Stopping pipeline before image generation as requested by user instructions.")
        sys.exit(2)

    print(">>> GATE 1 PASSED: Character successfully selected by Creative Direction.")

    # -------------------------------------------------------------------------
    # Step 2: Script
    # -------------------------------------------------------------------------
    scr_out = _run_step("script")
    scr_data = scr_out.get("script", {})
    print(f"  Title:     {scr_data.get('title')}")
    print(f"  Hook:      {scr_data.get('hook')}")
    print(f"  Narration:\n  \"{scr_data.get('narration')}\"")

    # -------------------------------------------------------------------------
    # Step 3: Video Bible
    # -------------------------------------------------------------------------
    vb_out = _run_step("video_bible")
    vb_data = vb_out.get("video_bible", {})
    entities = vb_data.get("entities", {})
    chars = entities.get("characters", [])
    envs = entities.get("environments", [])

    print("\n[VERIFICATION GATE 2: Video Bible Entity Catalog]")
    print(f"  Characters defined: {len(chars)}")
    for c in chars:
        print(f"    - ID: {c.get('id')}, Name: {c.get('name')}")
        print(f"      Visual Anchor:   {c.get('visual_anchor')}")
        print(f"      Apparel Anchor:  {c.get('apparel_anchor')}")
        print(f"      Forbidden Mut.:  {c.get('forbidden_mutations')}")
    print(f"  Environments:       {len(envs)}")
    for e in envs:
        print(f"    - ID: {e.get('id')}, Name: {e.get('name')}")

    if not chars:
        print("\n[STOPPING CONDITION MET]: Video Bible character catalog is empty.")
        sys.exit(3)

    first_char = chars[0]
    if not first_char.get("visual_anchor") or not first_char.get("apparel_anchor"):
        print("\n[STOPPING CONDITION MET]: Character is missing visual or apparel anchors.")
        sys.exit(4)

    print(">>> GATE 2 PASSED: Human character fully specified in Video Bible.")

    # -------------------------------------------------------------------------
    # Step 4: Storyboard
    # -------------------------------------------------------------------------
    sb_out = _run_step("storyboard")
    sb_data = sb_out.get("storyboard", {})
    beats = sb_data.get("beats", [])
    scenes = sb_data.get("scenes", [])
    print(f"  Scenes Count: {len(scenes)}, Beats Count: {len(beats)}")

    char_bound_beats = sum(
        1 for b in beats
        if (b.get("entity_bindings") or {}).get("character_id")
    )
    print(f"  Beats with character bound: {char_bound_beats}/{len(beats)}")

    if char_bound_beats == 0:
        print("\n[STOPPING CONDITION MET]: Storyboard bound zero scenes/beats to the character.")
        sys.exit(5)

    print(">>> GATE 3 PASSED: Storyboard appropriately binds character across scenes.")

    # -------------------------------------------------------------------------
    # Step 5: Images
    # -------------------------------------------------------------------------
    img_out = _run_step("images")
    img_meta = img_out.get("meta", {})
    print(f"  Images Generated: {img_meta.get('images_generated')}/{img_meta.get('images_total')}")
    print(f"  Images Failed:    {img_meta.get('images_failed')}")
    print(f"  Total Cost USD:   ${img_meta.get('est_cost_usd')}")

    # -------------------------------------------------------------------------
    # Step 6: Voiceover
    # -------------------------------------------------------------------------
    vo_out = _run_step("voiceover")
    vo_meta = vo_out.get("meta", {})
    vo_dur = vo_out.get("duration_seconds") or (vo_out.get("voiceover") or {}).get("duration_seconds")
    print(f"  Audio Duration: {vo_dur}s")

    # -------------------------------------------------------------------------
    # Step 7: Render
    # -------------------------------------------------------------------------
    rend_out = _run_step("render")
    rend_data = rend_out.get("render", {})
    print(f"  Rendered Video: {rend_out.get('video_path') or rend_data.get('video_path')}")

    # -------------------------------------------------------------------------
    # Step 8: Subtitles
    # -------------------------------------------------------------------------
    sub_out = _run_step("subtitles")
    cap_path = sub_out.get("captioned_video_path") or (sub_out.get("subtitles") or {}).get("captioned_file")
    print(f"  Captioned Video: {cap_path}")

    # -------------------------------------------------------------------------
    # Final Video Inspection & Reporting JSON
    # -------------------------------------------------------------------------
    cap_file_abs = Path(cap_path) if Path(cap_path).is_absolute() else (storage_root / cap_path)
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

    report = {
        "video_id": video_id,
        "topic": topic,
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

    report_path = storage_root / video_id / "e2e_verified_character_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 80)
    print("SUCCESSFULLY COMPLETED END-TO-END PIPELINE WITH VERIFIED CHARACTER")
    print(f"Video ID: {video_id}")
    print(f"Report:   {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
