import json
import time
import uuid
from pathlib import Path
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy import select

from app.db import SessionLocal
from app.models import Video, VideoStep
from app.steps.render import generate_render_step
from app.steps.subtitles import generate_subtitles_step

vid = uuid.UUID("e8342d48-2150-4b46-b05e-88c454f1fc87")

with SessionLocal() as session:
    video = session.execute(select(Video).where(Video.id == vid)).scalar_one_or_none()
    steps = session.execute(select(VideoStep).where(VideoStep.video_id == vid)).scalars().all()
    step_map = {s.name: s for s in steps}

    sb_step = step_map["storyboard"]
    img_step = step_map["images"]
    vo_step = step_map["voiceover"]
    render_step = step_map["render"]
    sub_step = step_map["subtitles"]

    print("=== Step 1: Pre-flight Verification ===")
    print(f"Video ID: {video.id}")
    print(f"Images step status: {img_step.status}")
    img_items = img_step.output.get("images", {}).get("items", [])
    print(f"Total image items: {len(img_items)}")
    for it in img_items:
        print(f" - {it.get('id') or it.get('beat_id')}: {it.get('file')} ({it.get('status')})")

    # Prepare prev for render
    prev_render = {
        "storyboard": sb_step.output,
        "images": img_step.output,
        "voiceover": vo_step.output,
    }

    print("\n=== Step 2: Executing Video Render Step ===")
    render_start = time.time()
    render_step.status = "running"
    render_step.error = None
    session.commit()

    render_out = generate_render_step(video, prev_render, render_step)
    render_time = time.time() - render_start
    print(f"Render step completed in {render_time:.2f}s")
    render_step.output = render_out
    render_step.status = "done"
    video.output_path = render_out.get("video_file")
    flag_modified(render_step, "output")
    session.commit()

    print("\n=== Step 3: Executing Subtitles Step ===")
    sub_start = time.time()
    sub_step.status = "running"
    sub_step.error = None
    session.commit()

    prev_sub = {
        "storyboard": sb_step.output,
        "voiceover": vo_step.output,
        "render": render_out,
    }
    sub_out = generate_subtitles_step(video, prev_sub, sub_step)
    sub_time = time.time() - sub_start
    print(f"Subtitles step completed in {sub_time:.2f}s")
    sub_step.output = sub_out
    sub_step.status = "done"
    flag_modified(sub_step, "output")
    video.status = "completed"
    session.commit()

    print("\n=== Rendering Pipeline Finished Successfully! ===")
    print(f"Render output: {render_out.get('video_file')}")
    print(f"Captioned output: {sub_out.get('captioned_video_file')}")
