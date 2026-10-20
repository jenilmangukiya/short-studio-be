"""Pipeline engine. Handlers are MOCKS for now (zero AI cost).
To go real, replace a handler with one that calls your LLM/image/TTS provider
and returns a JSON-serialisable dict. Signature: (video, prev_outputs, step) -> dict
"""
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from .config import settings
from .db import SessionLocal
from .models import Video, VideoStep

STEPS = ["script", "storyboard", "images", "voiceover", "render"]


class PipelineError(Exception):
    pass


def _now():
    return datetime.now(timezone.utc)


def _dir(video: Video) -> Path:
    p = Path(settings.storage_dir) / str(video.id)
    p.mkdir(parents=True, exist_ok=True)
    return p


from .steps.script import generate_script_step


# ---------- handlers ----------
def h_script(video, prev, step):
    return generate_script_step(video, prev, step)


def h_storyboard(video, prev, step):
    scenes = [
        {"index": i, "seconds": 5, "narration": f"Scene {i} narration about {video.topic}",
         "image_prompt": f"{video.topic}, scene {i}, cinematic"}
        for i in range(1, 9)
    ]
    return {"mock": True, "scenes": scenes, "total_seconds": sum(s["seconds"] for s in scenes)}


def h_images(video, prev, step):
    d = _dir(video) / "images"
    d.mkdir(exist_ok=True)
    out = []
    for s in prev["storyboard"]["scenes"]:
        f = d / f"scene_{s['index']}.txt"
        f.write_text(f"placeholder image for: {s['image_prompt']}")
        out.append({"scene": s["index"], "path": str(f)})
    return {"mock": True, "images": out}


def h_voiceover(video, prev, step):
    f = _dir(video) / "voice.txt"
    f.write_text("placeholder voiceover audio")
    return {"mock": True, "audio_path": str(f), "duration_seconds": prev["storyboard"]["total_seconds"]}


def h_render(video, prev, step):
    f = _dir(video) / "final_placeholder.txt"
    f.write_text(f"Placeholder for final video.\nTopic: {video.topic}\nScenes: {len(prev['images']['images'])}")
    video.output_path = str(f)
    return {"mock": True, "video_path": str(f)}


HANDLERS = {
    "script": h_script, "storyboard": h_storyboard, "images": h_images,
    "voiceover": h_voiceover, "render": h_render,
}


# ---------- engine ----------
def init_steps(db: Session, video: Video):
    for i, name in enumerate(STEPS):
        db.add(VideoStep(video_id=video.id, name=name, position=i, status="pending"))
    video.status = "in_progress"
    db.commit()
    db.refresh(video)


def claim_next_step(db: Session, video_id) -> VideoStep:
    """Lock the video row, pick the first non-done step, mark it running."""
    video = db.scalar(
        select(Video).where(Video.id == video_id).with_for_update().execution_options(populate_existing=True)
    )
    if not video.steps:
        raise PipelineError("Video not started yet")
    if any(s.status == "running" for s in video.steps):
        raise PipelineError("A step is already running")
    step = next((s for s in video.steps if s.status != "done"), None)
    if not step:
        raise PipelineError("All steps already completed")
    step.status, step.error, step.started_at = "running", None, _now()
    video.status = "in_progress"
    db.commit()
    return step


def execute(video_id, step_name: str, run_all: bool = False):
    """Runs in a background task with its own session (later: move to a worker queue)."""
    with SessionLocal() as db:
        name = step_name
        while name:
            video = db.scalar(
                select(Video).where(Video.id == video_id).execution_options(populate_existing=True)
            )
            step = next(s for s in video.steps if s.name == name)
            prev = {s.name: s.output for s in video.steps if s.status == "done" and s.output}
            try:
                step.output = HANDLERS[name](video, prev, step)
                step.status, step.error = "done", None
            except Exception as exc:  # noqa: BLE001
                step.status, step.error, step.finished_at = "failed", str(exc)[:2000], _now()
                video.status = "failed"
                db.commit()
                return
            step.finished_at = _now()
            nxt = next((s for s in video.steps if s.status != "done"), None)
            video.status = "in_progress" if nxt else "completed"
            if nxt and run_all:
                nxt.status, nxt.error, nxt.started_at = "running", None, _now()
                name = nxt.name
            else:
                name = None
            db.commit()


def rerun_step(db: Session, video: Video, step_name: str, prompt: str | None):
    """Reset one step (with optional revision prompt) and mark downstream steps stale."""
    if any(s.status == "running" for s in video.steps):
        raise PipelineError("A step is running; wait for it to finish")
    target = next((s for s in video.steps if s.name == step_name), None)
    if not target:
        raise PipelineError("Video not started yet")
    for s in video.steps:
        if s.position == target.position:
            s.status, s.user_prompt, s.error = "pending", prompt, None
        elif s.position > target.position and s.status == "done":
            s.status = "stale"
    video.status = "in_progress"
    video.output_path = None
    db.commit()
