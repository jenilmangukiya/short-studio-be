import re
import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import pipeline
from ..config import settings
from ..db import get_db
from ..deps import get_owned_channel, get_owned_video
from ..models import Channel, PipelineJob, Video
from ..queue.engine import JobConflictError, enqueue_step_job
from ..schemas import (
    JobOut,
    RerunBody,
    StepApproveBody,
    StepContentUpdate,
    StepRevisionOut,
    VideoCreate,
    VideoDetail,
    VideoSummary,
)

router = APIRouter(tags=["videos"])




@router.post("/channels/{channel_id}/videos", response_model=VideoSummary, status_code=201)
def create_video(body: VideoCreate, channel: Channel = Depends(get_owned_channel), db: Session = Depends(get_db)):
    video = Video(channel_id=channel.id, topic=body.topic, description=body.description)
    db.add(video)
    db.commit()
    return video


@router.get("/channels/{channel_id}/videos", response_model=list[VideoSummary])
def list_videos(channel: Channel = Depends(get_owned_channel), db: Session = Depends(get_db)):
    return db.scalars(select(Video).where(Video.channel_id == channel.id).order_by(Video.created_at.desc())).all()


@router.get("/videos/{video_id}", response_model=VideoDetail)
def get_video(video: Video = Depends(get_owned_video)):
    return video


@router.delete("/videos/{video_id}", status_code=204)
def delete_video(video: Video = Depends(get_owned_video), db: Session = Depends(get_db)):
    db.delete(video)
    db.commit()


@router.post("/videos/{video_id}/start", response_model=VideoDetail)
def start(video: Video = Depends(get_owned_video), db: Session = Depends(get_db)):
    """Creates the pipeline steps. Nothing runs yet; press /next (or /run-all)."""
    if video.steps:
        raise HTTPException(409, "Video already started")
    pipeline.init_steps(db, video)
    return video


@router.post("/videos/{video_id}/next", response_model=VideoDetail, status_code=202)
def next_step(video: Video = Depends(get_owned_video), db: Session = Depends(get_db)):
    """Enqueue the first non-done step for background worker execution.
    Poll GET /videos/{id} for progress."""
    if not video.steps:
        raise HTTPException(409, "Video not started yet")
    step = next((s for s in video.steps if s.status != "done"), None)
    if not step:
        raise HTTPException(409, "All steps already completed")
    try:
        enqueue_step_job(db, video.id, step.name, run_all=False)
    except JobConflictError as e:
        raise HTTPException(409, str(e))
    except pipeline.PipelineError as e:
        raise HTTPException(409, str(e))
    db.refresh(video)
    return video


@router.post("/videos/{video_id}/run-all", response_model=VideoDetail, status_code=202)
def run_all(video: Video = Depends(get_owned_video), db: Session = Depends(get_db)):
    """Enqueue the next step with run_all=True for continuous worker execution.
    Poll GET /videos/{id} for progress."""
    if not video.steps:
        raise HTTPException(409, "Video not started yet")
    step = next((s for s in video.steps if s.status != "done"), None)
    if not step:
        raise HTTPException(409, "All steps already completed")
    try:
        enqueue_step_job(db, video.id, step.name, run_all=True)
    except (JobConflictError, pipeline.PipelineError) as e:
        raise HTTPException(409, str(e))
    db.refresh(video)
    return video


@router.post("/videos/{video_id}/steps/{step_name}/rerun", response_model=VideoDetail)
def rerun(step_name: str, body: RerunBody, video: Video = Depends(get_owned_video), db: Session = Depends(get_db)):
    """Redo one step, optionally with a revision prompt. Later steps become 'stale'."""
    if step_name not in pipeline.STEPS:
        raise HTTPException(404, f"Unknown step. Valid: {pipeline.STEPS}")
    try:
        pipeline.rerun_step(db, video, step_name, body.prompt)
    except pipeline.PipelineError as e:
        raise HTTPException(409, str(e))
    return video


@router.post("/videos/{video_id}/steps/{step_name}/approve", response_model=VideoDetail)
def approve_step(
    step_name: str,
    body: StepApproveBody = StepApproveBody(),
    video: Video = Depends(get_owned_video),
    db: Session = Depends(get_db),
):
    """Approve a step currently awaiting approval and mark it done.
    If auto_advance is True, enqueues the next pipeline step for background worker execution."""
    if step_name not in pipeline.STEPS:
        raise HTTPException(404, f"Unknown step. Valid: {pipeline.STEPS}")
    try:
        pipeline.approve_step(db, video, step_name)
    except pipeline.PipelineError as e:
        raise HTTPException(409, str(e))

    if body.auto_advance:
        next_step = next((s for s in video.steps if s.status != "done"), None)
        if next_step:
            try:
                enqueue_step_job(db, video.id, next_step.name, run_all=False)
            except (JobConflictError, pipeline.PipelineError):
                pass

    db.refresh(video)
    return video


@router.get("/videos/{video_id}/jobs", response_model=list[JobOut])
def list_video_jobs(video: Video = Depends(get_owned_video), db: Session = Depends(get_db)):
    """List all background queue jobs for this video."""
    return db.scalars(
        select(PipelineJob).where(PipelineJob.video_id == video.id).order_by(PipelineJob.scheduled_at.desc())
    ).all()



@router.patch("/videos/{video_id}/steps/{step_name}", response_model=VideoDetail)
def update_step_content(
    step_name: str,
    body: StepContentUpdate,
    video: Video = Depends(get_owned_video),
    db: Session = Depends(get_db),
):
    """Inline editing of step output payload (e.g., narration, scene prompts).
    Automatically saves previous version to history and marks downstream steps stale."""
    if step_name not in pipeline.STEPS:
        raise HTTPException(404, f"Unknown step. Valid: {pipeline.STEPS}")
    try:
        pipeline.update_step_content(db, video, step_name, body.output, body.change_summary)
    except pipeline.PipelineError as e:
        raise HTTPException(409, str(e))

    db.refresh(video)
    return video


@router.get("/videos/{video_id}/steps/{step_name}/history", response_model=list[StepRevisionOut])
def get_step_history(
    step_name: str,
    video: Video = Depends(get_owned_video),
):
    """Retrieve full version history and recorded snapshots for a step."""
    step = next((s for s in video.steps if s.name == step_name), None)
    if not step:
        raise HTTPException(404, f"Step '{step_name}' not found")
    return step.revisions or []


@router.post("/videos/{video_id}/steps/{step_name}/rollback/{version}", response_model=VideoDetail)
def rollback_step(
    step_name: str,
    version: int,
    video: Video = Depends(get_owned_video),
    db: Session = Depends(get_db),
):
    """Rollback a step's output to a previous revision version."""
    if step_name not in pipeline.STEPS:
        raise HTTPException(404, f"Unknown step. Valid: {pipeline.STEPS}")
    try:
        pipeline.rollback_step(db, video, step_name, version)
    except pipeline.PipelineError as e:
        raise HTTPException(409, str(e))

    db.refresh(video)
    return video



@router.get("/videos/{video_id}/download")
def download(
    video: Video = Depends(get_owned_video),
    captioned: bool = True,
):
    storage_root = getattr(settings, "resolved_storage_dir", Path(settings.storage_dir).resolve())
    video_dir = storage_root / str(video.id)
    captioned_file = video_dir / "final_video_captioned.mp4"
    original_file = Path(video.output_path) if video.output_path else (video_dir / "final_video.mp4")

    # Clean topic slug for friendly filename in download header
    topic_slug = re.sub(r"[^a-zA-Z0-9]+", "-", (video.topic or "video").lower()).strip("-") or "video"

    if captioned:
        if captioned_file.is_file() and captioned_file.stat().st_size > 0:
            headers = {
                "Content-Disposition": f'attachment; filename="{topic_slug}-captioned.mp4"'
            }
            return FileResponse(
                path=str(captioned_file),
                media_type="video/mp4",
                filename=f"{topic_slug}-captioned.mp4",
                headers=headers,
            )

        # Fallback to original if captioned is not yet generated
        if original_file.is_file() and original_file.stat().st_size > 0:
            headers = {
                "Content-Disposition": f'attachment; filename="{topic_slug}.mp4"'
            }
            return FileResponse(
                path=str(original_file),
                media_type="video/mp4",
                filename=f"{topic_slug}.mp4",
                headers=headers,
            )

        raise HTTPException(404, "Final video output not available yet")
    else:
        if original_file.is_file() and original_file.stat().st_size > 0:
            headers = {
                "Content-Disposition": f'attachment; filename="{topic_slug}-original.mp4"'
            }
            return FileResponse(
                path=str(original_file),
                media_type="video/mp4",
                filename=f"{topic_slug}-original.mp4",
                headers=headers,
            )
        raise HTTPException(404, "Original uncaptioned video not available yet")


@router.get("/videos/{video_id}/images/{index}")
def get_scene_image(index: int, video: Video = Depends(get_owned_video)):
    """Serve a scene image by integer index if it has status 'ok', regardless of overall step status."""
    step = next((s for s in video.steps if s.name == "images"), None)
    if not step or not step.output or not isinstance(step.output, dict):
        raise HTTPException(404, f"No images found for video {video.id}")

    items = step.output.get("images", {}).get("items", [])
    target = next((it for it in items if it.get("index") == index and it.get("status") == "ok"), None)
    if not target or not target.get("file"):
        raise HTTPException(404, f"No generated image available for scene {index}")

    storage_root = Path(settings.storage_dir).resolve()
    file_rel = target["file"]
    abs_path = (storage_root / file_rel).resolve()

    # Security check: Ensure the resolved path resides strictly within the storage directory
    try:
        abs_path.relative_to(storage_root)
    except ValueError:
        raise HTTPException(403, "Access to requested file path is forbidden")

    if not abs_path.is_file():
        raise HTTPException(404, f"Image file for scene {index} does not exist on disk")

    prompt_hash = target.get("prompt_hash", "")
    headers = {
        "ETag": f'"{prompt_hash}"',
        "Cache-Control": "private, no-cache",
    }
    return FileResponse(path=str(abs_path), media_type="image/png", headers=headers)


@router.get("/videos/{video_id}/audio/{index}")
def get_scene_audio(index: int, video: Video = Depends(get_owned_video)):
    """Serve a scene narration audio by integer index if it has status 'ok', regardless of overall step status."""
    step = next((s for s in video.steps if s.name == "voiceover"), None)
    if not step or not step.output or not isinstance(step.output, dict):
        raise HTTPException(404, f"No voiceover found for video {video.id}")

    items = step.output.get("voiceover", {}).get("items", [])
    target = next((it for it in items if it.get("index") == index and it.get("status") == "ok"), None)
    if not target or not target.get("file"):
        raise HTTPException(404, f"No generated audio available for scene {index}")

    storage_root = Path(settings.storage_dir).resolve()
    file_rel = target["file"]
    abs_path = (storage_root / file_rel).resolve()

    # Security check: Ensure the resolved path resides strictly within the storage directory
    try:
        abs_path.relative_to(storage_root)
    except ValueError:
        raise HTTPException(403, "Access to requested file path is forbidden")

    if not abs_path.is_file():
        raise HTTPException(404, f"Audio file for scene {index} does not exist on disk")

    audio_hash = target.get("audio_hash", "")
    headers = {
        "ETag": f'"{audio_hash}"',
        "Cache-Control": "private, no-cache",
    }
    return FileResponse(path=str(abs_path), media_type="audio/wav", headers=headers)


@router.get("/videos/{video_id}/subtitles")
def get_video_subtitles(video: Video = Depends(get_owned_video)):
    """Retrieve the synchronized subtitle data and word-level timestamps for a video."""
    step = next((s for s in video.steps if s.name in ("subtitles", "subtitle")), None)
    if not step or not step.output or not isinstance(step.output, dict):
        raise HTTPException(404, f"No subtitles found for video {video.id}")

    subtitles_data = step.output.get("subtitles")
    if not subtitles_data:
        raise HTTPException(404, f"Subtitles data is not ready for video {video.id}")

    return subtitles_data


