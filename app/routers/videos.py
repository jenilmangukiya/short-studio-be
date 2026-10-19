import os

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from .. import pipeline
from ..db import get_db
from ..deps import get_owned_channel, get_owned_video
from ..models import Channel, Video
from ..schemas import RerunBody, VideoCreate, VideoDetail, VideoSummary

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
def next_step(bg: BackgroundTasks, video: Video = Depends(get_owned_video), db: Session = Depends(get_db)):
    """Run exactly one step (the first one not done). Poll GET /videos/{id} for progress."""
    try:
        step = pipeline.claim_next_step(db, video.id)
    except pipeline.PipelineError as e:
        raise HTTPException(409, str(e))
    bg.add_task(pipeline.execute, video.id, step.name, False)
    db.refresh(video)
    return video


@router.post("/videos/{video_id}/run-all", response_model=VideoDetail, status_code=202)
def run_all(bg: BackgroundTasks, video: Video = Depends(get_owned_video), db: Session = Depends(get_db)):
    try:
        step = pipeline.claim_next_step(db, video.id)
    except pipeline.PipelineError as e:
        raise HTTPException(409, str(e))
    bg.add_task(pipeline.execute, video.id, step.name, True)
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


@router.get("/videos/{video_id}/download")
def download(video: Video = Depends(get_owned_video)):
    if not video.output_path or not os.path.exists(video.output_path):
        raise HTTPException(404, "Final output not available yet")
    return FileResponse(video.output_path)
