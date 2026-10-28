import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field


class ORM(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str | None = None


class UserOut(ORM):
    id: uuid.UUID
    email: EmailStr
    full_name: str | None


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class ChannelCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    platform: Literal["youtube", "instagram"] = "youtube"
    niche: str | None = None
    handle: str | None = None
    settings: dict = {}


class ChannelUpdate(BaseModel):
    name: str | None = None
    niche: str | None = None
    handle: str | None = None
    settings: dict | None = None


class ChannelOut(ORM):
    id: uuid.UUID
    name: str
    platform: str
    niche: str | None
    handle: str | None
    settings: dict
    created_at: datetime


class VideoCreate(BaseModel):
    topic: str = Field(min_length=3, max_length=300)
    description: str | None = None


class StepOut(ORM):
    name: str
    position: int
    status: str
    output: dict | None
    user_prompt: str | None
    error: str | None
    started_at: datetime | None
    finished_at: datetime | None
    revisions_count: int = 0


class StepRevisionOut(ORM):
    id: uuid.UUID
    step_id: uuid.UUID
    version: int
    output: dict | None
    user_prompt: str | None
    change_summary: str | None
    created_at: datetime


class StepApproveBody(BaseModel):
    auto_advance: bool = False


class StepContentUpdate(BaseModel):
    output: dict
    change_summary: str | None = None


class JobOut(ORM):
    id: uuid.UUID
    video_id: uuid.UUID
    step_name: str
    status: str
    run_all: bool
    attempts: int
    max_attempts: int
    scheduled_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    worker_id: str | None
    error: str | None



class VideoSummary(ORM):
    id: uuid.UUID
    channel_id: uuid.UUID
    topic: str
    description: str | None
    status: str
    created_at: datetime


class VideoDetail(VideoSummary):
    steps: list[StepOut] = []


class RerunBody(BaseModel):
    prompt: str | None = None  # "make the hook more dramatic", etc.

