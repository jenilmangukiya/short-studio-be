import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base

JSONType = JSON().with_variant(JSONB(), "postgresql")


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )


class User(TimestampMixin, Base):
    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str | None] = mapped_column(String(120))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    channels: Mapped[list["Channel"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Channel(TimestampMixin, Base):
    __tablename__ = "channels"
    __table_args__ = (UniqueConstraint("user_id", "platform", "name"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    platform: Mapped[str] = mapped_column(String(20), default="youtube")  # youtube | instagram
    niche: Mapped[str | None] = mapped_column(String(120))
    handle: Mapped[str | None] = mapped_column(String(120))
    settings: Mapped[dict] = mapped_column(JSONType, default=dict)  # voice, style, music mood, ...
    user: Mapped[User] = relationship(back_populates="channels")
    videos: Mapped[list["Video"]] = relationship(back_populates="channel", cascade="all, delete-orphan")


class Video(TimestampMixin, Base):
    __tablename__ = "videos"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    channel_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("channels.id", ondelete="CASCADE"), index=True)
    topic: Mapped[str] = mapped_column(String(300))
    description: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(20), default="draft")  # draft|in_progress|completed|failed
    output_path: Mapped[str | None] = mapped_column(String(500))
    channel: Mapped[Channel] = relationship(back_populates="videos")
    steps: Mapped[list["VideoStep"]] = relationship(
        order_by="VideoStep.position", cascade="all, delete-orphan", lazy="selectin"
    )


class VideoStep(TimestampMixin, Base):
    __tablename__ = "video_steps"
    __table_args__ = (UniqueConstraint("video_id", "name"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(30))
    position: Mapped[int] = mapped_column(Integer)
    # pending | running | awaiting_approval | done | failed | stale
    status: Mapped[str] = mapped_column(String(20), default="pending")
    output: Mapped[dict | None] = mapped_column(JSONType)
    user_prompt: Mapped[str | None] = mapped_column(Text)  # revision instructions for this step
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revisions: Mapped[list["VideoStepRevision"]] = relationship(
        back_populates="step",
        order_by="VideoStepRevision.version",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    @property
    def revisions_count(self) -> int:
        return len(self.revisions) if self.revisions else 0


class VideoStepRevision(TimestampMixin, Base):
    __tablename__ = "video_step_revisions"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    step_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("video_steps.id", ondelete="CASCADE"), index=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    output: Mapped[dict | None] = mapped_column(JSONType)
    user_prompt: Mapped[str | None] = mapped_column(Text)
    change_summary: Mapped[str | None] = mapped_column(String(255))
    step: Mapped[VideoStep] = relationship(back_populates="revisions")


class PipelineJob(TimestampMixin, Base):
    __tablename__ = "pipeline_jobs"
    __table_args__ = (
        Index("ix_pipeline_jobs_claim", "status", "scheduled_at"),
        Index("ix_pipeline_jobs_video_step", "video_id", "step_name"),
        Index(
            "uq_pipeline_jobs_active_step",
            "video_id",
            "step_name",
            unique=True,
            postgresql_where=text("status IN ('queued', 'running')"),
            sqlite_where=text("status IN ('queued', 'running')"),
        ),
    )
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    video_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("videos.id", ondelete="CASCADE"), index=True)
    step_name: Mapped[str] = mapped_column(String(30))
    # queued | running | completed | failed | dead_letter | cancelled
    status: Mapped[str] = mapped_column(String(20), default="queued", index=True)
    run_all: Mapped[bool] = mapped_column(Boolean, default=False)

    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)

    scheduled_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now(), index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    last_heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    worker_id: Mapped[str | None] = mapped_column(String(100))
    lease_token: Mapped[uuid.UUID | None] = mapped_column(Uuid)

    error: Mapped[str | None] = mapped_column(Text)


