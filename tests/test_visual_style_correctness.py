"""Regression and correctness tests for Visual Style System Phase 1 audit fixes:
1. Image step must fail closed for invalid visual styles (malformed snapshot, unknown ID, version mismatch)
   while preserving legacy Clean 3D when style fields are NULL, and not calling the image provider.
2. Same-style PATCH is an idempotent no-op (preserves outputs, output_path, completed status, running jobs, revisions),
   while legacy NULL style fields transitioning to legacy_default properly freezes snapshot.
3. Direct pipeline exceptions do not overwrite newer stale status or newer execution ownership,
   while genuine failures with no concurrent modification are accurately marked failed.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.db import Base, get_db
from app.deps import current_user, get_owned_channel, get_owned_video
from app.images.mock import MockImageProvider
from app.main import app
from app.models import Channel, PipelineJob, User, Video, VideoStep, VideoStepRevision
from app.pipeline import execute, init_steps
from app.steps.images import generate_images_step
from app.styles import (
    LEGACY_DEFAULT_PRESET,
    STICKMAN_TACTICAL_EXPLAINER_PRESET,
    VisualStyleResolutionError,
)


@pytest.fixture
def db_session():
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(test_engine)
    with Session(test_engine) as session:
        yield session


@pytest.fixture
def test_user(db_session: Session) -> User:
    user = User(
        email=f"audit_user_{uuid.uuid4().hex[:8]}@example.com",
        password_hash="fakehash",
        full_name="Audit Test User",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_channel(db_session: Session, test_user: User) -> Channel:
    channel = Channel(
        user_id=test_user.id,
        platform="youtube",
        name="Audit Channel",
        niche="Science & Technology",
        default_visual_style_id="legacy_default",
    )
    db_session.add(channel)
    db_session.commit()
    db_session.refresh(channel)
    return channel


@pytest.fixture
def client(db_session: Session, test_user: User) -> TestClient:
    def _override_db():
        yield db_session

    def _override_user():
        return test_user

    app.dependency_overrides[get_db] = _override_db
    app.dependency_overrides[current_user] = _override_user
    app.dependency_overrides[get_owned_channel] = lambda: db_session.query(Channel).first()
    app.dependency_overrides[get_owned_video] = lambda: db_session.query(Video).first()

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


SAMPLE_STORYBOARD_DATA = {
    "storyboard": {
        "scenes": [
            {
                "index": 1,
                "narration": "Have you ever wondered how stars are born?",
                "visual_prompt": "A swirling interstellar nebula in deep space",
                "final_prompt": "A swirling interstellar nebula in deep space",
                "shot_type": "wide",
                "visual_purpose": "establishing",
            }
        ]
    }
}


# =========================================================================
# Fix 1: Image Step Must Fail Closed for Invalid Visual Styles
# =========================================================================

def test_images_step_fails_closed_on_malformed_snapshot(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    provider = MockImageProvider()

    video = Video(
        channel_id=test_channel.id,
        topic="Malformed Snapshot Test",
        visual_style_id="stickman_tactical_explainer",
        visual_style_snapshot={"invalid": "malformed_snapshot_missing_fields"},
        visual_style_version=1,
    )
    db_session.add(video)
    db_session.commit()
    init_steps(db_session, video)

    img_step = next(s for s in video.steps if s.name == "images")
    img_step.status = "running"
    db_session.commit()

    with pytest.raises(VisualStyleResolutionError, match="Malformed or invalid visual style snapshot"):
        generate_images_step(
            video=video,
            prev=SAMPLE_STORYBOARD_DATA,
            step=img_step,
            image_provider=provider,
        )

    # Must fail closed BEFORE invoking image generation API
    assert len(provider.call_history) == 0


def test_images_step_fails_closed_on_unknown_style_id(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    provider = MockImageProvider()

    video = Video(
        channel_id=test_channel.id,
        topic="Unknown Style ID Test",
        visual_style_id="unknown_fantasy_preset_xyz",
        visual_style_snapshot=None,
        visual_style_version=None,
    )
    db_session.add(video)
    db_session.commit()
    init_steps(db_session, video)

    img_step = next(s for s in video.steps if s.name == "images")
    img_step.status = "running"
    db_session.commit()

    with pytest.raises(VisualStyleResolutionError, match="Unknown visual style ID 'unknown_fantasy_preset_xyz'"):
        generate_images_step(
            video=video,
            prev=SAMPLE_STORYBOARD_DATA,
            step=img_step,
            image_provider=provider,
        )

    # Must fail closed BEFORE invoking image generation API
    assert len(provider.call_history) == 0


def test_images_step_fails_closed_on_incompatible_version(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    provider = MockImageProvider()

    video = Video(
        channel_id=test_channel.id,
        topic="Incompatible Version Test",
        visual_style_id="stickman_tactical_explainer",
        visual_style_snapshot=None,
        visual_style_version=999,
    )
    db_session.add(video)
    db_session.commit()
    init_steps(db_session, video)

    img_step = next(s for s in video.steps if s.name == "images")
    img_step.status = "running"
    db_session.commit()

    with pytest.raises(VisualStyleResolutionError, match="Incompatible visual style version 999"):
        generate_images_step(
            video=video,
            prev=SAMPLE_STORYBOARD_DATA,
            step=img_step,
            image_provider=provider,
        )

    # Must fail closed BEFORE invoking image generation API
    assert len(provider.call_history) == 0


def test_images_step_preserves_legacy_clean3d_when_style_fields_null(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    provider = MockImageProvider()

    video = Video(
        channel_id=test_channel.id,
        topic="Legacy Clean 3D Test",
        visual_style_id=None,
        visual_style_snapshot=None,
        visual_style_version=None,
    )
    db_session.add(video)
    db_session.commit()
    init_steps(db_session, video)

    img_step = next(s for s in video.steps if s.name == "images")
    img_step.status = "running"
    db_session.commit()

    # Must execute cleanly without raising VisualStyleResolutionError
    result = generate_images_step(
        video=video,
        prev=SAMPLE_STORYBOARD_DATA,
        step=img_step,
        image_provider=provider,
    )

    assert result is not None
    assert "images" in result
    assert len(result["images"]["items"]) == 1
    assert result["images"]["items"][0]["status"] == "ok"
    assert len(provider.call_history) == 1


# =========================================================================
# Fix 2: Same-Style PATCH Idempotency
# =========================================================================

def test_patch_video_visual_style_same_style_is_idempotent_noop(client: TestClient, db_session: Session, test_channel: Channel):
    resp = client.post(
        f"/channels/{test_channel.id}/videos",
        json={"topic": "Quantum Computing", "visual_style_id": "stickman_tactical_explainer"},
    )
    vid_id = resp.json()["id"]

    video = db_session.get(Video, uuid.UUID(vid_id))
    init_steps(db_session, video)

    for step in video.steps:
        step.status = "done"
        step.output = {step.name: {"data": f"mock_{step.name}_output"}}
    video.status = "completed"
    video.output_path = f"storage/{video.id}/final_video_captioned.mp4"

    job = PipelineJob(
        video_id=video.id,
        step_name="images",
        status="running",
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(video)

    revs_before = len(db_session.query(VideoStepRevision).all())

    patch_resp = client.patch(
        f"/videos/{vid_id}/visual-style",
        json={"visual_style_id": "stickman_tactical_explainer"},
    )
    assert patch_resp.status_code == 200
    data = patch_resp.json()
    assert data["visual_style_id"] == "stickman_tactical_explainer"

    db_session.refresh(video)
    db_session.refresh(job)

    assert video.status == "completed"
    assert video.output_path == f"storage/{video.id}/final_video_captioned.mp4"

    step_map = {s.name: s for s in video.steps}
    assert step_map["images"].status == "done"
    assert step_map["images"].output == {"images": {"data": "mock_images_output"}}
    assert step_map["render"].status == "done"
    assert step_map["render"].output == {"render": {"data": "mock_render_output"}}
    assert step_map["subtitles"].status == "done"
    assert step_map["subtitles"].output == {"subtitles": {"data": "mock_subtitles_output"}}

    assert job.status == "running"

    revs_after = len(db_session.query(VideoStepRevision).all())
    assert revs_after == revs_before


def test_patch_video_visual_style_legacy_null_to_legacy_default_freezes_snapshot(client: TestClient, db_session: Session, test_channel: Channel):
    video = Video(
        channel_id=test_channel.id,
        topic="Pre-Style Era Video",
        visual_style_id=None,
        visual_style_version=None,
        visual_style_snapshot=None,
    )
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)
    init_steps(db_session, video)

    patch_resp = client.patch(
        f"/videos/{video.id}/visual-style",
        json={"visual_style_id": "legacy_default"},
    )
    assert patch_resp.status_code == 200
    data = patch_resp.json()

    assert data["visual_style_id"] == "legacy_default"
    assert data["visual_style_version"] == 1
    assert data["visual_style_snapshot"] is not None
    assert data["visual_style_snapshot"]["id"] == "legacy_default"

    db_session.refresh(video)
    assert video.visual_style_id == "legacy_default"
    assert video.visual_style_version == 1
    assert video.visual_style_snapshot == LEGACY_DEFAULT_PRESET.model_dump()


# =========================================================================
# Fix 3: Direct Pipeline Exception Safety & Ownership
# =========================================================================

def test_direct_execution_exception_does_not_overwrite_stale_step(db_session: Session, test_channel: Channel, monkeypatch):
    """An obsolete direct execution encounters an exception after concurrent operation marks it stale."""
    video = Video(
        channel_id=test_channel.id,
        topic="Direct Pipeline Stale Race Test",
        visual_style_id="legacy_default",
        visual_style_snapshot=LEGACY_DEFAULT_PRESET.model_dump(),
        visual_style_version=1,
    )
    db_session.add(video)
    db_session.commit()
    init_steps(db_session, video)

    step = next(s for s in video.steps if s.name == "images")
    step.status = "running"
    step.started_at = datetime.now(timezone.utc)
    db_session.commit()

    def _simulated_concurrent_fail_handler(vid, prev, stp):
        # Concurrently mark step stale and switch style
        with Session(db_session.bind) as concurrent_db:
            c_step = concurrent_db.get(VideoStep, step.id)
            c_video = concurrent_db.get(Video, video.id)
            c_step.status = "stale"
            c_step.output = None
            c_video.status = "in_progress"
            c_video.visual_style_id = "stickman_tactical_explainer"
            c_video.visual_style_snapshot = STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump()
            concurrent_db.commit()
        # Now raise an exception in the obsolete in-flight handler
        raise RuntimeError("Image generation API rate limit on obsolete run")

    monkeypatch.setattr("app.pipeline.HANDLERS", {"images": _simulated_concurrent_fail_handler})
    monkeypatch.setattr("app.pipeline.SessionLocal", lambda: Session(db_session.bind))

    # Run direct execution
    execute(video.id, "images", run_all=False)

    # Verify the step was NOT overwritten with "failed"
    db_session.refresh(step)
    db_session.refresh(video)

    assert step.status == "stale"
    assert step.error is None
    assert video.status == "in_progress"


def test_direct_execution_exception_does_not_overwrite_newer_execution(db_session: Session, test_channel: Channel, monkeypatch):
    """An older direct execution raises after a newer execution of the same step took ownership."""
    video = Video(
        channel_id=test_channel.id,
        topic="Direct Pipeline Ownership Race Test",
        visual_style_id="stickman_tactical_explainer",
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
        visual_style_version=1,
    )
    db_session.add(video)
    db_session.commit()
    init_steps(db_session, video)

    step = next(s for s in video.steps if s.name == "images")
    initial_started_at = datetime.now(timezone.utc)
    step.status = "running"
    step.started_at = initial_started_at
    db_session.commit()

    def _simulated_newer_execution_handler(vid, prev, stp):
        # Concurrently, Execution 2 starts and claims the step with a newer started_at
        with Session(db_session.bind) as concurrent_db:
            c_step = concurrent_db.get(VideoStep, step.id)
            c_step.status = "running"
            c_step.started_at = initial_started_at + timedelta(seconds=2)
            concurrent_db.commit()
        # Execution 1 now fails
        raise RuntimeError("Execution 1 failed after superseded")

    monkeypatch.setattr("app.pipeline.HANDLERS", {"images": _simulated_newer_execution_handler})
    monkeypatch.setattr("app.pipeline.SessionLocal", lambda: Session(db_session.bind))

    execute(video.id, "images", run_all=False)

    db_session.refresh(step)
    db_session.refresh(video)

    # Step must remain "running" with Execution 2's started_at, NOT marked "failed"
    assert step.status == "running"
    step_started_naive = step.started_at.replace(tzinfo=None) if step.started_at.tzinfo else step.started_at
    initial_started_naive = initial_started_at.replace(tzinfo=None) if initial_started_at.tzinfo else initial_started_at
    assert step_started_naive > initial_started_naive
    assert step.error is None
    assert video.status == "in_progress"


def test_direct_execution_exception_marks_failed_when_not_stale(db_session: Session, test_channel: Channel, monkeypatch):
    """A genuine exception with no concurrent modification marks the step and video failed."""
    video = Video(
        channel_id=test_channel.id,
        topic="Direct Pipeline Genuine Failure Test",
        visual_style_id="stickman_tactical_explainer",
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
        visual_style_version=1,
    )
    db_session.add(video)
    db_session.commit()
    init_steps(db_session, video)

    step = next(s for s in video.steps if s.name == "images")
    step.status = "running"
    step.started_at = datetime.now(timezone.utc)
    db_session.commit()

    def _genuine_failing_handler(vid, prev, stp):
        raise RuntimeError("Genuine network error: 503 Service Unavailable")

    monkeypatch.setattr("app.pipeline.HANDLERS", {"images": _genuine_failing_handler})
    monkeypatch.setattr("app.pipeline.SessionLocal", lambda: Session(db_session.bind))

    execute(video.id, "images", run_all=False)

    db_session.refresh(step)
    db_session.refresh(video)

    # Legitimate failure must be recorded
    assert step.status == "failed"
    assert "Genuine network error: 503 Service Unavailable" in (step.error or "")
    assert video.status == "failed"
