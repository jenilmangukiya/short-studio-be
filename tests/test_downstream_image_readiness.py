"""Regression tests for Downstream Image Readiness Guards (B-1 and B-2).

Verifies that rerunning or rolling back 'render', 'subtitles', or 'subtitle'
is rejected with HTTP 409 Conflict whenever the upstream 'images' step is:
- stale
- pending
- running
- failed
- missing

Also verifies that rejected requests do NOT modify step states, outputs, or jobs,
and that reruns and rollbacks succeed when images is 'done' (both 8-step and legacy 6-step pipelines).
"""

import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.deps import get_db, get_owned_video
from app.main import app
from app.models import Channel, PipelineJob, User, Video, VideoStep, VideoStepRevision
from app.pipeline import (
    LEGACY_STEPS,
    STEPS,
    PipelineError,
    init_steps,
    record_revision,
    rerun_step,
    rollback_step,
)


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _create_fixture_video(session: Session, steps: list[str] = STEPS) -> Video:
    user = User(email=f"u_{uuid.uuid4().hex[:8]}@example.com", password_hash="hash")
    session.add(user)
    session.flush()

    channel = Channel(user_id=user.id, name="TestChannel", settings={})
    session.add(channel)
    session.flush()

    video = Video(
        channel_id=channel.id,
        topic="Downstream Image Readiness",
        output_path="/storage/video_render.mp4",
        status="in_progress",
    )
    session.add(video)
    session.flush()

    init_steps(session, video, steps=steps)
    return video


# ---------------------------------------------------------------------------
# API-level tests: Rerun Render & Subtitles rejected when images is not ready
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_status", ["stale", "pending", "running", "failed"])
@pytest.mark.parametrize("target_step", ["render", "subtitles"])
def test_api_rerun_rejected_when_images_not_ready(db_session, bad_status, target_step):
    """POST /videos/{id}/steps/{step}/rerun returns 409 when images is stale, pending, running, or failed."""
    video = _create_fixture_video(db_session)
    for s in video.steps:
        s.status = "done"

    images_step = next(s for s in video.steps if s.name == "images")
    images_step.status = bad_status
    target = next(s for s in video.steps if s.name == target_step)
    target.output = {target_step: {"sample": "initial"}}
    db_session.commit()

    initial_target_status = target.status
    initial_output_path = video.output_path

    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_owned_video] = lambda: video
    client = TestClient(app)

    try:
        resp = client.post(
            f"/videos/{video.id}/steps/{target_step}/rerun",
            json={"prompt": "Please re-do this"},
        )
        assert resp.status_code == 409
        err_msg = resp.json().get("detail", "")
        assert f"Cannot rerun '{target_step}'" in err_msg
        assert f"images step is not ready (status: {bad_status})" in err_msg

        # State must NOT be modified
        db_session.refresh(target)
        db_session.refresh(video)
        assert target.status == initial_target_status
        assert video.output_path == initial_output_path
        # No revisions archived due to rejected rerun
        revs = target.revisions or []
        assert not any("Archived before rerun" in (r.change_summary or "") for r in revs)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("target_step", ["render", "subtitles"])
def test_api_rerun_rejected_when_images_missing(db_session, target_step):
    """POST /videos/{id}/steps/{step}/rerun returns 409 when images step is missing from pipeline."""
    # Pipeline without images step
    custom_steps = ["script", "voiceover", "render", "subtitles"]
    video = _create_fixture_video(db_session, steps=custom_steps)
    for s in video.steps:
        s.status = "done"
    db_session.commit()

    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_owned_video] = lambda: video
    client = TestClient(app)

    try:
        resp = client.post(
            f"/videos/{video.id}/steps/{target_step}/rerun",
            json={"prompt": "Redo"},
        )
        assert resp.status_code == 409
        err_msg = resp.json().get("detail", "")
        assert f"Cannot rerun '{target_step}': images step is missing" in err_msg
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# API-level tests: Rollback Render & Subtitles rejected when images is not ready
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_status", ["stale", "pending", "running", "failed"])
@pytest.mark.parametrize("target_step", ["render", "subtitles"])
def test_api_rollback_rejected_when_images_not_ready(db_session, bad_status, target_step):
    """POST /videos/{id}/steps/{step}/rollback/{v} returns 409 when images is stale, pending, running, or failed."""
    video = _create_fixture_video(db_session)
    for s in video.steps:
        s.status = "done"

    target = next(s for s in video.steps if s.name == target_step)
    target.output = {target_step: {"version": 2}}

    # Record revision 1 to roll back to
    rev1 = record_revision(
        db=db_session,
        step=target,
        output={target_step: {"version": 1}},
        change_summary="Initial V1",
    )
    db_session.commit()
    db_session.refresh(target)
    v1_version = rev1.version

    # Set images to bad status
    images_step = next(s for s in video.steps if s.name == "images")
    images_step.status = bad_status
    db_session.commit()

    initial_target_output = dict(target.output)
    initial_target_status = target.status

    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_owned_video] = lambda: video
    client = TestClient(app)

    try:
        resp = client.post(
            f"/videos/{video.id}/steps/{target_step}/rollback/{v1_version}",
        )
        assert resp.status_code == 409
        err_msg = resp.json().get("detail", "")
        assert f"Cannot roll back '{target_step}'" in err_msg
        assert f"images step is not ready (status: {bad_status})" in err_msg

        # Output and status must NOT be rolled back
        db_session.refresh(target)
        assert target.output == initial_target_output
        assert target.status == initial_target_status
        # No rollback revision created
        assert not any(f"Rolled back to version {v1_version}" in (r.change_summary or "") for r in target.revisions)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("target_step", ["render", "subtitles"])
def test_api_rollback_rejected_when_images_missing(db_session, target_step):
    """POST /videos/{id}/steps/{step}/rollback/{v} returns 409 when images step is missing from pipeline."""
    custom_steps = ["script", "voiceover", "render", "subtitles"]
    video = _create_fixture_video(db_session, steps=custom_steps)
    for s in video.steps:
        s.status = "done"

    target = next(s for s in video.steps if s.name == target_step)
    target.output = {target_step: {"version": 2}}
    rev = record_revision(
        db=db_session,
        step=target,
        output={target_step: {"version": 1}},
        change_summary="Initial V1",
    )
    db_session.commit()
    db_session.refresh(target)
    v1_version = rev.version

    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_owned_video] = lambda: video
    client = TestClient(app)

    try:
        resp = client.post(
            f"/videos/{video.id}/steps/{target_step}/rollback/{v1_version}",
        )
        assert resp.status_code == 409
        err_msg = resp.json().get("detail", "")
        assert f"Cannot roll back '{target_step}': images step is missing" in err_msg
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Rejected requests do not modify jobs, revisions, or step states
# ---------------------------------------------------------------------------


def test_rejected_rerun_does_not_cancel_or_enqueue_jobs(db_session):
    """When rerun is rejected due to image readiness, existing queued/running jobs are not cancelled."""
    video = _create_fixture_video(db_session)
    images_step = next(s for s in video.steps if s.name == "images")
    images_step.status = "stale"

    # Enqueue a mock job for another step (e.g. voiceover)
    job = PipelineJob(
        video_id=video.id,
        step_name="voiceover",
        status="queued",
    )
    db_session.add(job)
    db_session.commit()

    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_owned_video] = lambda: video
    client = TestClient(app)

    try:
        resp = client.post(
            f"/videos/{video.id}/steps/render/rerun",
            json={"prompt": "Rerun attempt"},
        )
        assert resp.status_code == 409

        # Job must NOT have been cancelled
        db_session.refresh(job)
        assert job.status == "queued"
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Allowed behavior when images are ready ('done')
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target_step", ["render", "subtitles"])
def test_rerun_allowed_when_images_ready(db_session, target_step):
    """When images step is 'done', rerun of render and subtitles succeeds (200)."""
    video = _create_fixture_video(db_session)
    for s in video.steps:
        s.status = "done"
    db_session.commit()

    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_owned_video] = lambda: video
    client = TestClient(app)

    try:
        resp = client.post(
            f"/videos/{video.id}/steps/{target_step}/rerun",
            json={"prompt": "Redo downstream"},
        )
        assert resp.status_code == 200
        data = resp.json()
        target_data = next(s for s in data["steps"] if s["name"] == target_step)
        assert target_data["status"] == "pending"
        assert target_data["user_prompt"] == "Redo downstream"
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("target_step", ["render", "subtitles"])
def test_rollback_allowed_when_images_ready(db_session, target_step):
    """When images step is 'done', rollback of render and subtitles succeeds (200)."""
    video = _create_fixture_video(db_session)
    for s in video.steps:
        s.status = "done"

    target = next(s for s in video.steps if s.name == target_step)
    target.output = {target_step: {"content": "version 2"}}
    rev = record_revision(
        db=db_session,
        step=target,
        output={target_step: {"content": "version 1"}},
        change_summary="Original V1",
    )
    db_session.commit()
    db_session.refresh(target)
    v1_version = rev.version

    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_owned_video] = lambda: video
    client = TestClient(app)

    try:
        resp = client.post(
            f"/videos/{video.id}/steps/{target_step}/rollback/{v1_version}",
        )
        assert resp.status_code == 200
        data = resp.json()
        target_data = next(s for s in data["steps"] if s["name"] == target_step)
        assert target_data["output"][target_step]["content"] == "version 1"
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Legacy six-step pipeline compatibility
# ---------------------------------------------------------------------------


def test_legacy_six_step_pipeline_rerun_and_rollback(db_session):
    """Legacy 6-step videos respect image readiness guards and allow rerun/rollback when images is done."""
    video = _create_fixture_video(db_session, steps=LEGACY_STEPS)
    for s in video.steps:
        s.status = "done"

    render_step = next(s for s in video.steps if s.name == "render")
    render_step.output = {"render": {"video": "v2.mp4"}}
    rev = record_revision(
        db=db_session,
        step=render_step,
        output={"render": {"video": "v1.mp4"}},
        change_summary="Legacy V1",
    )
    db_session.commit()
    db_session.refresh(render_step)
    v1_version = rev.version

    # 1. Reject when images is stale in legacy pipeline
    images_step = next(s for s in video.steps if s.name == "images")
    images_step.status = "stale"
    db_session.commit()

    with pytest.raises(PipelineError) as exc_rerun:
        rerun_step(db_session, video, "render", prompt="Redo")
    assert "Cannot rerun 'render': images step is not ready (status: stale)" in str(exc_rerun.value)

    with pytest.raises(PipelineError) as exc_rb:
        rollback_step(db_session, video, "render", target_version=v1_version)
    assert "Cannot roll back 'render': images step is not ready (status: stale)" in str(exc_rb.value)

    # 2. Allow when images is marked done
    images_step.status = "done"
    db_session.commit()

    # Rollback succeeds
    rb_res = rollback_step(db_session, video, "render", target_version=v1_version)
    assert rb_res.output == {"render": {"video": "v1.mp4"}}

    # Rerun succeeds
    rerun_step(db_session, video, "render", prompt="Redo render")
    assert render_step.status == "pending"
    assert render_step.user_prompt == "Redo render"


def test_singular_subtitle_step_name_supported(db_session):
    """Direct calls targeting singular 'subtitle' step name are also guarded."""
    video = _create_fixture_video(db_session)
    images_step = next(s for s in video.steps if s.name == "images")
    images_step.status = "failed"
    db_session.commit()

    with pytest.raises(PipelineError) as exc:
        rerun_step(db_session, video, "subtitle", prompt=None)
    assert "Cannot rerun 'subtitle': images step is not ready (status: failed)" in str(exc.value)

    with pytest.raises(PipelineError) as exc_rb:
        rollback_step(db_session, video, "subtitle", target_version=1)
    assert "Cannot roll back 'subtitle': images step is not ready (status: failed)" in str(exc_rb.value)
