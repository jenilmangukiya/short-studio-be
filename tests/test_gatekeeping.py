"""Tests for Phase 4: Production Gatekeeping, Inline Revision, and Version History."""
import uuid
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db import Base
from app.models import Channel, User, Video, VideoStep, VideoStepRevision
from app.pipeline import (
    STEPS,
    PipelineError,
    approve_step,
    claim_next_step,
    execute,
    init_steps,
    rerun_step,
    rollback_step,
    update_step_content,
)


from sqlalchemy.pool import StaticPool

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



def _create_test_video(session: Session, require_approval: bool = True) -> Video:
    user = User(email=f"user_{uuid.uuid4().hex[:8]}@example.com", password_hash="hash")
    session.add(user)
    session.flush()

    channel = Channel(
        user_id=user.id,
        name=f"Channel {uuid.uuid4().hex[:6]}",
        settings={"require_approval": require_approval},
    )
    session.add(channel)
    session.flush()

    video = Video(channel_id=channel.id, topic="Why Bridges Don't Collapse")
    session.add(video)
    session.flush()

    init_steps(session, video)
    return video


def test_gatekeeping_pauses_at_awaiting_approval(db_session, monkeypatch):
    """When gatekeeping is active, completed step transitions to awaiting_approval."""
    video = _create_test_video(db_session, require_approval=True)

    # Mock handler to return valid mock script output
    def mock_script(v, prev, s):
        return {"script": {"title": "Bridges", "narration": "Arch bridges distribute load effectively."}}

    from contextlib import contextmanager

    @contextmanager
    def _mock_session_local():
        yield db_session

    monkeypatch.setitem(__import__("app.pipeline", fromlist=["HANDLERS"]).HANDLERS, "script", mock_script)
    monkeypatch.setattr("app.pipeline.SessionLocal", _mock_session_local)

    # Claim and execute step 0 (script)

    step = claim_next_step(db_session, video.id)
    assert step.name == "script"
    assert step.status == "running"

    execute(video.id, "script", run_all=False)
    db_session.refresh(video)

    assert video.steps[0].status == "awaiting_approval"
    assert video.steps[0].output is not None
    assert video.steps[0].revisions_count == 1
    assert video.steps[0].revisions[0].version == 1
    assert video.status == "in_progress"

    # Attempting to claim the next step without approving should raise PipelineError
    with pytest.raises(PipelineError, match="awaiting approval"):
        claim_next_step(db_session, video.id)


def test_approve_step_transitions_to_done_and_allows_next(db_session):
    """Approving step marks it done and enables claiming the next step."""
    video = _create_test_video(db_session, require_approval=True)

    video.steps[0].status = "awaiting_approval"
    video.steps[0].output = {"script": {"title": "Test", "narration": "Narration"}}
    db_session.commit()

    approved_step = approve_step(db_session, video, "script")
    assert approved_step.status == "done"
    assert video.steps[0].status == "done"

    # Now claim_next_step should successfully claim step 1 (storyboard)
    next_step = claim_next_step(db_session, video.id)
    assert next_step.name == "storyboard"
    assert next_step.status == "running"


def test_inline_edit_script_creates_revision_and_invalidates_downstream(db_session):
    """Inline editing updates script narration, records revision history, and sets downstream steps stale."""
    video = _create_test_video(db_session, require_approval=True)

    # Mark script done and storyboard done
    video.steps[0].status = "done"
    video.steps[0].output = {
        "script": {
            "title": "Old Title",
            "hook": "Old Hook",
            "narration": "Old Narration Text",
        }
    }
    video.steps[1].status = "done"
    video.steps[1].output = {"storyboard": {"scenes": []}}
    video.output_path = "/storage/test/final.mp4"
    db_session.commit()

    # Perform inline edit
    updated_payload = {
        "script": {
            "narration": "New Edited Narration Text with more suspense.",
            "hook": "New Hook",
        }
    }
    updated_step = update_step_content(
        db=db_session,
        video=video,
        step_name="script",
        updated_payload=updated_payload,
        change_summary="Edited narration text",
    )

    assert updated_step.output["script"]["narration"] == "New Edited Narration Text with more suspense."
    assert updated_step.output["script"]["hook"] == "New Hook"
    assert updated_step.output["script"]["title"] == "Old Title"  # Preserved!

    # Downstream step 1 (storyboard) must now be stale
    db_session.refresh(video)
    assert video.steps[1].status == "stale"
    assert video.output_path is None  # Video output invalidated

    # Revisions history must contain snapshots
    assert updated_step.revisions_count >= 2
    assert updated_step.revisions[-1].change_summary == "Edited narration text"
    assert updated_step.revisions[-1].output["script"]["narration"] == "New Edited Narration Text with more suspense."


def test_inline_edit_storyboard_scenes(db_session):
    """Inline editing updates scene image prompts and sets downstream images and audio stale."""
    video = _create_test_video(db_session, require_approval=True)

    video.steps[0].status = "done"
    video.steps[1].status = "awaiting_approval"
    video.steps[1].output = {
        "storyboard": {
            "scenes": [
                {"index": 1, "narration": "Scene 1", "image_prompt": "Old Prompt 1", "shot_type": "wide"},
                {"index": 2, "narration": "Scene 2", "image_prompt": "Old Prompt 2", "shot_type": "close-up"},
            ]
        }
    }
    video.steps[2].status = "done"  # images
    video.steps[3].status = "done"  # voiceover
    db_session.commit()

    # User modifies prompt for scene 1
    new_scenes = [
        {"index": 1, "narration": "Scene 1", "image_prompt": "Dramatic new prompt for scene 1", "shot_type": "macro"},
        {"index": 2, "narration": "Scene 2", "image_prompt": "Old Prompt 2", "shot_type": "close-up"},
    ]
    update_step_content(
        db=db_session,
        video=video,
        step_name="storyboard",
        updated_payload={"storyboard": {"scenes": new_scenes}},
        change_summary="Updated scene 1 prompt to macro angle",
    )

    db_session.refresh(video)
    assert video.steps[1].output["storyboard"]["scenes"][0]["image_prompt"] == "Dramatic new prompt for scene 1"
    assert video.steps[1].output["storyboard"]["scenes"][0]["shot_type"] == "macro"

    # Images and voiceover should now be stale
    assert video.steps[2].status == "stale"
    assert video.steps[3].status == "stale"


def test_version_rollback(db_session):
    """Rollback restores an earlier revision and creates an explicit rollback revision entry."""
    video = _create_test_video(db_session, require_approval=True)

    video.steps[0].status = "done"
    video.steps[0].output = {"script": {"narration": "Version 1 text"}}
    db_session.commit()

    # Edit to Version 2
    update_step_content(
        db=db_session,
        video=video,
        step_name="script",
        updated_payload={"script": {"narration": "Version 2 edited text"}},
        change_summary="Edit to v2",
    )

    # Edit to Version 3
    update_step_content(
        db=db_session,
        video=video,
        step_name="script",
        updated_payload={"script": {"narration": "Version 3 mistake"}},
        change_summary="Edit to v3",
    )

    db_session.refresh(video)
    step = video.steps[0]
    assert step.output["script"]["narration"] == "Version 3 mistake"

    # Find the revision with version 1
    v1_rev = next(r for r in step.revisions if "Version 1" in str(r.output))
    target_v = v1_rev.version

    # Rollback to v1
    rollback_step(db_session, video, "script", target_version=target_v)

    db_session.refresh(video)
    rolled_back_step = video.steps[0]
    assert rolled_back_step.output["script"]["narration"] == "Version 1 text"
    assert rolled_back_step.revisions[-1].change_summary == f"Rolled back to version {target_v}"


def test_rerun_preserves_revision_history(db_session):
    """Rerunning a step snapshots the existing output before resetting to pending."""
    video = _create_test_video(db_session, require_approval=True)

    video.steps[0].status = "done"
    video.steps[0].output = {"script": {"narration": "Initial version before rerun"}}
    db_session.commit()

    rerun_step(db_session, video, "script", prompt="Make it more punchy")

    db_session.refresh(video)
    step = video.steps[0]
    assert step.status == "pending"
    assert step.user_prompt == "Make it more punchy"
    assert step.revisions_count >= 1
    assert "Archived before rerun" in step.revisions[-1].change_summary


def test_api_approve_endpoint(db_session):
    """Test POST /videos/{id}/steps/{step_name}/approve endpoint."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.deps import get_db, get_owned_video

    video = _create_test_video(db_session, require_approval=True)
    video.steps[0].status = "awaiting_approval"
    video.steps[0].output = {"script": {"title": "Test Title", "narration": "Test Narration"}}
    db_session.commit()

    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_owned_video] = lambda: video

    client = TestClient(app)
    try:
        resp = client.post(f"/videos/{video.id}/steps/script/approve", json={"auto_advance": False})
        assert resp.status_code == 200
        data = resp.json()
        step = next(s for s in data["steps"] if s["name"] == "script")
        assert step["status"] == "done"
    finally:
        app.dependency_overrides.clear()


def test_api_patch_endpoint(db_session):
    """Test PATCH /videos/{id}/steps/{step_name} inline editing endpoint."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.deps import get_db, get_owned_video

    video = _create_test_video(db_session, require_approval=True)
    video.steps[0].status = "done"
    video.steps[0].output = {"script": {"title": "Title", "narration": "Original Narration"}}
    video.steps[1].status = "done"
    video.steps[1].output = {"storyboard": {"scenes": []}}
    db_session.commit()

    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_owned_video] = lambda: video

    client = TestClient(app)
    try:
        patch_payload = {
            "output": {"script": {"narration": "Directly Edited Narration via API"}},
            "change_summary": "API inline patch test",
        }
        resp = client.patch(f"/videos/{video.id}/steps/script", json=patch_payload)
        assert resp.status_code == 200
        data = resp.json()
        script_step = next(s for s in data["steps"] if s["name"] == "script")
        assert script_step["output"]["script"]["narration"] == "Directly Edited Narration via API"
        assert script_step["output"]["script"]["title"] == "Title"

        # Downstream storyboard should now be stale
        sb_step = next(s for s in data["steps"] if s["name"] == "storyboard")
        assert sb_step["status"] == "stale"
    finally:
        app.dependency_overrides.clear()


def test_api_history_and_rollback_endpoints(db_session):
    """Test GET history and POST rollback endpoints."""
    from fastapi.testclient import TestClient
    from app.main import app
    from app.deps import get_db, get_owned_video

    video = _create_test_video(db_session, require_approval=True)
    video.steps[0].status = "done"
    video.steps[0].output = {"script": {"narration": "Original V1"}}
    db_session.commit()

    # Create V2 via update
    update_step_content(db_session, video, "script", {"script": {"narration": "Modified V2"}}, "V2 edit")

    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[get_owned_video] = lambda: video

    client = TestClient(app)
    try:
        # 1. Check history
        resp_hist = client.get(f"/videos/{video.id}/steps/script/history")
        assert resp_hist.status_code == 200
        history = resp_hist.json()
        assert len(history) >= 2
        v1_entry = next(h for h in history if "Original V1" in str(h["output"]))

        # 2. Rollback to v1
        resp_rollback = client.post(f"/videos/{video.id}/steps/script/rollback/{v1_entry['version']}")
        assert resp_rollback.status_code == 200
        data = resp_rollback.json()
        script_step = next(s for s in data["steps"] if s["name"] == "script")
        assert script_step["output"]["script"]["narration"] == "Original V1"
    finally:
        app.dependency_overrides.clear()

