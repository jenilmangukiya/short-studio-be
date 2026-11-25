"""Tests for Phase 2: Visual Style System persistence, database migration, and API endpoints."""

from __future__ import annotations

import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db import Base, get_db
from app.deps import current_user, get_owned_channel, get_owned_video
from app.main import app
from app.models import Channel, PipelineJob, User, Video, VideoStepRevision
from app.pipeline import init_steps
from app.styles import (
    DEFAULT_STYLE_ID,
    LEGACY_DEFAULT_PRESET,
    STICKMAN_TACTICAL_EXPLAINER_PRESET,
    STYLIZED_2D_NARRATIVE_PRESET,
    list_styles,
)


@pytest.fixture
def db_session():
    """Create an isolated in-memory SQLite database for test runs."""
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
        email=f"user_{uuid.uuid4().hex[:8]}@example.com",
        password_hash="fakehash",
        full_name="Style Test User",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def other_user(db_session: Session) -> User:
    user = User(
        email=f"other_{uuid.uuid4().hex[:8]}@example.com",
        password_hash="fakehash",
        full_name="Other User",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_channel(db_session: Session, test_user: User) -> Channel:
    channel = Channel(
        user_id=test_user.id,
        name=f"Channel {uuid.uuid4().hex[:6]}",
        platform="youtube",
        default_visual_style_id="stickman_tactical_explainer",
    )
    db_session.add(channel)
    db_session.commit()
    db_session.refresh(channel)
    return channel


@pytest.fixture
def client(db_session: Session, test_user: User) -> TestClient:
    def override_get_db():
        yield db_session

    def override_current_user():
        return test_user

    def override_get_owned_channel(channel_id: uuid.UUID):
        ch = db_session.get(Channel, channel_id)
        if not ch or ch.user_id != test_user.id:
            from fastapi import HTTPException
            raise HTTPException(404, "Channel not found")
        return ch

    def override_get_owned_video(video_id: uuid.UUID):
        vid = db_session.get(Video, video_id)
        if not vid or vid.channel.user_id != test_user.id:
            from fastapi import HTTPException
            raise HTTPException(404, "Video not found")
        return vid

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[current_user] = override_current_user
    app.dependency_overrides[get_owned_channel] = override_get_owned_channel
    app.dependency_overrides[get_owned_video] = override_get_owned_video

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


# ==============================================================================
# 1. GET /visual-styles & GET /visual-styles/{style_id}
# ==============================================================================

def test_list_visual_styles(client: TestClient):
    """GET /visual-styles returns registry summaries and specifies default style."""
    resp = client.get("/visual-styles")
    assert resp.status_code == 200
    data = resp.json()

    assert data["default_style_id"] == DEFAULT_STYLE_ID
    assert len(data["styles"]) == 3

    style_ids = [s["id"] for s in data["styles"]]
    assert style_ids == [
        "legacy_default",
        "stickman_tactical_explainer",
        "stylized_2d_narrative",
    ]

    legacy = next(s for s in data["styles"] if s["id"] == "legacy_default")
    assert legacy["is_default"] is True
    assert legacy["category"] == "3d_render"
    assert legacy["version"] == 1
    assert legacy["preview_image_path"] == "previews/legacy_default.jpg"

    stickman = next(s for s in data["styles"] if s["id"] == "stickman_tactical_explainer")
    assert stickman["is_default"] is False
    assert stickman["category"] == "illustrated_2d"
    assert stickman["version"] == 1


def test_get_visual_style_detail(client: TestClient):
    """GET /visual-styles/{style_id} returns full preset schema including rules and art direction."""
    resp = client.get("/visual-styles/stickman_tactical_explainer")
    assert resp.status_code == 200
    data = resp.json()

    assert data["id"] == "stickman_tactical_explainer"
    assert data["display_name"] == STICKMAN_TACTICAL_EXPLAINER_PRESET.display_name
    assert "rules" in data
    assert "linework_and_outlines" in data["rules"]
    assert "permanent_prompt_anchors" in data
    assert "clean 2D animated stickman illustration style" in data["permanent_prompt_anchors"]
    assert "art_direction" in data
    assert data["art_direction"]["style_family"] == "illustrated"
    assert "capabilities" in data
    assert data["capabilities"]["supports_reference_images"] is True


def test_get_visual_style_detail_not_found(client: TestClient):
    """GET /visual-styles/{style_id} returns 404 for unknown preset."""
    resp = client.get("/visual-styles/nonexistent_preset_xyz")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()


# ==============================================================================
# 2. Channel Default Visual Style CRUD & Validation
# ==============================================================================

def test_channel_create_with_default_style(client: TestClient):
    """Channel can be created with an optional default_visual_style_id."""
    payload = {
        "name": "Stickman Explainer Hub",
        "platform": "youtube",
        "default_visual_style_id": "stickman_tactical_explainer",
    }
    resp = client.post("/channels", json=payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["name"] == "Stickman Explainer Hub"
    assert data["default_visual_style_id"] == "stickman_tactical_explainer"


def test_channel_create_with_invalid_style_fails(client: TestClient):
    """Channel creation fails validation when default_visual_style_id does not exist."""
    payload = {
        "name": "Invalid Style Hub",
        "platform": "youtube",
        "default_visual_style_id": "totally_unknown_style",
    }
    resp = client.post("/channels", json=payload)
    assert resp.status_code == 422


def test_channel_create_without_style_defaults_to_none(client: TestClient):
    """Channel created without default_visual_style_id stores None."""
    payload = {
        "name": "Default Channel",
        "platform": "youtube",
    }
    resp = client.post("/channels", json=payload)
    assert resp.status_code == 201
    assert resp.json()["default_visual_style_id"] is None


def test_channel_update_default_style(client: TestClient, test_channel: Channel):
    """Updating channel default_visual_style_id persists change and allows clearing."""
    # Update to stylized_2d_narrative
    resp = client.patch(
        f"/channels/{test_channel.id}",
        json={"default_visual_style_id": "stylized_2d_narrative"},
    )
    assert resp.status_code == 200
    assert resp.json()["default_visual_style_id"] == "stylized_2d_narrative"

    # Clear back to None
    resp = client.patch(
        f"/channels/{test_channel.id}",
        json={"default_visual_style_id": None},
    )
    assert resp.status_code == 200
    assert resp.json()["default_visual_style_id"] is None


def test_channel_update_invalid_style_fails(client: TestClient, test_channel: Channel):
    """Updating channel with an invalid style fails validation."""
    resp = client.patch(
        f"/channels/{test_channel.id}",
        json={"default_visual_style_id": "invalid_cyberpunk_neon"},
    )
    assert resp.status_code == 422


# ==============================================================================
# 3. Video Creation & Visual Style Inheritance
# ==============================================================================

def test_video_creation_missing_or_invalid_style_fails_422(
    client: TestClient, test_channel: Channel
):
    """New video requires an explicit valid visual style; missing, null, empty, or invalid fails with HTTP 422."""
    # 1. Missing visual_style_id fails with 422 even if channel has a default_visual_style_id
    resp = client.post(
        f"/channels/{test_channel.id}/videos",
        json={"topic": "How Jet Engines Work"},
    )
    assert resp.status_code == 422

    # 2. visual_style_id = None fails with 422
    resp_none = client.post(
        f"/channels/{test_channel.id}/videos",
        json={"topic": "How Jet Engines Work", "visual_style_id": None},
    )
    assert resp_none.status_code == 422

    # 3. visual_style_id = "" (blank/whitespace) fails with 422
    resp_empty = client.post(
        f"/channels/{test_channel.id}/videos",
        json={"topic": "How Jet Engines Work", "visual_style_id": "   "},
    )
    assert resp_empty.status_code == 422

    # 4. Unknown visual_style_id fails with 422
    resp_invalid = client.post(
        f"/channels/{test_channel.id}/videos",
        json={"topic": "How Jet Engines Work", "visual_style_id": "unknown_fantasy_art"},
    )
    assert resp_invalid.status_code == 422


def test_video_creation_succeeds_with_all_three_presets(
    client: TestClient, test_channel: Channel
):
    """Each of the 3 canonical presets can be explicitly selected when creating a video."""
    presets = ["legacy_default", "stickman_tactical_explainer", "stylized_2d_narrative"]
    for style_id in presets:
        resp = client.post(
            f"/channels/{test_channel.id}/videos",
            json={
                "topic": f"Topic for {style_id}",
                "visual_style_id": style_id,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["visual_style_id"] == style_id
        assert data["visual_style_version"] == 1

        detail_resp = client.get(f"/videos/{data['id']}")
        assert detail_resp.status_code == 200
        detail = detail_resp.json()
        assert detail["visual_style_id"] == style_id
        assert detail["visual_style_version"] == 1
        assert detail["visual_style_snapshot"] is not None
        assert detail["visual_style_snapshot"]["id"] == style_id


def test_existing_legacy_video_with_null_style_fields_preserved_and_prompt_parity(
    client: TestClient, db_session: Session, test_channel: Channel
):
    """Existing database records with NULL style fields are preserved as NULL and use legacy prompt path."""
    legacy_video = Video(
        channel_id=test_channel.id,
        topic="Pre-existing Legacy Video",
        description="Created before visual style system was introduced",
        visual_style_id=None,
        visual_style_version=None,
        visual_style_snapshot=None,
    )
    db_session.add(legacy_video)
    db_session.commit()
    db_session.refresh(legacy_video)

    # 1. Verify detail view returns None (no backfill or forced default on existing records)
    resp = client.get(f"/videos/{legacy_video.id}")
    assert resp.status_code == 200
    detail = resp.json()
    assert detail["visual_style_id"] is None
    assert detail["visual_style_version"] is None
    assert detail["visual_style_snapshot"] is None

    # 2. Verify summary in list_videos returns None
    list_resp = client.get(f"/channels/{test_channel.id}/videos")
    assert list_resp.status_code == 200
    summary = next(v for v in list_resp.json() if v["id"] == str(legacy_video.id))
    assert summary["visual_style_id"] is None
    assert summary["visual_style_version"] is None

    # 3. Verify legacy prompt composition path handles None style without failure
    from app.steps.prompt_compose import compose_final_prompt, adapt_visual_style
    from app.steps.storyboard import load_default_visual_style

    # adapt_visual_style with None returns empty string safely
    assert adapt_visual_style(None) == ""

    # compose_final_prompt with legacy default fallback produces original clean 3D prompt
    default_style = load_default_visual_style()
    scene = {
        "visual_description": "A streamlined rocket rising on a pillar of exhaust flames.",
        "narration": "A rocket launches into the night sky.",
    }
    prompt = compose_final_prompt(scene=scene, visual_style=default_style)
    assert "Clean 3D render" in prompt
    assert "soft studio lighting" in prompt


def test_video_creation_invalid_style_fails(client: TestClient, test_channel: Channel):
    """Explicitly providing unknown visual_style_id in video create fails validation."""
    resp = client.post(
        f"/channels/{test_channel.id}/videos",
        json={
            "topic": "Space Exploration",
            "visual_style_id": "invalid_retro_3d",
        },
    )
    assert resp.status_code == 422


# ==============================================================================
# 4. Immutability & Channel Updates Not Mutating Existing Videos
# ==============================================================================

def test_channel_default_update_does_not_mutate_existing_videos(
    client: TestClient, db_session: Session, test_channel: Channel
):
    """Updating a channel's default visual style does not change already created videos."""
    # 1. Create Video 1 with explicit style ("stickman_tactical_explainer")
    resp1 = client.post(
        f"/channels/{test_channel.id}/videos",
        json={"topic": "Video 1 - Explicit Style", "visual_style_id": "stickman_tactical_explainer"},
    )
    v1_id = resp1.json()["id"]

    # 2. Update channel default to "stylized_2d_narrative"
    client.patch(
        f"/channels/{test_channel.id}",
        json={"default_visual_style_id": "stylized_2d_narrative"},
    )

    # 3. Create Video 2 with explicit style ("stylized_2d_narrative")
    resp2 = client.post(
        f"/channels/{test_channel.id}/videos",
        json={"topic": "Video 2 - Explicit Style", "visual_style_id": "stylized_2d_narrative"},
    )
    v2_id = resp2.json()["id"]

    # 4. Verify Video 1 remains Stickman
    v1_detail = client.get(f"/videos/{v1_id}").json()
    assert v1_detail["visual_style_id"] == "stickman_tactical_explainer"
    assert v1_detail["visual_style_snapshot"]["id"] == "stickman_tactical_explainer"

    # 5. Verify Video 2 received Stylized 2D Narrative
    v2_detail = client.get(f"/videos/{v2_id}").json()
    assert v2_detail["visual_style_id"] == "stylized_2d_narrative"
    assert v2_detail["visual_style_snapshot"]["id"] == "stylized_2d_narrative"


# ==============================================================================
# 5. PATCH /videos/{video_id}/visual-style
# ==============================================================================

def test_patch_video_visual_style_success(
    client: TestClient, db_session: Session, test_channel: Channel
):
    """PATCH /videos/{video_id}/visual-style updates style and freezes new snapshot."""
    # Create video with legacy default
    resp = client.post(
        f"/channels/{test_channel.id}/videos",
        json={"topic": "Microbiology Basics", "visual_style_id": "legacy_default"},
    )
    vid_id = resp.json()["id"]

    # Initialize steps to ensure changing style does not delete steps in Phase 2
    video = db_session.get(Video, uuid.UUID(vid_id))
    init_steps(db_session, video)
    assert len(video.steps) > 0
    step_count_before = len(video.steps)

    # Patch to stickman_tactical_explainer
    patch_resp = client.patch(
        f"/videos/{vid_id}/visual-style",
        json={"visual_style_id": "stickman_tactical_explainer"},
    )
    assert patch_resp.status_code == 200
    data = patch_resp.json()

    assert data["visual_style_id"] == "stickman_tactical_explainer"
    assert data["visual_style_version"] == 1
    assert data["visual_style_snapshot"]["id"] == "stickman_tactical_explainer"
    assert data["visual_style_snapshot"]["display_name"] == STICKMAN_TACTICAL_EXPLAINER_PRESET.display_name
    # Steps are intact (not invalidated in Phase 2)
    assert len(data["steps"]) == step_count_before


def test_patch_video_visual_style_invalid_fails(client: TestClient, test_channel: Channel):
    """PATCH with unknown visual style returns validation error."""
    resp = client.post(
        f"/channels/{test_channel.id}/videos",
        json={"topic": "Ancient Civilizations", "visual_style_id": "legacy_default"},
    )
    vid_id = resp.json()["id"]

    patch_resp = client.patch(
        f"/videos/{vid_id}/visual-style",
        json={"visual_style_id": "unknown_pixel_art"},
    )
    assert patch_resp.status_code == 422


def test_patch_video_visual_style_same_style_is_idempotent_noop(
    client: TestClient, db_session: Session, test_channel: Channel
):
    """PATCH with already active canonical style is an idempotent no-op that preserves outputs and jobs."""
    # Create video with stickman style
    resp = client.post(
        f"/channels/{test_channel.id}/videos",
        json={"topic": "Quantum Computing", "visual_style_id": "stickman_tactical_explainer"},
    )
    vid_id = resp.json()["id"]

    video = db_session.get(Video, uuid.UUID(vid_id))
    init_steps(db_session, video)

    # Populate steps with outputs and done status, mark video completed
    for step in video.steps:
        step.status = "done"
        step.output = {step.name: {"data": f"mock_{step.name}_output"}}
    video.status = "completed"
    video.output_path = f"storage/{video.id}/final_video_captioned.mp4"

    # Enqueue a running pipeline job for images
    job = PipelineJob(
        video_id=video.id,
        step_name="images",
        status="running",
    )
    db_session.add(job)
    db_session.commit()
    db_session.refresh(video)

    # Count revisions before PATCH
    revs_before = len(db_session.query(VideoStepRevision).all())

    # Send same-style PATCH
    patch_resp = client.patch(
        f"/videos/{vid_id}/visual-style",
        json={"visual_style_id": "stickman_tactical_explainer"},
    )
    assert patch_resp.status_code == 200
    data = patch_resp.json()
    assert data["visual_style_id"] == "stickman_tactical_explainer"

    # Verify no-op preservation
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

    # Job must NOT be cancelled
    assert job.status == "running"

    # Revisions must NOT increase
    revs_after = len(db_session.query(VideoStepRevision).all())
    assert revs_after == revs_before


def test_patch_video_visual_style_legacy_null_to_legacy_default_freezes_snapshot(
    client: TestClient, db_session: Session, test_channel: Channel
):
    """PATCH from legacy NULL fields to legacy_default must NOT be short-circuited and must freeze snapshot."""
    # Create legacy video with all style fields NULL
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

    # Patch to explicit legacy_default
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

    # Verify database model is updated
    db_session.refresh(video)
    assert video.visual_style_id == "legacy_default"
    assert video.visual_style_version == 1
    assert video.visual_style_snapshot == LEGACY_DEFAULT_PRESET.model_dump()


# ==============================================================================
# 6. Alembic Migration Structure Verification
# ==============================================================================

def test_channel_partial_update_preserves_default_visual_style(
    client: TestClient, test_channel: Channel
):
    """Updating unrelated channel fields does not clear or overwrite default_visual_style_id."""
    assert test_channel.default_visual_style_id == "stickman_tactical_explainer"

    # Update name and niche only
    resp = client.patch(
        f"/channels/{test_channel.id}",
        json={"name": "Renamed Channel Name", "niche": "Tech Explanations"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "Renamed Channel Name"
    assert data["niche"] == "Tech Explanations"
    assert data["default_visual_style_id"] == "stickman_tactical_explainer"

    # Fetch channel to verify persisted state
    get_resp = client.get(f"/channels/{test_channel.id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["default_visual_style_id"] == "stickman_tactical_explainer"


def test_snapshot_stored_on_video_is_immutable_to_in_memory_preset_mutation(
    client: TestClient, db_session: Session, test_channel: Channel
):
    """Mutating preset objects or dictionaries in memory does not mutate stored video snapshots."""
    resp = client.post(
        f"/channels/{test_channel.id}/videos",
        json={"topic": "Snapshot Immutability Check", "visual_style_id": "legacy_default"},
    )
    vid_id = uuid.UUID(resp.json()["id"])
    video = db_session.get(Video, vid_id)
    original_display_name = video.visual_style_snapshot["display_name"]
    original_anchors = list(video.visual_style_snapshot["permanent_prompt_anchors"])

    # Attempt to mutate the in-memory dictionary retrieved from video
    video.visual_style_snapshot["display_name"] = "MUTATED DISPLAY NAME"
    video.visual_style_snapshot["permanent_prompt_anchors"].append("MUTATED ANCHOR")

    # Expire/refresh the session to reload from database
    db_session.expire(video)
    reloaded = db_session.get(Video, vid_id)

    assert reloaded.visual_style_snapshot["display_name"] == original_display_name
    assert reloaded.visual_style_snapshot["permanent_prompt_anchors"] == original_anchors


def test_patch_video_visual_style_auth_and_ownership(
    db_session: Session, test_user: User, other_user: User, test_channel: Channel
):
    """PATCH /videos/{video_id}/visual-style enforces 401 for unauthenticated and 404 for unowned videos."""
    # Create video owned by test_user
    video = Video(
        channel_id=test_channel.id,
        topic="Owned by test_user",
        visual_style_id="legacy_default",
    )
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)

    # 1. Unauthenticated client (no overrides)
    app.dependency_overrides.clear()
    with TestClient(app) as raw_client:
        unauth_resp = raw_client.patch(
            f"/videos/{video.id}/visual-style",
            json={"visual_style_id": "stickman_tactical_explainer"},
        )
        assert unauth_resp.status_code == 401

    # 2. Authenticated as other_user attempting to patch test_user's video
    def other_db():
        yield db_session

    def other_current_user():
        return other_user

    def other_owned_video(video_id: uuid.UUID):
        vid = db_session.get(Video, video_id)
        if not vid or vid.channel.user_id != other_user.id:
            from fastapi import HTTPException
            raise HTTPException(404, "Video not found")
        return vid

    app.dependency_overrides[get_db] = other_db
    app.dependency_overrides[current_user] = other_current_user
    app.dependency_overrides[get_owned_video] = other_owned_video

    try:
        with TestClient(app) as other_client:
            forbidden_resp = other_client.patch(
                f"/videos/{video.id}/visual-style",
                json={"visual_style_id": "stickman_tactical_explainer"},
            )
            assert forbidden_resp.status_code == 404
    finally:
        app.dependency_overrides.clear()


# ==============================================================================
# 6. Alembic Migration Upgrade & Downgrade Verification
# ==============================================================================

def test_alembic_visual_style_migration():
    """Verify Alembic migration metadata and operations for visual style fields."""
    from alembic.config import Config
    from alembic.script import ScriptDirectory

    cfg = Config("alembic.ini")
    script = ScriptDirectory.from_config(cfg)
    head_rev = script.get_current_head()
    assert head_rev == "c7a2e81d4b60"

    rev_step = script.get_revision(head_rev)
    assert rev_step.down_revision == "544f91af7a3f"
    assert "visual_style" in rev_step.doc.lower()


def test_alembic_visual_style_migration_upgrade_and_downgrade_sqlite():
    """Verify actual upgrade and downgrade execution on SQLite using Alembic MigrationContext."""
    import importlib
    import sqlalchemy as sa
    from sqlalchemy import create_engine, inspect
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    test_engine = create_engine("sqlite:///:memory:")
    with test_engine.begin() as conn:
        # Create base channels and videos tables simulating pre-migration schema
        conn.execute(sa.text("""
            CREATE TABLE channels (
                id VARCHAR(36) PRIMARY KEY,
                name VARCHAR(120) NOT NULL,
                platform VARCHAR(20) NOT NULL,
                settings JSON NOT NULL
            )
        """))
        conn.execute(sa.text("""
            CREATE TABLE videos (
                id VARCHAR(36) PRIMARY KEY,
                channel_id VARCHAR(36) NOT NULL,
                topic VARCHAR(300) NOT NULL,
                status VARCHAR(20) NOT NULL
            )
        """))

        ctx = MigrationContext.configure(conn, opts={"as_sql": False})
        op = Operations(ctx)

        # Import migration module via Alembic ScriptDirectory
        from alembic.config import Config
        from alembic.script import ScriptDirectory

        cfg = Config("alembic.ini")
        script = ScriptDirectory.from_config(cfg)
        mig = script.get_revision("c7a2e81d4b60").module
        orig_op = getattr(mig, "op", None)
        mig.op = op

        try:
            # 1. Run upgrade
            mig.upgrade()

            # Confirm all 4 columns exist with correct types and nullable settings
            insp = inspect(conn)
            ch_cols = {c["name"]: c for c in insp.get_columns("channels")}
            vid_cols = {c["name"]: c for c in insp.get_columns("videos")}

            assert "default_visual_style_id" in ch_cols, "channels.default_visual_style_id missing after upgrade"
            assert ch_cols["default_visual_style_id"]["nullable"] is True

            assert "visual_style_id" in vid_cols, "videos.visual_style_id missing after upgrade"
            assert vid_cols["visual_style_id"]["nullable"] is True

            assert "visual_style_version" in vid_cols, "videos.visual_style_version missing after upgrade"
            assert vid_cols["visual_style_version"]["nullable"] is True

            assert "visual_style_snapshot" in vid_cols, "videos.visual_style_snapshot missing after upgrade"
            assert vid_cols["visual_style_snapshot"]["nullable"] is True

            # 2. Run downgrade back to previous schema
            mig.downgrade()

            insp_after = inspect(conn)
            ch_cols_after = {c["name"] for c in insp_after.get_columns("channels")}
            vid_cols_after = {c["name"] for c in insp_after.get_columns("videos")}

            assert "default_visual_style_id" not in ch_cols_after, "channels.default_visual_style_id not removed after downgrade"
            assert "visual_style_id" not in vid_cols_after, "videos.visual_style_id not removed after downgrade"
            assert "visual_style_version" not in vid_cols_after, "videos.visual_style_version not removed after downgrade"
            assert "visual_style_snapshot" not in vid_cols_after, "videos.visual_style_snapshot not removed after downgrade"

        finally:
            mig.op = orig_op


def test_alembic_visual_style_migration_postgres_lifecycle():
    """Verify upgrade and downgrade lifecycle on real PostgreSQL database if reachable."""
    from sqlalchemy import create_engine, inspect, text
    from alembic import command
    from alembic.config import Config
    from app.db import SessionLocal, engine

    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL dialect required for PostgreSQL migration integration test")
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL container not reachable: {exc}")

    test_db_name = f"test_mig_style_{uuid.uuid4().hex[:8]}"

    # Create temporary database
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"CREATE DATABASE {test_db_name}"))

    test_url = f"postgresql+psycopg://studio:studio@localhost:5432/{test_db_name}"
    test_eng = None

    try:
        cfg = Config("alembic.ini")
        cfg.set_main_option("sqlalchemy.url", test_url)

        # 1. Upgrade to previous head (544f91af7a3f)
        command.upgrade(cfg, "544f91af7a3f")

        test_eng = create_engine(test_url)
        insp_pre = inspect(test_eng)
        ch_cols_pre = {c["name"] for c in insp_pre.get_columns("channels")}
        vid_cols_pre = {c["name"] for c in insp_pre.get_columns("videos")}
        assert "default_visual_style_id" not in ch_cols_pre
        assert "visual_style_id" not in vid_cols_pre
        test_eng.dispose()

        # 2. Upgrade to new revision (c7a2e81d4b60)
        command.upgrade(cfg, "c7a2e81d4b60")

        test_eng = create_engine(test_url)
        insp_post = inspect(test_eng)
        ch_cols_post = {c["name"]: c for c in insp_post.get_columns("channels")}
        vid_cols_post = {c["name"]: c for c in insp_post.get_columns("videos")}

        assert "default_visual_style_id" in ch_cols_post
        assert ch_cols_post["default_visual_style_id"]["nullable"] is True

        assert "visual_style_id" in vid_cols_post
        assert vid_cols_post["visual_style_id"]["nullable"] is True

        assert "visual_style_version" in vid_cols_post
        assert vid_cols_post["visual_style_version"]["nullable"] is True

        assert "visual_style_snapshot" in vid_cols_post
        assert vid_cols_post["visual_style_snapshot"]["nullable"] is True
        test_eng.dispose()

        # 3. Downgrade back to previous revision (544f91af7a3f)
        command.downgrade(cfg, "544f91af7a3f")

        test_eng = create_engine(test_url)
        insp_down = inspect(test_eng)
        ch_cols_down = {c["name"] for c in insp_down.get_columns("channels")}
        vid_cols_down = {c["name"] for c in insp_down.get_columns("videos")}
        assert "default_visual_style_id" not in ch_cols_down
        assert "visual_style_id" not in vid_cols_down
        assert "visual_style_version" not in vid_cols_down
        assert "visual_style_snapshot" not in vid_cols_down
        test_eng.dispose()

    finally:
        if test_eng:
            test_eng.dispose()
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(f"DROP DATABASE IF EXISTS {test_db_name} WITH (FORCE)"))
