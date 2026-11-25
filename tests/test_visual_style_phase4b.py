"""Phase 4B Tests — Visual Style Invalidation & Image Cache Correctness.

Verifies:
1. Selective invalidation on visual style change (script, bible, voiceover preserved; storyboard prompts recomposed; images, render, subtitles stale).
2. Deterministic storyboard prompt recomposition preserving all semantic fields with zero LLM calls.
3. Image cache key identity covering all 9 canonical dimensions.
4. Reference image byte hashing and deterministic ordering.
5. Stale render and image outputs cannot be presented as current.
6. Atomic rollback on failed style update.
7. Concurrency and stale job protection across queue worker and direct pipeline execution.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
from typing import Any
import uuid

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.db import Base, get_db
from app.deps import current_user, get_owned_channel, get_owned_video
from app.main import app
from app.models import Channel, PipelineJob, User, Video, VideoStep, VideoStepRevision
from app.pipeline import init_steps, record_revision
from app.queue.engine import claim_next_job, complete_job, enqueue_step_job
from app.steps.images import (
    compute_image_cache_key,
    generate_images_step,
    hash_reference_images,
)
from app.steps.prompt_compose import recompute_final_prompts
from app.styles import (
    DEFAULT_STYLE_ID,
    LEGACY_DEFAULT_PRESET,
    STICKMAN_TACTICAL_EXPLAINER_PRESET,
    STYLIZED_2D_NARRATIVE_PRESET,
    canonical_snapshot_hash,
    get_style,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(autouse=True)
def mock_llm_settings(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_provider_storyboard", "mock")


@pytest.fixture
def db_session():
    """Isolated in-memory SQLite database for testing."""
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
        full_name="Phase 4B User",
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


@pytest.fixture
def test_channel(db_session: Session, test_user: User) -> Channel:
    channel = Channel(
        user_id=test_user.id,
        name="Test 4B Channel",
        handle="@test4b",
        default_visual_style_id="legacy_default",
    )
    db_session.add(channel)
    db_session.commit()
    db_session.refresh(channel)
    return channel


@pytest.fixture
def client(db_session: Session, test_user: User) -> TestClient:
    app.dependency_overrides[get_db] = lambda: db_session
    app.dependency_overrides[current_user] = lambda: test_user
    app.dependency_overrides[get_owned_channel] = lambda: test_user.channels[0]
    yield TestClient(app)
    app.dependency_overrides.clear()


def _populate_mock_completed_pipeline(db: Session, video: Video) -> None:
    """Helper to populate a video with full 8-step completed pipeline outputs."""
    init_steps(db, video)
    
    # 1. Creative Direction
    cd_step = next(s for s in video.steps if s.name == "creative_direction")
    cd_step.status = "done"
    cd_step.output = {
        "creative_direction": {
            "hook_angle": "Surprising science hook",
            "pacing": "fast",
        }
    }

    # 2. Script
    script_step = next(s for s in video.steps if s.name == "script")
    script_step.status = "done"
    script_step.output = {
        "script": {
            "narration": "Have you ever wondered why stars flicker? It is actually Earth's atmosphere.",
            "scenes_text": [
                "Have you ever wondered why stars flicker?",
                "It is actually Earth's atmosphere.",
            ],
        }
    }

    # 3. Video Bible
    vb_step = next(s for s in video.steps if s.name == "video_bible")
    vb_step.status = "done"
    vb_step.output = {
        "video_bible": {
            "characters": [{"id": "astronomer_jenil", "name": "Astronomer Jenil"}],
            "environments": [{"id": "observatory", "name": "Hilltop Observatory"}],
        }
    }

    # 4. Storyboard
    sb_step = next(s for s in video.steps if s.name == "storyboard")
    sb_step.status = "done"
    sb_step.output = {
        "storyboard": {
            "visual_style": "Clean 3D Studio (Default)",
            "subject_anchor": "stylized astronomer",
            "environment_anchor": "night observatory",
            "scenes": [
                {
                    "index": 1,
                    "scene_id": "scene_1",
                    "narration": "Have you ever wondered why stars flicker?",
                    "start_time": 0.0,
                    "end_time": 3.0,
                    "duration_seconds": 3.0,
                    "shot_type": "medium_shot",
                    "camera_angle": "eye_level",
                    "camera_direction": {"pan": "right", "tilt": "none"},
                    "actions_and_events": ["Look through telescope"],
                    "entity_bindings": {"character_ids": ["astronomer_jenil"], "environment_ids": ["observatory"]},
                    "image_prompt": "An astronomer looking through a telescope at the starry night sky.",
                    "final_prompt": "Clean 3D render of an astronomer looking through telescope.",
                    "visual_beats": [
                        {
                            "id": "beat_1_1",
                            "order": 1,
                            "visual_description": "Astronomer focusing eyepiece",
                            "final_prompt": "Clean 3D render of astronomer focusing eyepiece.",
                        }
                    ],
                },
                {
                    "index": 2,
                    "scene_id": "scene_2",
                    "narration": "It is actually Earth's atmosphere.",
                    "start_time": 3.0,
                    "end_time": 6.0,
                    "duration_seconds": 3.0,
                    "shot_type": "wide_shot",
                    "camera_angle": "low_angle",
                    "camera_direction": {"pan": "none", "tilt": "up"},
                    "actions_and_events": ["Show atmospheric air turbulence"],
                    "entity_bindings": {"environment_ids": ["observatory"]},
                    "image_prompt": "Atmospheric air layers refracting starlight.",
                    "final_prompt": "Clean 3D render of atmosphere layers refracting starlight.",
                    "visual_beats": [
                        {
                            "id": "beat_2_1",
                            "order": 1,
                            "visual_description": "Atmosphere layers waving",
                            "final_prompt": "Clean 3D render of atmosphere layers waving.",
                        }
                    ],
                },
            ],
        }
    }

    # 5. Images
    img_step = next(s for s in video.steps if s.name == "images")
    img_step.status = "done"
    img_step.output = {
        "images": {
            "items": [
                {
                    "index": 1,
                    "id": "beat_1_1",
                    "file": f"{video.id}/images/scene_01_old12345.png",
                    "status": "ok",
                    "prompt": "Clean 3D render of an astronomer looking through telescope.",
                    "prompt_hash": "old12345fullhash",
                },
                {
                    "index": 2,
                    "id": "beat_2_1",
                    "file": f"{video.id}/images/scene_02_old67890.png",
                    "status": "ok",
                    "prompt": "Clean 3D render of atmosphere layers refracting starlight.",
                    "prompt_hash": "old67890fullhash",
                },
            ],
            "complete": True,
        },
        "meta": {"task": "images"},
    }

    # 6. Voiceover
    vo_step = next(s for s in video.steps if s.name == "voiceover")
    vo_step.status = "done"
    vo_step.output = {
        "voiceover": {
            "audio_file": f"{video.id}/audio/narration_abc12345.wav",
            "audio_hash": "abc12345fullaudiohash",
            "duration_seconds": 6.0,
            "items": [
                {"index": 1, "status": "ok", "start_time": 0.0, "end_time": 3.0, "duration_seconds": 3.0, "timing_type": "aligned"},
                {"index": 2, "status": "ok", "start_time": 3.0, "end_time": 6.0, "duration_seconds": 3.0, "timing_type": "aligned"},
            ],
        }
    }

    # 7. Render
    render_step = next(s for s in video.steps if s.name == "render")
    render_step.status = "done"
    render_step.output = {
        "render": {
            "video_path": f"/tmp/{video.id}/final_video.mp4",
            "duration_seconds": 6.0,
        },
        "meta": {"task": "render"},
    }
    video.output_path = f"/tmp/{video.id}/final_video.mp4"

    # 8. Subtitles
    sub_step = next(s for s in video.steps if s.name in ("subtitles", "subtitle"))
    sub_step.status = "done"
    sub_step.output = {
        "subtitles": {
            "captioned_video_path": f"/tmp/{video.id}/final_video_captioned.mp4",
        },
        "meta": {"task": "subtitles", "captioned_burned": True},
    }

    video.status = "completed"
    db.commit()


# ==============================================================================
# 1. Objective A — Selective Invalidation on Style Change Tests
# ==============================================================================

def test_style_invalidation_legacy_to_stickman(client: TestClient, db_session: Session, test_channel: Channel):
    """Changing from legacy to Stickman invalidates visual steps while preserving script, bible, voiceover."""
    # 1. Create completed legacy video
    video = Video(
        channel_id=test_channel.id,
        topic="Atmospheric Science",
        visual_style_id="legacy_default",
        visual_style_version=1,
        visual_style_snapshot=LEGACY_DEFAULT_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)
    _populate_mock_completed_pipeline(db_session, video)

    # 2. Patch visual style to stickman_tactical_explainer
    resp = client.patch(
        f"/videos/{video.id}/visual-style",
        json={"visual_style_id": "stickman_tactical_explainer"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["visual_style_id"] == "stickman_tactical_explainer"
    assert data["visual_style_version"] == 1
    assert data["visual_style_snapshot"]["id"] == "stickman_tactical_explainer"

    # Refresh DB model
    db_session.refresh(video)
    assert video.status == "in_progress"
    assert video.output_path is None

    # Verify preserved non-visual steps
    step_map = {s.name: s for s in video.steps}
    assert step_map["creative_direction"].status == "done"
    assert step_map["creative_direction"].output is not None

    assert step_map["script"].status == "done"
    assert step_map["script"].output["script"]["narration"].startswith("Have you ever wondered")

    assert step_map["video_bible"].status == "done"
    assert len(step_map["video_bible"].output["video_bible"]["characters"]) == 1

    assert step_map["voiceover"].status == "done"
    assert step_map["voiceover"].output["voiceover"]["audio_file"] == f"{video.id}/audio/narration_abc12345.wav"
    assert len(step_map["voiceover"].output["voiceover"]["items"]) == 2

    # Verify storyboard semantic data preserved and prompts recomposed
    sb_step = step_map["storyboard"]
    assert sb_step.status == "done"
    scenes = sb_step.output["storyboard"]["scenes"]
    assert len(scenes) == 2
    assert scenes[0]["narration"] == "Have you ever wondered why stars flicker?"
    assert scenes[0]["shot_type"] == "medium_shot"
    assert scenes[0]["camera_direction"] == {"pan": "right", "tilt": "none"}
    assert scenes[0]["actions_and_events"] == ["Look through telescope"]
    assert scenes[0]["entity_bindings"]["character_ids"] == ["astronomer_jenil"]
    # Recomposed prompt should now carry Stickman anchors/linework rules
    assert "clean 2D animated stickman illustration style" in scenes[0]["final_prompt"]

    # Verify visual steps invalidated
    assert step_map["images"].status == "stale"
    assert step_map["images"].output is None

    assert step_map["render"].status == "stale"
    assert step_map["render"].output is None

    assert step_map["subtitles"].status == "stale"
    assert step_map["subtitles"].output is None

    # Verify revision records exist
    revs = db_session.scalars(select(VideoStepRevision).where(VideoStepRevision.step_id == sb_step.id)).all()
    assert len(revs) >= 1
    assert "Recomputed visual prompts for style 'stickman_tactical_explainer'" in revs[-1].change_summary


def test_style_invalidation_stickman_to_stylized_2d(client: TestClient, db_session: Session, test_channel: Channel):
    """Changing from Stickman to Stylized 2D recomposes prompts to Stylized 2D rules."""
    video = Video(
        channel_id=test_channel.id,
        topic="Deep Sea Biology",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)
    _populate_mock_completed_pipeline(db_session, video)

    resp = client.patch(
        f"/videos/{video.id}/visual-style",
        json={"visual_style_id": "stylized_2d_narrative"},
    )
    assert resp.status_code == 200
    db_session.refresh(video)
    assert video.visual_style_id == "stylized_2d_narrative"

    step_map = {s.name: s for s in video.steps}
    # Voiceover preserved
    assert step_map["voiceover"].status == "done"
    # Prompts recomposed to Stylized 2D
    sb_scenes = step_map["storyboard"].output["storyboard"]["scenes"]
    assert "stylized 2D digital cartoon animation art" in sb_scenes[0]["final_prompt"]
    # Downstream stale
    assert step_map["images"].status == "stale"
    assert step_map["render"].status == "stale"
    assert step_map["subtitles"].status == "stale"


def test_stale_render_output_cannot_be_presented_as_current(client: TestClient, db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    """Stale render outputs return HTTP 409 and stale images return 404."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    video = Video(
        channel_id=test_channel.id,
        topic="Space Physics",
        visual_style_id="legacy_default",
        visual_style_version=1,
        visual_style_snapshot=LEGACY_DEFAULT_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)
    _populate_mock_completed_pipeline(db_session, video)

    # Patch style to stickman
    client.patch(
        f"/videos/{video.id}/visual-style",
        json={"visual_style_id": "stickman_tactical_explainer"},
    )

    # 1. Download uncaptioned video returns 409
    dl_resp_orig = client.get(f"/videos/{video.id}/download?captioned=false")
    assert dl_resp_orig.status_code == 409

    # 2. Download captioned video returns 409
    dl_resp_cap = client.get(f"/videos/{video.id}/download?captioned=true")
    assert dl_resp_cap.status_code == 409

    # 3. Scene image endpoint returns 404
    img_resp = client.get(f"/videos/{video.id}/images/1")
    assert img_resp.status_code == 404


def test_failed_style_update_does_not_leave_partial_database_changes(client: TestClient, db_session: Session, test_channel: Channel):
    """If style update fails (e.g. unknown style), transaction rolls back cleanly."""
    video = Video(
        channel_id=test_channel.id,
        topic="Roman Architecture",
        visual_style_id="legacy_default",
        visual_style_version=1,
        visual_style_snapshot=LEGACY_DEFAULT_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)
    _populate_mock_completed_pipeline(db_session, video)

    resp = client.patch(
        f"/videos/{video.id}/visual-style",
        json={"visual_style_id": "completely_unknown_preset_xyz"},
    )
    assert resp.status_code == 422 or resp.status_code == 400

    db_session.refresh(video)
    assert video.visual_style_id == "legacy_default"
    assert video.status == "completed"
    assert video.output_path is not None


# ==============================================================================
# 2. Objective B — Image Cache Correctness Tests
# ==============================================================================

def test_image_cache_identical_complete_inputs_produce_identical_keys():
    """Identical complete inputs across all 9 canonical dimensions produce identical cache keys."""
    prompt = "A high-tech laboratory with glowing holographic charts."
    snap = STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump()
    ref_bytes = [b"mock_png_header_and_pixels_one", b"mock_png_header_and_pixels_two"]

    key1 = compute_image_cache_key(
        prompt=prompt,
        provider_name="openai",
        model="dall-e-3",
        size="1024x1536",
        quality="low",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=snap,
        reference_images=ref_bytes,
    )
    key2 = compute_image_cache_key(
        prompt=prompt,
        provider_name="openai",
        model="dall-e-3",
        size="1024x1536",
        quality="low",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=snap,
        reference_images=ref_bytes,
    )
    assert len(key1) == 64
    assert key1 == key2


def test_image_cache_different_style_ids_produce_different_keys():
    """Different visual style IDs produce different cache keys."""
    snap = STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump()
    key_stick = compute_image_cache_key(
        prompt="A stickman on a hill",
        provider_name="mock",
        model="mock-v1",
        size="1024x1024",
        quality="standard",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=snap,
    )
    key_stylized = compute_image_cache_key(
        prompt="A stickman on a hill",
        provider_name="mock",
        model="mock-v1",
        size="1024x1024",
        quality="standard",
        visual_style_id="stylized_2d_narrative",
        visual_style_version=1,
        visual_style_snapshot=snap,
    )
    assert key_stick != key_stylized


def test_image_cache_different_style_versions_produce_different_keys():
    """Different visual style versions produce different cache keys."""
    snap = STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump()
    key_v1 = compute_image_cache_key(
        prompt="A figure running",
        provider_name="mock",
        model="mock-v1",
        size="1024x1024",
        quality="standard",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=snap,
    )
    key_v2 = compute_image_cache_key(
        prompt="A figure running",
        provider_name="mock",
        model="mock-v1",
        size="1024x1024",
        quality="standard",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=2,
        visual_style_snapshot=snap,
    )
    assert key_v1 != key_v2


def test_image_cache_different_snapshot_contents_produce_different_keys():
    """Mutating frozen snapshot contents produces a different cache key."""
    snap1 = STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump()
    snap2 = copy.deepcopy(snap1)
    snap2["permanent_prompt_anchors"].append("added extra anchor")

    key1 = compute_image_cache_key(
        prompt="Sample scene",
        provider_name="mock",
        model="mock-v1",
        size="1024x1024",
        quality="standard",
        visual_style_snapshot=snap1,
    )
    key2 = compute_image_cache_key(
        prompt="Sample scene",
        provider_name="mock",
        model="mock-v1",
        size="1024x1024",
        quality="standard",
        visual_style_snapshot=snap2,
    )
    assert key1 != key2


def test_image_cache_different_final_prompts_produce_different_keys():
    """Different final composed prompts produce different cache keys."""
    key1 = compute_image_cache_key(
        prompt="Prompt version A",
        provider_name="mock",
        model="mock-v1",
        size="1024x1024",
        quality="standard",
    )
    key2 = compute_image_cache_key(
        prompt="Prompt version B",
        provider_name="mock",
        model="mock-v1",
        size="1024x1024",
        quality="standard",
    )
    assert key1 != key2


def test_image_cache_different_provider_models_sizes_qualities_produce_different_keys():
    """Varying provider model, size, or quality produces distinct keys."""
    base = {
        "prompt": "Test scene",
        "provider_name": "openai",
        "model": "dall-e-3",
        "size": "1024x1536",
        "quality": "low",
    }
    k_base = compute_image_cache_key(**base)

    k_diff_prov = compute_image_cache_key(**{**base, "provider_name": "custom"})
    k_diff_model = compute_image_cache_key(**{**base, "model": "dall-e-2"})
    k_diff_size = compute_image_cache_key(**{**base, "size": "1024x1024"})
    k_diff_qual = compute_image_cache_key(**{**base, "quality": "hd"})

    assert k_base != k_diff_prov
    assert k_base != k_diff_model
    assert k_base != k_diff_size
    assert k_base != k_diff_qual


def test_image_cache_different_reference_contents_produce_different_keys():
    """Changed reference image bytes alter cache identity."""
    ref_a = [b"bytes_version_1"]
    ref_b = [b"bytes_version_2"]

    key_a = compute_image_cache_key(
        prompt="A machine diagram",
        provider_name="mock",
        model="mock-v1",
        size="1024x1024",
        quality="standard",
        reference_images=ref_a,
    )
    key_b = compute_image_cache_key(
        prompt="A machine diagram",
        provider_name="mock",
        model="mock-v1",
        size="1024x1024",
        quality="standard",
        reference_images=ref_b,
    )
    assert key_a != key_b


def test_image_cache_reference_ordering_is_deterministic():
    """Reference ordering is deterministic: swapping reference order changes key identity."""
    ref1 = b"image_asset_first"
    ref2 = b"image_asset_second"

    key_ordered = compute_image_cache_key(
        prompt="Ordered references scene",
        provider_name="mock",
        model="mock-v1",
        size="1024x1024",
        quality="standard",
        reference_images=[ref1, ref2],
    )
    key_reversed = compute_image_cache_key(
        prompt="Ordered references scene",
        provider_name="mock",
        model="mock-v1",
        size="1024x1024",
        quality="standard",
        reference_images=[ref2, ref1],
    )
    assert key_ordered != key_reversed


def test_image_cache_no_reference_compatible():
    """No-reference generation works compatibly and deterministically."""
    key_none = compute_image_cache_key(
        prompt="Scene with no references",
        provider_name="mock",
        model="mock-v1",
        size="1024x1024",
        quality="standard",
        reference_images=None,
    )
    key_empty = compute_image_cache_key(
        prompt="Scene with no references",
        provider_name="mock",
        model="mock-v1",
        size="1024x1024",
        quality="standard",
        reference_images=[],
    )
    assert key_none == key_empty


# ==============================================================================
# 3. Objective C — Concurrency & Stale Job Protection Tests
# ==============================================================================

def test_old_style_image_job_cannot_publish_after_style_change(db_session: Session, test_channel: Channel):
    """A running image job using Stickman is prevented from publishing output after style is changed."""
    # 1. Create video with stickman
    video = Video(
        channel_id=test_channel.id,
        topic="Quantum Computing",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)
    init_steps(db_session, video)

    # 2. Enqueue and claim images job
    job = enqueue_step_job(db_session, video.id, "images")
    claimed = claim_next_job(db_session, worker_id="worker-test-1")
    assert claimed is not None
    claimed_job, lease_token = claimed

    # 3. User changes visual style concurrently to stylized_2d_narrative
    # Emulate style patch endpoint logic
    video.visual_style_id = "stylized_2d_narrative"
    video.visual_style_version = 1
    video.visual_style_snapshot = STYLIZED_2D_NARRATIVE_PRESET.model_dump()
    img_step = next(s for s in video.steps if s.name == "images")
    img_step.status = "stale"
    claimed_job.status = "cancelled"
    db_session.commit()

    # 4. Old job finishes and tries to complete
    old_output = {
        "images": {"items": [{"index": 1, "file": "old_stickman.png", "status": "ok"}]},
        "meta": {
            "task": "images",
            "visual_style_id": "stickman_tactical_explainer",
            "visual_style_snapshot_hash": canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump()),
        },
    }
    success = complete_job(
        db=db_session,
        job_id=claimed_job.id,
        lease_token=lease_token,
        output=old_output,
    )
    assert success is False

    # 5. Verify step output was NOT overwritten with old images
    db_session.refresh(img_step)
    assert img_step.status == "stale"
    assert img_step.output is None


def test_old_style_render_job_cannot_publish_after_style_change(db_session: Session, test_channel: Channel):
    """A running render job is rejected from publishing video or setting output_path after style change."""
    video = Video(
        channel_id=test_channel.id,
        topic="Astrobiology",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)
    init_steps(db_session, video)

    # Enqueue and claim render job
    job = enqueue_step_job(db_session, video.id, "render")
    claimed = claim_next_job(db_session, worker_id="worker-test-2")
    assert claimed is not None
    claimed_job, lease_token = claimed

    # Concurrently style is changed
    video.visual_style_id = "stylized_2d_narrative"
    video.visual_style_version = 1
    video.visual_style_snapshot = STYLIZED_2D_NARRATIVE_PRESET.model_dump()
    render_step = next(s for s in video.steps if s.name == "render")
    render_step.status = "stale"
    claimed_job.status = "cancelled"
    db_session.commit()

    # Old render job tries to complete
    old_render_output = {
        "render": {"video_path": "/tmp/old_render.mp4", "duration_seconds": 5.0},
        "meta": {
            "task": "render",
            "visual_style_id": "stickman_tactical_explainer",
            "visual_style_snapshot_hash": canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump()),
        },
    }
    success = complete_job(
        db=db_session,
        job_id=claimed_job.id,
        lease_token=lease_token,
        output=old_render_output,
    )
    assert success is False

    db_session.refresh(render_step)
    db_session.refresh(video)
    assert render_step.status == "stale"
    assert render_step.output is None
    assert video.output_path is None


def test_current_style_jobs_publish_normally(db_session: Session, test_channel: Channel):
    """A job matching the video's current visual style completes and publishes normally."""
    video = Video(
        channel_id=test_channel.id,
        topic="Robotics",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)
    init_steps(db_session, video)

    job = enqueue_step_job(db_session, video.id, "images")
    claimed = claim_next_job(db_session, worker_id="worker-test-3")
    assert claimed is not None
    claimed_job, lease_token = claimed

    curr_snap_hash = canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump())
    new_output = {
        "images": {"items": [{"index": 1, "file": "stickman_img.png", "status": "ok"}]},
        "meta": {
            "task": "images",
            "visual_style_id": "stickman_tactical_explainer",
            "visual_style_snapshot_hash": curr_snap_hash,
        },
    }
    success = complete_job(
        db=db_session,
        job_id=claimed_job.id,
        lease_token=lease_token,
        output=new_output,
    )
    assert success is True

    img_step = next(s for s in video.steps if s.name == "images")
    db_session.refresh(img_step)
    assert img_step.status in ("done", "awaiting_approval")
    assert img_step.output["images"]["items"][0]["file"] == "stickman_img.png"


def test_old_job_finishing_after_new_job_has_started(db_session: Session, test_channel: Channel):
    """Critical scenario: Old job finishes after a new job has started and changed step status back to running."""
    # 1. Video starts in Stickman
    video = Video(
        channel_id=test_channel.id,
        topic="Neural Networks",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)
    init_steps(db_session, video)

    # 2. Enqueue and claim Stickman job 1
    job_1 = enqueue_step_job(db_session, video.id, "images")
    claimed_1 = claim_next_job(db_session, worker_id="worker-old")
    assert claimed_1 is not None
    claimed_job_1, lease_token_1 = claimed_1

    # 3. User changes visual style to Stylized 2D
    video.visual_style_id = "stylized_2d_narrative"
    video.visual_style_version = 1
    video.visual_style_snapshot = STYLIZED_2D_NARRATIVE_PRESET.model_dump()
    img_step = next(s for s in video.steps if s.name == "images")
    img_step.status = "stale"
    claimed_job_1.status = "cancelled"
    db_session.commit()

    # 4. New Stylized 2D job 2 is enqueued and claimed, resetting step status to running
    job_2 = enqueue_step_job(db_session, video.id, "images")
    claimed_2 = claim_next_job(db_session, worker_id="worker-new")
    assert claimed_2 is not None
    claimed_job_2, lease_token_2 = claimed_2

    db_session.refresh(img_step)
    assert img_step.status == "running"

    # 5. Old Stickman job 1 finishes now and attempts to complete
    old_output = {
        "images": {"items": [{"index": 1, "file": "stale_stickman.png", "status": "ok"}]},
        "meta": {
            "task": "images",
            "visual_style_id": "stickman_tactical_explainer",
            "visual_style_snapshot_hash": canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump()),
        },
    }
    old_success = complete_job(
        db=db_session,
        job_id=claimed_job_1.id,
        lease_token=lease_token_1,
        output=old_output,
    )
    assert old_success is False

    # 6. Verify step is STILL running under Job 2, output is NOT overwritten
    db_session.refresh(img_step)
    assert img_step.status == "running"
    assert img_step.output is None

    # 7. New Stylized 2D Job 2 finishes and publishes output
    new_snap_hash = canonical_snapshot_hash(STYLIZED_2D_NARRATIVE_PRESET.model_dump())
    new_output = {
        "images": {"items": [{"index": 1, "file": "current_stylized2d.png", "status": "ok"}]},
        "meta": {
            "task": "images",
            "visual_style_id": "stylized_2d_narrative",
            "visual_style_snapshot_hash": new_snap_hash,
        },
    }
    new_success = complete_job(
        db=db_session,
        job_id=claimed_job_2.id,
        lease_token=lease_token_2,
        output=new_output,
    )
    assert new_success is True

    # 8. Step is now done with new Stylized 2D output
    db_session.refresh(img_step)
    assert img_step.status in ("done", "awaiting_approval")
    assert img_step.output["images"]["items"][0]["file"] == "current_stylized2d.png"


def test_old_subtitle_job_attempting_publication_after_style_change(db_session: Session, test_channel: Channel):
    """A running subtitle job is rejected from publishing subtitles after style change."""
    video = Video(
        channel_id=test_channel.id,
        topic="Climate Patterns",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)
    init_steps(db_session, video)

    job = enqueue_step_job(db_session, video.id, "subtitles")
    claimed = claim_next_job(db_session, worker_id="worker-sub")
    assert claimed is not None
    claimed_job, lease_token = claimed

    # Style change occurs concurrently
    video.visual_style_id = "stylized_2d_narrative"
    video.visual_style_version = 1
    video.visual_style_snapshot = STYLIZED_2D_NARRATIVE_PRESET.model_dump()
    sub_step = next(s for s in video.steps if s.name in ("subtitles", "subtitle"))
    sub_step.status = "stale"
    claimed_job.status = "cancelled"
    db_session.commit()

    old_sub_output = {
        "subtitles": {"captioned_video_path": "/tmp/old_captioned.mp4"},
        "meta": {
            "task": "subtitles",
            "visual_style_id": "stickman_tactical_explainer",
            "visual_style_snapshot_hash": canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump()),
        },
    }
    success = complete_job(
        db=db_session,
        job_id=claimed_job.id,
        lease_token=lease_token,
        output=old_sub_output,
    )
    assert success is False

    db_session.refresh(sub_step)
    assert sub_step.status == "stale"
    assert sub_step.output is None


def test_old_job_attempting_publication_after_lease_has_been_replaced(db_session: Session, test_channel: Channel):
    """A worker whose lease token was revoked or replaced cannot publish output."""
    video = Video(
        channel_id=test_channel.id,
        topic="Plate Tectonics",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)
    init_steps(db_session, video)

    job = enqueue_step_job(db_session, video.id, "images")
    claimed = claim_next_job(db_session, worker_id="worker-1")
    assert claimed is not None
    claimed_job, lease_token_1 = claimed

    # Simulate lease expiry and recovery by worker-2 (fencing token replacement)
    stale_lease_token = lease_token_1
    new_lease_token = uuid.uuid4()
    claimed_job.lease_token = new_lease_token
    claimed_job.worker_id = "worker-2"
    db_session.commit()

    # Worker-1 tries to publish with stale lease token
    output = {
        "images": {"items": [{"index": 1, "file": "w1_img.png", "status": "ok"}]},
        "meta": {
            "task": "images",
            "visual_style_snapshot_hash": canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump()),
        },
    }
    success = complete_job(
        db=db_session,
        job_id=claimed_job.id,
        lease_token=stale_lease_token,
        output=output,
    )
    assert success is False

    img_step = next(s for s in video.steps if s.name == "images")
    db_session.refresh(img_step)
    assert img_step.output is None


def test_old_partial_image_persistence_attempting_to_overwrite_newer_output(db_session: Session, test_channel: Channel):
    """Stale partial image persistence does not overwrite newer step output."""
    video = Video(
        channel_id=test_channel.id,
        topic="Evolutionary Trees",
        visual_style_id="stylized_2d_narrative",
        visual_style_version=1,
        visual_style_snapshot=STYLIZED_2D_NARRATIVE_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)
    init_steps(db_session, video)

    img_step = next(s for s in video.steps if s.name == "images")
    new_hash = canonical_snapshot_hash(STYLIZED_2D_NARRATIVE_PRESET.model_dump())
    img_step.output = {
        "images": {"items": [{"index": 1, "file": "stylized_img_1.png", "status": "ok"}]},
        "meta": {"visual_style_snapshot_hash": new_hash},
    }
    db_session.commit()

    # Old Stickman job attempts to save partial progress
    old_hash = canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump())
    stale_payload = {
        "images": {"items": [{"index": 1, "file": "old_stickman_1.png", "status": "ok"}]},
        "meta": {"visual_style_snapshot_hash": old_hash},
    }
    from app.steps.images import _persist_partial
    _persist_partial(
        step=img_step,
        payload=stale_payload,
        initial_snapshot_hash=old_hash,
        initial_style_id="stickman_tactical_explainer",
    )

    db_session.refresh(img_step)
    # Output must still be the newer Stylized 2D output
    assert img_step.output["images"]["items"][0]["file"] == "stylized_img_1.png"


def test_stale_job_not_changing_newer_jobs_status_or_output(db_session: Session, test_channel: Channel):
    """A failing or stale job cannot mark a newer job's step as failed or alter its status."""
    video = Video(
        channel_id=test_channel.id,
        topic="Volcanoes",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    db_session.refresh(video)
    init_steps(db_session, video)

    job_1 = enqueue_step_job(db_session, video.id, "images")
    claimed_1 = claim_next_job(db_session, worker_id="worker-old")
    assert claimed_1 is not None
    claimed_job_1, lease_token_1 = claimed_1

    # Style change and newer job starts
    video.visual_style_id = "stylized_2d_narrative"
    video.visual_style_snapshot = STYLIZED_2D_NARRATIVE_PRESET.model_dump()
    claimed_job_1.status = "cancelled"
    db_session.commit()

    job_2 = enqueue_step_job(db_session, video.id, "images")
    claimed_2 = claim_next_job(db_session, worker_id="worker-new")
    assert claimed_2 is not None

    img_step = next(s for s in video.steps if s.name == "images")
    db_session.refresh(img_step)
    assert img_step.status == "running"

    # Stale Job 1 fails with an error
    from app.queue.engine import fail_job
    fail_res = fail_job(
        db=db_session,
        job_id=claimed_job_1.id,
        lease_token=lease_token_1,
        error_message="Image generation timed out in old worker",
    )
    assert fail_res is False

    # Newer step is STILL running, NOT marked failed
    db_session.refresh(img_step)
    db_session.refresh(video)
    assert img_step.status == "running"
    assert video.status == "in_progress"


def test_old_render_job_does_not_overwrite_newer_file_on_disk(tmp_path: Path, monkeypatch):
    """A stale render job does not overwrite a newer render file on disk."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    video_id = uuid.uuid4()
    video_dir = tmp_path / str(video_id)
    video_dir.mkdir(parents=True, exist_ok=True)

    final_mp4 = video_dir / "final_video.mp4"
    final_mp4.write_text("NEW_STYLIZED_2D_VIDEO_CONTENT")

    # Mock video model
    from unittest.mock import MagicMock
    mock_video = MagicMock(spec=Video)
    mock_video.id = video_id
    mock_video.visual_style_id = "stylized_2d_narrative"
    mock_video.visual_style_snapshot = STYLIZED_2D_NARRATIVE_PRESET.model_dump()
    mock_video.output_path = str(final_mp4)

    mock_step = MagicMock(spec=VideoStep)
    mock_step.status = "stale"

    # Old render job was started under stickman
    from app.steps.render import generate_render_step
    # Call generate_render_step with step marked stale; verify PipelineError is raised
    with pytest.raises(Exception):
        generate_render_step(
            video=mock_video,
            prev={
                "images": {"images": {"items": []}},
                "voiceover": {"voiceover": {"audio_file": "audio/narration.wav"}},
            },
            step=mock_step,
        )

    # Newer video file on disk remains completely intact
    assert final_mp4.read_text() == "NEW_STYLIZED_2D_VIDEO_CONTENT"


def test_toctou_race_old_render_fails_promotion_when_style_updated_concurrently(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    """Scenario 1: Old render passes ffmpeg generation; style update occurs; old render cannot promote file."""
    import threading
    import shutil
    from app.steps.render import generate_render_step, PipelineError
    from app.routers.videos import update_video_visual_style, VideoStyleUpdate
    from app.images.mock import create_solid_png
    from app.voiceover.mock import create_mock_wav

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = Video(
        channel_id=test_channel.id,
        topic="TOCTOU Render",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    _populate_mock_completed_pipeline(db_session, video)

    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)
    final_mp4 = video_dir / "final_video.mp4"
    final_mp4.write_text("INITIAL_VIDEO_CONTENT")

    render_step = next(s for s in video.steps if s.name == "render")
    render_step.status = "running"
    db_session.commit()

    # Create dummy images and voiceover audio files
    img_dir = video_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    img1 = img_dir / "scene_01_old12345.png"
    img1.write_bytes(create_solid_png(100, 150, 200, 100, 50))
    img2 = img_dir / "scene_02_old67890.png"
    img2.write_bytes(create_solid_png(100, 150, 200, 100, 50))
    audio_dir = video_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_file = audio_dir / "narration_abc12345.wav"
    audio_file.write_bytes(create_mock_wav(6.0, 24000))

    prev = {
        "storyboard": next(s for s in video.steps if s.name == "storyboard").output,
        "images": next(s for s in video.steps if s.name == "images").output,
        "voiceover": next(s for s in video.steps if s.name == "voiceover").output,
    }

    # Deterministic barrier to trigger style update right before promotion
    from app.db import atomic_promote_file
    real_promote = atomic_promote_file
    at_promotion_barrier = threading.Event()
    style_updated_barrier = threading.Event()

    def hooked_promote(temp_path, final_path, session, video_id, expected_hash, expected_style_id, step_name, *args, **kwargs):
        at_promotion_barrier.set()
        style_updated_barrier.wait(timeout=5.0)
        return real_promote(temp_path, final_path, session, video_id, expected_hash, expected_style_id, step_name, *args, **kwargs)

    monkeypatch.setattr("app.steps.render.atomic_promote_file", hooked_promote)

    # Mock ffmpeg probe and subprocess to simulate fast successful render
    from unittest.mock import MagicMock
    import subprocess

    def mock_subprocess_run(cmd, *args, **kwargs):
        for arg in cmd:
            arg_str = str(arg)
            if arg_str.endswith(".mp4"):
                p = Path(arg_str)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"dummy_mp4_bytes_1234567890")
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)
    monkeypatch.setattr(
        "app.steps.render._probe_video_file",
        MagicMock(return_value={
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p", "width": 1080, "height": 1920, "duration": 6.0},
                {"codec_type": "audio", "codec_name": "aac", "duration": 6.0},
            ],
            "format": {"duration": "6.0", "size": "10000"},
        }),
    )

    render_error = []

    def run_render_worker():
        try:
            # Run render step in thread attached to video
            generate_render_step(video=video, prev=prev, step=render_step)
        except Exception as exc:
            render_error.append(exc)

    t = threading.Thread(target=run_render_worker)
    t.start()

    # Wait until render thread reaches promotion
    reached = at_promotion_barrier.wait(timeout=5.0)
    if not reached:
        t.join(timeout=1.0)
        raise RuntimeError(f"Render worker failed to reach promotion barrier: {render_error}")

    # Perform concurrent visual style update to Stylized 2D
    update_video_visual_style(
        body=VideoStyleUpdate(visual_style_id="stylized_2d_narrative"),
        video=video,
        db=db_session,
    )
    final_mp4.write_text("NEW_STYLIZED_2D_PUBLISHED_VIDEO")
    style_updated_barrier.set()
    t.join(timeout=5.0)

    # Verify old render job aborted with PipelineError and did NOT overwrite final_video.mp4
    assert len(render_error) == 1
    assert isinstance(render_error[0], PipelineError)
    assert "superseded by visual style change" in str(render_error[0])
    assert final_mp4.read_text() == "NEW_STYLIZED_2D_PUBLISHED_VIDEO"


def test_toctou_race_old_subtitles_fails_promotion_when_style_updated_concurrently(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    """Scenario 2: Old subtitle job passes burning; style update occurs; old subtitle cannot promote file."""
    import threading
    import shutil
    from app.steps.subtitles import generate_subtitles_step, PipelineError
    from app.routers.videos import update_video_visual_style, VideoStyleUpdate
    from app.subtitles.mock import MockSubtitleProvider

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = Video(
        channel_id=test_channel.id,
        topic="Sub TOCTOU",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    _populate_mock_completed_pipeline(db_session, video)

    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)
    captioned_mp4 = video_dir / "final_video_captioned.mp4"
    captioned_mp4.write_text("INITIAL_CAPTIONED_CONTENT")

    raw_mp4 = video_dir / "final_video.mp4"
    raw_mp4.write_text("RAW_MP4_CONTENT")

    audio_dir = video_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    from app.voiceover.mock import create_mock_wav
    audio_file = audio_dir / "narration_abc12345.wav"
    audio_file.write_bytes(create_mock_wav(6.0, 24000))

    sub_step = next(s for s in video.steps if s.name in ("subtitles", "subtitle"))
    sub_step.status = "running"
    db_session.commit()

    prev = {
        "script": next(s for s in video.steps if s.name == "script").output,
        "storyboard": next(s for s in video.steps if s.name == "storyboard").output,
        "voiceover": next(s for s in video.steps if s.name == "voiceover").output,
        "render": {"video_file": f"{video.id}/final_video.mp4"},
    }

    from app.db import atomic_promote_file
    real_sub_promote = atomic_promote_file
    at_sub_promotion_barrier = threading.Event()
    style_updated_sub_barrier = threading.Event()

    def hooked_sub_promote(temp_path, final_path, session, video_id, expected_hash, expected_style_id, step_name, *args, **kwargs):
        at_sub_promotion_barrier.set()
        style_updated_sub_barrier.wait(timeout=5.0)
        return real_sub_promote(temp_path, final_path, session, video_id, expected_hash, expected_style_id, step_name, *args, **kwargs)

    monkeypatch.setattr("app.steps.subtitles.atomic_promote_file", hooked_sub_promote)

    # Mock ffmpeg burn command to write temp captioned file
    from unittest.mock import MagicMock
    import subprocess

    def mock_sub_subprocess_run(cmd, *args, **kwargs):
        for arg in cmd:
            arg_str = str(arg)
            if "temp_captioned" in arg_str or arg_str.endswith(".mp4"):
                p = Path(arg_str)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"captioned_mp4_bytes")
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", mock_sub_subprocess_run)

    sub_error = []

    def run_sub_worker():
        try:
            generate_subtitles_step(
                video=video,
                prev=prev,
                step=sub_step,
                subtitle_provider=MockSubtitleProvider(),
            )
        except Exception as exc:
            sub_error.append(exc)

    t = threading.Thread(target=run_sub_worker)
    t.start()

    # Wait until sub thread reaches promotion
    reached = at_sub_promotion_barrier.wait(timeout=5.0)
    if not reached:
        t.join(timeout=1.0)
        raise RuntimeError(f"Sub worker failed to reach promotion barrier: {sub_error}")

    # Concurrent visual style update
    update_video_visual_style(
        body=VideoStyleUpdate(visual_style_id="stylized_2d_narrative"),
        video=video,
        db=db_session,
    )
    captioned_mp4.write_text("NEW_STYLIZED_2D_CAPTIONED_CONTENT")
    style_updated_sub_barrier.set()
    t.join(timeout=5.0)

    # Verify old subtitle job was aborted with PipelineError and did NOT overwrite captioned mp4
    assert len(sub_error) == 1
    assert isinstance(sub_error[0], PipelineError)
    assert "superseded by visual style change" in str(sub_error[0])
    assert captioned_mp4.read_text() == "NEW_STYLIZED_2D_CAPTIONED_CONTENT"


def test_old_image_job_attempts_to_replace_scene_image_after_newer_job_published(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    """Scenario 3: Old image job cannot replace a scene image file or output after newer job published."""
    from app.steps.images import _promote_image_bytes, _persist_partial

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = Video(
        channel_id=test_channel.id,
        topic="Image Race",
        visual_style_id="stylized_2d_narrative",
        visual_style_version=1,
        visual_style_snapshot=STYLIZED_2D_NARRATIVE_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    init_steps(db_session, video)

    img_step = next(s for s in video.steps if s.name == "images")
    img_step.status = "running"
    # Newer job published scene 1 output
    new_hash = "new_stylized_2d_hash"
    img_step.output = {
        "images": {
            "items": [{"index": 1, "status": "ok", "file": f"{video.id}/images/scene_01_{new_hash[:8]}.png"}],
            "complete": False,
        },
        "meta": {"visual_style_snapshot_hash": canonical_snapshot_hash(STYLIZED_2D_NARRATIVE_PRESET.model_dump())},
    }
    db_session.commit()

    scene_img = tmp_path / str(video.id) / "images" / f"scene_01_{new_hash[:8]}.png"
    scene_img.parent.mkdir(parents=True, exist_ok=True)
    scene_img.write_bytes(b"NEW_STYLIZED_2D_IMAGE_BYTES")

    # Old job for Stickman attempts to promote its image bytes
    old_snap_hash = canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump())
    promoted = _promote_image_bytes(
        dest_path=scene_img,
        image_bytes=b"OLD_STICKMAN_IMAGE_BYTES",
        step=img_step,
        initial_snapshot_hash=old_snap_hash,
        initial_style_id="stickman_tactical_explainer",
    )
    assert promoted is False
    assert scene_img.read_bytes() == b"NEW_STYLIZED_2D_IMAGE_BYTES"

    # Old job attempts partial persistence
    _persist_partial(
        step=img_step,
        payload={
            "images": {"items": [{"index": 1, "status": "ok", "file": "old_file.png"}]},
            "meta": {"visual_style_snapshot_hash": old_snap_hash},
        },
        initial_snapshot_hash=old_snap_hash,
        initial_style_id="stickman_tactical_explainer",
    )
    db_session.refresh(img_step)
    # Output must still reflect newer job's items
    items = img_step.output["images"]["items"]
    assert items[0]["file"] == f"{video.id}/images/scene_01_{new_hash[:8]}.png"


def test_current_job_can_still_publish_normally(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    """Scenario 4: Current active job publishes render and subtitles normally without race interference."""
    from app.steps.render import generate_render_step
    from app.steps.subtitles import generate_subtitles_step
    from app.subtitles.mock import MockSubtitleProvider

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = Video(
        channel_id=test_channel.id,
        topic="Normal Publication",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    _populate_mock_completed_pipeline(db_session, video)

    from app.images.mock import create_solid_png
    from app.voiceover.mock import create_mock_wav

    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)

    img_dir = video_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    (img_dir / "scene_01_old12345.png").write_bytes(create_solid_png(100, 150, 200, 100, 50))
    (img_dir / "scene_02_old67890.png").write_bytes(create_solid_png(100, 150, 200, 100, 50))

    audio_dir = video_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    (audio_dir / "narration_abc12345.wav").write_bytes(create_mock_wav(6.0, 24000))

    from unittest.mock import MagicMock
    import subprocess

    def mock_normal_subprocess_run(cmd, *args, **kwargs):
        for arg in cmd:
            arg_str = str(arg)
            if "temp_captioned" in arg_str or arg_str.endswith(".mp4"):
                p = Path(arg_str)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"dummy_mp4_content_1234567890")
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", mock_normal_subprocess_run)
    monkeypatch.setattr(
        "app.steps.render._probe_video_file",
        MagicMock(return_value={
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p", "width": 1080, "height": 1920, "duration": 6.0},
                {"codec_type": "audio", "codec_name": "aac", "duration": 6.0},
            ],
            "format": {"duration": "6.0", "size": "10000"},
        }),
    )

    render_step = next(s for s in video.steps if s.name == "render")
    render_step.status = "running"
    db_session.commit()

    prev_render = {
        "storyboard": next(s for s in video.steps if s.name == "storyboard").output,
        "images": next(s for s in video.steps if s.name == "images").output,
        "voiceover": next(s for s in video.steps if s.name == "voiceover").output,
    }

    render_res = generate_render_step(video=video, prev=prev_render, step=render_step)
    assert render_res is not None
    db_session.refresh(video)
    assert Path(video.output_path).is_file()
    assert (video_dir / "final_video.mp4").is_file()

    # Subtitles normal publication
    sub_step = next(s for s in video.steps if s.name in ("subtitles", "subtitle"))
    sub_step.status = "running"
    db_session.commit()

    prev_sub = {
        "script": next(s for s in video.steps if s.name == "script").output,
        "storyboard": next(s for s in video.steps if s.name == "storyboard").output,
        "voiceover": next(s for s in video.steps if s.name == "voiceover").output,
        "render": {"video_file": f"{video.id}/final_video.mp4"},
    }

    sub_res = generate_subtitles_step(video=video, prev=prev_sub, step=sub_step, subtitle_provider=MockSubtitleProvider())
    assert sub_res is not None
    assert (video_dir / "final_video_captioned.mp4").is_file()


def test_style_update_and_publication_cannot_leave_db_and_filesystem_inconsistent(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    """Scenario 5: Style update and publication cannot leave database and filesystem pointing to different generations."""
    from app.routers.videos import update_video_visual_style, VideoStyleUpdate

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = Video(
        channel_id=test_channel.id,
        topic="Consistency Check",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    _populate_mock_completed_pipeline(db_session, video)

    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)
    final_mp4 = video_dir / "final_video.mp4"
    final_mp4.write_text("STICKMAN_MEDIA")
    video.output_path = str(final_mp4)
    db_session.commit()

    # Video is currently Stickman with published path pointing to STICKMAN_MEDIA
    assert video.visual_style_id == "stickman_tactical_explainer"
    assert video.output_path == str(final_mp4)

    # Style updated to Stylized 2D
    update_video_visual_style(
        body=VideoStyleUpdate(visual_style_id="stylized_2d_narrative"),
        video=video,
        db=db_session,
    )
    db_session.refresh(video)

    # Immediately after style update, output_path must be cleared, not pointing to old generation media
    assert video.visual_style_id == "stylized_2d_narrative"
    assert video.output_path is None

    # Downstream visual steps must be stale
    for s_name in ("images", "render", "subtitles"):
        s = next(step for step in video.steps if step.name == s_name)
        assert s.status == "stale"
        assert s.output is None


def test_concurrent_competing_style_update_and_job_completion_no_deadlock(tmp_path: Path, monkeypatch):
    """Area 1: Concurrent style update and complete_job adhere to strict Video -> PipelineJob hierarchy without deadlock."""
    import threading
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.routers.videos import update_video_visual_style, VideoStyleUpdate
    from app.queue.engine import complete_job, _now

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    stickman_hash = canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump())

    file_db = tmp_path / "test_concurrency.db"
    file_engine = create_engine(
        f"sqlite:///{file_db}?timeout=30",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(file_engine)
    SessionLocalTest = sessionmaker(bind=file_engine, expire_on_commit=False)

    # Setup channel, video and job in isolated file DB
    with SessionLocalTest() as init_db:
        user = User(email="conc_user@example.com", password_hash="hash", full_name="Conc User")
        init_db.add(user)
        init_db.commit()
        init_db.refresh(user)

        channel = Channel(user_id=user.id, name="Conc Channel", handle="@conc", default_visual_style_id="legacy_default")
        init_db.add(channel)
        init_db.commit()
        init_db.refresh(channel)

        video = Video(
            channel_id=channel.id,
            topic="Deadlock Test",
            visual_style_id="stickman_tactical_explainer",
            visual_style_version=1,
            visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
        )
        init_db.add(video)
        init_db.commit()
        _populate_mock_completed_pipeline(init_db, video)

        render_step = next(s for s in video.steps if s.name == "render")
        render_step.status = "running"
        job = PipelineJob(
            video_id=video.id,
            step_name="render",
            status="running",
            lease_token=uuid.uuid4(),
            worker_id="worker_test_1",
            lease_expires_at=_now() + timedelta(minutes=5),
            scheduled_at=_now(),
        )
        init_db.add(job)
        init_db.commit()
        vid_id = video.id
        job_id = job.id
        lease_token = job.lease_token

    results = {}
    errors = []

    def run_style_update():
        try:
            with SessionLocalTest() as db_a:
                vid = db_a.get(Video, vid_id)
                update_video_visual_style(
                    body=VideoStyleUpdate(visual_style_id="stylized_2d_narrative"),
                    video=vid,
                    db=db_a,
                )
                results["style_update"] = "success"
        except Exception as exc:
            errors.append(f"style_update failed: {exc}")

    def run_complete_job():
        try:
            with SessionLocalTest() as db_b:
                res = complete_job(
                    db=db_b,
                    job_id=job_id,
                    lease_token=lease_token,
                    output={
                        "render": {"video_path": "some_path.mp4"},
                        "meta": {
                            "visual_style_id": "stickman_tactical_explainer",
                            "visual_style_snapshot_hash": stickman_hash,
                        },
                    },
                )
                results["complete_job"] = res
        except Exception as exc:
            errors.append(f"complete_job failed: {exc}")

    t1 = threading.Thread(target=run_style_update)
    t2 = threading.Thread(target=run_complete_job)

    t1.start()
    t2.start()

    t1.join(timeout=5)
    t2.join(timeout=5)

    assert not t1.is_alive(), "Thread 1 deadlocked!"
    assert not t2.is_alive(), "Thread 2 deadlocked!"
    assert len(errors) == 0, f"Errors occurred during concurrent execution: {errors}"

    # Verify final state consistency
    with SessionLocalTest() as check_db:
        final_vid = check_db.get(Video, vid_id)
        assert final_vid.visual_style_id == "stylized_2d_narrative"
        r_step = next(s for s in final_vid.steps if s.name == "render")
        assert r_step.status == "stale"


def test_filesystem_promotion_db_commit_failure_preserves_previous_media(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    """Area 2: Filesystem promotion followed by DB commit failure leaves previous valid output intact."""
    from app.db import atomic_promote_file
    from app.routers.videos import download

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    stickman_hash = canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump())

    video = Video(
        channel_id=test_channel.id,
        topic="Commit Failure Test",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    _populate_mock_completed_pipeline(db_session, video)

    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)
    gen_dir = video_dir / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)

    # 1. Establish previous valid output
    prev_gen_path = gen_dir / "render_prev_12345678.mp4"
    prev_gen_path.write_bytes(b"PREVIOUS_VALID_MEDIA_CONTENT")
    final_canon = video_dir / "final_video.mp4"
    final_canon.write_bytes(b"PREVIOUS_VALID_MEDIA_CONTENT")

    video.output_path = str(prev_gen_path)
    render_step = next(s for s in video.steps if s.name == "render")
    render_step.status = "done"
    db_session.commit()

    # 2. Stage a new render in temp file
    temp_render = video_dir / "temp_render_new.mp4"
    temp_render.write_bytes(b"NEW_UNCOMMITTED_MEDIA_CONTENT")

    # 3. Simulate DB commit failure during atomic_promote_file
    original_commit = db_session.commit

    def failing_commit():
        raise RuntimeError("Simulated Database Commit Failure (e.g. disk full / integrity violation)")

    monkeypatch.setattr(db_session, "commit", failing_commit)

    # Promotion must fail and raise PipelineError
    from app.steps.render import PipelineError
    with pytest.raises(PipelineError) as exc_info:
        atomic_promote_file(
            temp_path=temp_render,
            final_path=final_canon,
            session=db_session,
            video_id=video.id,
            expected_hash=stickman_hash,
            expected_style_id="stickman_tactical_explainer",
            step_name="render",
        )
    assert "Database commit failed" in str(exc_info.value)

    # Restore commit
    monkeypatch.setattr(db_session, "commit", original_commit)
    db_session.refresh(video)

    # 4. Verify previous output remains intact on disk and in DB
    assert video.output_path == str(prev_gen_path)
    assert prev_gen_path.is_file()
    assert prev_gen_path.read_bytes() == b"PREVIOUS_VALID_MEDIA_CONTENT"
    assert final_canon.read_bytes() == b"PREVIOUS_VALID_MEDIA_CONTENT"

    # Verify no uncommitted generation files were left behind
    all_gen_files = list(gen_dir.glob("*.mp4"))
    assert all_gen_files == [prev_gen_path]

    # Verify download endpoint serves the previous valid media
    resp = download(video=video, captioned=False)
    assert resp.path == str(prev_gen_path)
    assert Path(resp.path).read_bytes() == b"PREVIOUS_VALID_MEDIA_CONTENT"


def test_image_promotion_ownership_older_execution_blocked(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    """Area 3A: Older execution with same style and cache key cannot promote after newer execution publishes."""
    from app.steps.images import _promote_image_bytes, _persist_partial

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    stickman_hash = canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump())

    video = Video(
        channel_id=test_channel.id,
        topic="Image Ownership Test",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    _populate_mock_completed_pipeline(db_session, video)

    img_step = next(s for s in video.steps if s.name == "images")
    img_step.status = "running"
    # Newer execution "exec_new" already published initial output
    img_step.output = {
        "images": {"items": [], "complete": False},
        "meta": {
            "visual_style_id": "stickman_tactical_explainer",
            "visual_style_snapshot_hash": stickman_hash,
            "execution_id": "exec_new",
        },
    }
    db_session.commit()

    dest_img = tmp_path / str(video.id) / "images" / "scene_01_abc12345.png"
    dest_img.parent.mkdir(parents=True, exist_ok=True)
    dest_img.write_bytes(b"NEW_EXECUTION_IMAGE_BYTES")

    # Older execution "exec_old" attempts to promote image bytes
    promoted = _promote_image_bytes(
        dest_path=dest_img,
        image_bytes=b"OLD_EXECUTION_IMAGE_BYTES",
        step=img_step,
        initial_snapshot_hash=stickman_hash,
        initial_style_id="stickman_tactical_explainer",
        execution_id="exec_old",
    )
    assert promoted is False

    # The existing image on disk must NOT have been overwritten
    assert dest_img.read_bytes() == b"NEW_EXECUTION_IMAGE_BYTES"

    # Older execution attempts to persist partial progress
    _persist_partial(
        step=img_step,
        payload={
            "images": {"items": [{"index": 1, "status": "ok"}], "complete": False},
            "meta": {
                "visual_style_id": "stickman_tactical_explainer",
                "visual_style_snapshot_hash": stickman_hash,
                "execution_id": "exec_old",
            },
        },
        initial_snapshot_hash=stickman_hash,
        initial_style_id="stickman_tactical_explainer",
        execution_id="exec_old",
    )
    db_session.refresh(img_step)

    # DB step output must NOT be overwritten by exec_old
    assert img_step.output["meta"]["execution_id"] == "exec_new"
    assert img_step.output["images"]["items"] == []


def test_image_promotion_ownership_stale_execution_preserves_current_image(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    """Area 3B: Stale execution does not delete or replace current execution's image."""
    from app.steps.images import _promote_image_bytes

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    stickman_hash = canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump())

    video = Video(
        channel_id=test_channel.id,
        topic="Stale Check",
        visual_style_id="stylized_2d_narrative",
        visual_style_version=1,
        visual_style_snapshot=STYLIZED_2D_NARRATIVE_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    _populate_mock_completed_pipeline(db_session, video)

    img_step = next(s for s in video.steps if s.name == "images")
    img_step.status = "stale"
    db_session.commit()

    dest_img = tmp_path / str(video.id) / "images" / "scene_01_xyz99999.png"
    dest_img.parent.mkdir(parents=True, exist_ok=True)
    dest_img.write_bytes(b"EXISTING_VALID_IMAGE")

    # Stale execution attempts to promote
    promoted = _promote_image_bytes(
        dest_path=dest_img,
        image_bytes=b"STALE_JOB_IMAGE",
        step=img_step,
        initial_snapshot_hash=stickman_hash,
        initial_style_id="stickman_tactical_explainer",
        execution_id="exec_stale",
    )
    assert promoted is False
    assert dest_img.read_bytes() == b"EXISTING_VALID_IMAGE"


def test_image_promotion_ownership_valid_current_execution_succeeds(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    """Area 3C: Valid current execution promotes image bytes and persists partial progress successfully."""
    from app.steps.images import _promote_image_bytes, _persist_partial

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    stickman_hash = canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump())

    video = Video(
        channel_id=test_channel.id,
        topic="Valid Current Execution",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    _populate_mock_completed_pipeline(db_session, video)

    img_step = next(s for s in video.steps if s.name == "images")
    img_step.status = "running"
    img_step.output = {
        "images": {"items": [], "complete": False},
        "meta": {
            "visual_style_id": "stickman_tactical_explainer",
            "visual_style_snapshot_hash": stickman_hash,
            "execution_id": "exec_current",
        },
    }
    db_session.commit()

    dest_img = tmp_path / str(video.id) / "images" / "scene_01_current.png"

    # Current execution promotes image
    promoted = _promote_image_bytes(
        dest_path=dest_img,
        image_bytes=b"CURRENT_VALID_IMAGE_BYTES",
        step=img_step,
        initial_snapshot_hash=stickman_hash,
        initial_style_id="stickman_tactical_explainer",
        execution_id="exec_current",
    )
    assert promoted is True
    assert dest_img.read_bytes() == b"CURRENT_VALID_IMAGE_BYTES"

    # Current execution persists partial progress
    _persist_partial(
        step=img_step,
        payload={
            "images": {"items": [{"index": 1, "status": "ok"}], "complete": False},
            "meta": {
                "visual_style_id": "stickman_tactical_explainer",
                "visual_style_snapshot_hash": stickman_hash,
                "execution_id": "exec_current",
            },
        },
        initial_snapshot_hash=stickman_hash,
        initial_style_id="stickman_tactical_explainer",
        execution_id="exec_current",
    )
    db_session.refresh(img_step)
    assert img_step.output["meta"]["execution_id"] == "exec_current"
    assert len(img_step.output["images"]["items"]) == 1


def test_post_commit_old_render_cannot_overwrite_newer_canonical_file(tmp_path: Path, monkeypatch):
    """Scenario 1: Old render paused after DB commit cannot overwrite newer canonical file when resumed."""
    import threading
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db import atomic_promote_file
    from app.routers.videos import update_video_visual_style, VideoStyleUpdate

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    stickman_hash = canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump())
    stylized_hash = canonical_snapshot_hash(STYLIZED_2D_NARRATIVE_PRESET.model_dump())

    file_db = tmp_path / "test_post_commit_render.db"
    file_engine = create_engine(
        f"sqlite:///{file_db}?timeout=30",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(file_engine)
    SessionLocalTest = sessionmaker(bind=file_engine, expire_on_commit=False)

    with SessionLocalTest() as init_db:
        user = User(email="render_race_user@example.com", password_hash="hash", full_name="Render Race User")
        init_db.add(user)
        init_db.commit()
        init_db.refresh(user)

        channel = Channel(user_id=user.id, name="Render Channel", handle="@render_race", default_visual_style_id="legacy_default")
        init_db.add(channel)
        init_db.commit()
        init_db.refresh(channel)

        video = Video(
            channel_id=channel.id,
            topic="Render Race Test",
            visual_style_id="stickman_tactical_explainer",
            visual_style_version=1,
            visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
        )
        init_db.add(video)
        init_db.commit()
        _populate_mock_completed_pipeline(init_db, video)
        vid_id = video.id

    video_dir = tmp_path / str(vid_id)
    video_dir.mkdir(parents=True, exist_ok=True)
    final_canon = video_dir / "final_video.mp4"

    old_temp = video_dir / "temp_old_render.mp4"
    old_temp.write_bytes(b"OLD_RENDER_MEDIA_CONTENT")

    event_old_committed = threading.Event()
    event_resume_old = threading.Event()

    def old_post_commit_hook():
        event_old_committed.set()
        event_resume_old.wait(timeout=5)

    def run_old_render():
        with SessionLocalTest() as old_session:
            atomic_promote_file(
                temp_path=old_temp,
                final_path=final_canon,
                session=old_session,
                video_id=vid_id,
                expected_hash=stickman_hash,
                expected_style_id="stickman_tactical_explainer",
                step_name="render",
                _post_commit_hook=old_post_commit_hook,
            )

    t_old = threading.Thread(target=run_old_render)
    t_old.start()

    # Wait for old render to complete its DB commit and pause before replacing final_video.mp4
    assert event_old_committed.wait(timeout=5), "Old render did not reach post-commit hook"

    # Style change occurs and invalidates the old render
    with SessionLocalTest() as style_db:
        v = style_db.get(Video, vid_id)
        update_video_visual_style(
            body=VideoStyleUpdate(visual_style_id="stylized_2d_narrative"),
            video=v,
            db=style_db,
        )

    # New render runs and publishes its canonical file
    new_temp = video_dir / "temp_new_render.mp4"
    new_temp.write_bytes(b"NEW_RENDER_MEDIA_CONTENT")
    with SessionLocalTest() as new_session:
        v_new = new_session.get(Video, vid_id)
        r_step = next(s for s in v_new.steps if s.name == "render")
        r_step.status = "running"
        new_session.commit()
        atomic_promote_file(
            temp_path=new_temp,
            final_path=final_canon,
            session=new_session,
            video_id=vid_id,
            expected_hash=stylized_hash,
            expected_style_id="stylized_2d_narrative",
            step_name="render",
        )

    assert final_canon.read_bytes() == b"NEW_RENDER_MEDIA_CONTENT"

    # Now resume the old render
    event_resume_old.set()
    t_old.join(timeout=5)
    assert not t_old.is_alive()

    # The canonical file MUST NOT have been overwritten by the old render!
    assert final_canon.read_bytes() == b"NEW_RENDER_MEDIA_CONTENT"


def test_post_commit_old_subtitles_cannot_overwrite_newer_canonical_file(tmp_path: Path, monkeypatch):
    """Scenario 2: Old subtitles paused after DB commit cannot overwrite newer canonical file when resumed."""
    import threading
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db import atomic_promote_file
    from app.routers.videos import update_video_visual_style, VideoStyleUpdate

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    stickman_hash = canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump())
    stylized_hash = canonical_snapshot_hash(STYLIZED_2D_NARRATIVE_PRESET.model_dump())

    file_db = tmp_path / "test_post_commit_subs.db"
    file_engine = create_engine(
        f"sqlite:///{file_db}?timeout=30",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(file_engine)
    SessionLocalTest = sessionmaker(bind=file_engine, expire_on_commit=False)

    with SessionLocalTest() as init_db:
        user = User(email="sub_race_user@example.com", password_hash="hash", full_name="Sub Race User")
        init_db.add(user)
        init_db.commit()
        init_db.refresh(user)

        channel = Channel(user_id=user.id, name="Sub Channel", handle="@sub_race", default_visual_style_id="legacy_default")
        init_db.add(channel)
        init_db.commit()
        init_db.refresh(channel)

        video = Video(
            channel_id=channel.id,
            topic="Subtitle Race Test",
            visual_style_id="stickman_tactical_explainer",
            visual_style_version=1,
            visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
        )
        init_db.add(video)
        init_db.commit()
        _populate_mock_completed_pipeline(init_db, video)
        vid_id = video.id

    video_dir = tmp_path / str(vid_id)
    video_dir.mkdir(parents=True, exist_ok=True)
    captioned_canon = video_dir / "final_video_captioned.mp4"

    old_sub_temp = video_dir / "temp_old_subtitles.mp4"
    old_sub_temp.write_bytes(b"OLD_CAPTIONED_MEDIA_CONTENT")

    event_sub_committed = threading.Event()
    event_resume_sub = threading.Event()

    def old_sub_post_commit_hook():
        event_sub_committed.set()
        event_resume_sub.wait(timeout=5)

    def run_old_subtitles():
        with SessionLocalTest() as old_session:
            atomic_promote_file(
                temp_path=old_sub_temp,
                final_path=captioned_canon,
                session=old_session,
                video_id=vid_id,
                expected_hash=stickman_hash,
                expected_style_id="stickman_tactical_explainer",
                step_name="subtitles",
                _post_commit_hook=old_sub_post_commit_hook,
            )

    t_sub = threading.Thread(target=run_old_subtitles)
    t_sub.start()

    assert event_sub_committed.wait(timeout=5), "Old subtitles did not reach post-commit hook"

    # Style change occurs and invalidates subtitles step
    with SessionLocalTest() as style_db:
        v = style_db.get(Video, vid_id)
        update_video_visual_style(
            body=VideoStyleUpdate(visual_style_id="stylized_2d_narrative"),
            video=v,
            db=style_db,
        )

    # New subtitles runs and publishes its canonical file
    new_sub_temp = video_dir / "temp_new_subtitles.mp4"
    new_sub_temp.write_bytes(b"NEW_CAPTIONED_MEDIA_CONTENT")
    with SessionLocalTest() as new_session:
        v_new = new_session.get(Video, vid_id)
        s_step = next(s for s in v_new.steps if s.name in ("subtitles", "subtitle"))
        s_step.status = "running"
        v_new.output_path = str(video_dir / "final_video.mp4")
        new_session.commit()
        atomic_promote_file(
            temp_path=new_sub_temp,
            final_path=captioned_canon,
            session=new_session,
            video_id=vid_id,
            expected_hash=stylized_hash,
            expected_style_id="stylized_2d_narrative",
            step_name="subtitles",
        )

    assert captioned_canon.read_bytes() == b"NEW_CAPTIONED_MEDIA_CONTENT"

    # Resume the old subtitles
    event_resume_sub.set()
    t_sub.join(timeout=5)
    assert not t_sub.is_alive()

    # The canonical captioned file MUST NOT have been overwritten by the old subtitles!
    assert captioned_canon.read_bytes() == b"NEW_CAPTIONED_MEDIA_CONTENT"


def test_canonical_file_fallback_blocked_after_style_invalidation(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    """Scenario 3: Download and subtitle burn never fall back to orphan canonical files after style invalidation."""
    from fastapi import HTTPException
    from app.routers.videos import update_video_visual_style, VideoStyleUpdate, download
    from app.steps.subtitles import generate_subtitles_step
    from app.subtitles.mock import MockSubtitleProvider

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = Video(
        channel_id=test_channel.id,
        topic="Fallback Guard Test",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    _populate_mock_completed_pipeline(db_session, video)

    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)
    raw_canon = video_dir / "final_video.mp4"
    raw_canon.write_bytes(b"ORPHAN_RAW_CONTENT")
    cap_canon = video_dir / "final_video_captioned.mp4"
    cap_canon.write_bytes(b"ORPHAN_CAPTIONED_CONTENT")

    video.output_path = str(raw_canon)
    db_session.commit()

    # Style change invalidates visual steps and clears video.output_path
    update_video_visual_style(
        body=VideoStyleUpdate(visual_style_id="stylized_2d_narrative"),
        video=video,
        db=db_session,
    )
    db_session.refresh(video)
    assert video.output_path is None

    # Download original must refuse to serve orphan final_video.mp4 (must raise 409)
    with pytest.raises(HTTPException) as exc_raw:
        download(video=video, captioned=False)
    assert exc_raw.value.status_code in (404, 409)

    # Download captioned must refuse to serve orphan final_video_captioned.mp4 (must raise 409)
    with pytest.raises(HTTPException) as exc_cap:
        download(video=video, captioned=True)
    assert exc_cap.value.status_code in (404, 409)

    # Subtitles step execution without valid render output_path must fail, not silently use final_video.mp4
    from app.steps.subtitles import PipelineError
    sub_step = next(s for s in video.steps if s.name in ("subtitles", "subtitle"))
    with pytest.raises(PipelineError) as exc_sub:
        generate_subtitles_step(
            video=video,
            prev={
                "storyboard": next(s for s in video.steps if s.name == "storyboard").output,
                "voiceover": next(s for s in video.steps if s.name == "voiceover").output,
            },
            step=sub_step,
            subtitle_provider=MockSubtitleProvider(),
        )
    assert "render step must be completed" in str(exc_sub.value).lower()


def test_newer_image_execution_claims_ownership_before_first_image(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    """Scenario 4: Newer execution claims ownership durably before publishing its first image, blocking older execution."""
    from app.steps.images import _claim_step_ownership, _promote_image_bytes, _persist_partial, _build_output_payload
    from app.images.mock import MockImageProvider

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    stickman_hash = canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump())

    video = Video(
        channel_id=test_channel.id,
        topic="Image Early Claim Test",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    _populate_mock_completed_pipeline(db_session, video)

    img_step = next(s for s in video.steps if s.name == "images")
    img_step.status = "running"
    db_session.commit()

    # 1. Execution 1 establishes early ownership
    exec1_payload = _build_output_payload(
        items=[],
        warnings=[],
        provider=MockImageProvider(),
        total_scenes=2,
        is_complete=False,
        visual_style_id="stickman_tactical_explainer",
        visual_style_snapshot_hash=stickman_hash,
        execution_id="exec_001",
    )
    claimed_1 = _claim_step_ownership(
        step=img_step,
        payload=exec1_payload,
        initial_snapshot_hash=stickman_hash,
        initial_style_id="stickman_tactical_explainer",
        execution_id="exec_001",
    )
    assert claimed_1 is True
    assert img_step.output["meta"]["execution_id"] == "exec_001"

    # 2. Execution 2 starts (same style, same key) and claims ownership BEFORE publishing any image
    exec2_payload = _build_output_payload(
        items=[],
        warnings=[],
        provider=MockImageProvider(),
        total_scenes=2,
        is_complete=False,
        visual_style_id="stickman_tactical_explainer",
        visual_style_snapshot_hash=stickman_hash,
        execution_id="exec_002",
    )
    claimed_2 = _claim_step_ownership(
        step=img_step,
        payload=exec2_payload,
        initial_snapshot_hash=stickman_hash,
        initial_style_id="stickman_tactical_explainer",
        execution_id="exec_002",
    )
    assert claimed_2 is True
    db_session.refresh(img_step)
    assert img_step.output["meta"]["execution_id"] == "exec_002"
    assert img_step.output["images"]["items"] == []

    # 3. Execution 1 finishes its first image and attempts to promote and persist
    dest_img = tmp_path / str(video.id) / "images" / "scene_01_claimtest.png"
    promoted_1 = _promote_image_bytes(
        dest_path=dest_img,
        image_bytes=b"EXEC1_IMAGE_DATA",
        step=img_step,
        initial_snapshot_hash=stickman_hash,
        initial_style_id="stickman_tactical_explainer",
        execution_id="exec_001",
    )
    # Execution 1 must be rejected
    assert promoted_1 is False
    assert not dest_img.exists()

    _persist_partial(
        step=img_step,
        payload={
            "images": {"items": [{"index": 1, "status": "ok"}], "complete": False},
            "meta": {
                "visual_style_id": "stickman_tactical_explainer",
                "visual_style_snapshot_hash": stickman_hash,
                "execution_id": "exec_001",
            },
        },
        initial_snapshot_hash=stickman_hash,
        initial_style_id="stickman_tactical_explainer",
        execution_id="exec_001",
    )
    db_session.refresh(img_step)
    # Execution 1 must NOT have overwritten step.output
    assert img_step.output["meta"]["execution_id"] == "exec_002"
    assert img_step.output["images"]["items"] == []

    # 4. Execution 2 promotes its image successfully
    promoted_2 = _promote_image_bytes(
        dest_path=dest_img,
        image_bytes=b"EXEC2_IMAGE_DATA",
        step=img_step,
        initial_snapshot_hash=stickman_hash,
        initial_style_id="stickman_tactical_explainer",
        execution_id="exec_002",
    )
    assert promoted_2 is True
    assert dest_img.read_bytes() == b"EXEC2_IMAGE_DATA"

    _persist_partial(
        step=img_step,
        payload={
            "images": {"items": [{"index": 1, "status": "ok"}], "complete": False},
            "meta": {
                "visual_style_id": "stickman_tactical_explainer",
                "visual_style_snapshot_hash": stickman_hash,
                "execution_id": "exec_002",
            },
        },
        initial_snapshot_hash=stickman_hash,
        initial_style_id="stickman_tactical_explainer",
        execution_id="exec_002",
    )
    db_session.refresh(img_step)
    assert img_step.output["meta"]["execution_id"] == "exec_002"
    assert len(img_step.output["images"]["items"]) == 1


def test_same_style_same_snapshot_render_race_older_job_preserves_newer_canonical(tmp_path: Path, monkeypatch):
    """Scenario 5: Two render jobs with the same visual style and snapshot.
    An older render paused post-commit cannot overwrite a newer render's canonical file.
    """
    import threading
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db import atomic_promote_file

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    stickman_hash = canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump())

    file_db = tmp_path / "test_same_style_render_race.db"
    file_engine = create_engine(
        f"sqlite:///{file_db}?timeout=30",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(file_engine)
    SessionLocalTest = sessionmaker(bind=file_engine, expire_on_commit=False)

    with SessionLocalTest() as init_db:
        user = User(email="render_race_samesnap@example.com", password_hash="hash", full_name="Render Race Same Snap")
        init_db.add(user)
        init_db.commit()

        channel = Channel(user_id=user.id, name="Render Race Channel", handle="@render_race_snap", default_visual_style_id="legacy_default")
        init_db.add(channel)
        init_db.commit()

        video = Video(
            channel_id=channel.id,
            topic="Same Snap Render Race",
            visual_style_id="stickman_tactical_explainer",
            visual_style_version=1,
            visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
        )
        init_db.add(video)
        init_db.commit()
        _populate_mock_completed_pipeline(init_db, video)
        vid_id = video.id

    video_dir = tmp_path / str(vid_id)
    video_dir.mkdir(parents=True, exist_ok=True)
    final_canon = video_dir / "final_video.mp4"

    old_temp = video_dir / "temp_old_render.mp4"
    old_temp.write_bytes(b"OLD_RENDER_GEN1_CONTENT")

    event_old_committed = threading.Event()
    event_resume_old = threading.Event()
    old_promoted_result: list[str] = []

    def old_post_commit_hook():
        event_old_committed.set()
        event_resume_old.wait(timeout=5)

    def run_old_render():
        with SessionLocalTest() as old_session:
            promoted = atomic_promote_file(
                temp_path=old_temp,
                final_path=final_canon,
                session=old_session,
                video_id=vid_id,
                expected_hash=stickman_hash,
                expected_style_id="stickman_tactical_explainer",
                expected_execution_id="render_exec_001",
                step_name="render",
                _post_commit_hook=old_post_commit_hook,
            )
            old_promoted_result.append(promoted)

    t_old = threading.Thread(target=run_old_render)
    t_old.start()

    assert event_old_committed.wait(timeout=5), "Old render did not reach post-commit hook"

    # A newer render with the SAME visual style and snapshot runs and completes
    new_temp = video_dir / "temp_new_render.mp4"
    new_temp.write_bytes(b"NEW_RENDER_GEN2_CONTENT")
    new_promoted_result: list[str] = []
    with SessionLocalTest() as new_session:
        promoted = atomic_promote_file(
            temp_path=new_temp,
            final_path=final_canon,
            session=new_session,
            video_id=vid_id,
            expected_hash=stickman_hash,
            expected_style_id="stickman_tactical_explainer",
            expected_execution_id="render_exec_002",
            step_name="render",
        )
        new_promoted_result.append(promoted)

    assert final_canon.read_bytes() == b"NEW_RENDER_GEN2_CONTENT"
    assert new_promoted_result[0] == str(final_canon)

    # Resume the old render
    event_resume_old.set()
    t_old.join(timeout=5)
    assert not t_old.is_alive()

    # The canonical file MUST NOT have been overwritten by the older render!
    assert final_canon.read_bytes() == b"NEW_RENDER_GEN2_CONTENT"
    # Old render returned its immutable generation file, leaving canonical convenience file untouched
    assert old_promoted_result and old_promoted_result[0] != str(final_canon)
    assert "generations" in old_promoted_result[0]
    assert Path(old_promoted_result[0]).read_bytes() == b"OLD_RENDER_GEN1_CONTENT"


def test_same_style_same_snapshot_subtitles_race_older_job_preserves_newer_canonical(tmp_path: Path, monkeypatch):
    """Scenario 6: Two subtitle jobs with the same visual style and snapshot.
    An older subtitles paused post-commit cannot overwrite a newer subtitles canonical file.
    """
    import threading
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db import atomic_promote_file

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    stickman_hash = canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump())

    file_db = tmp_path / "test_same_style_sub_race.db"
    file_engine = create_engine(
        f"sqlite:///{file_db}?timeout=30",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(file_engine)
    SessionLocalTest = sessionmaker(bind=file_engine, expire_on_commit=False)

    with SessionLocalTest() as init_db:
        user = User(email="sub_race_samesnap@example.com", password_hash="hash", full_name="Sub Race Same Snap")
        init_db.add(user)
        init_db.commit()

        channel = Channel(user_id=user.id, name="Sub Race Channel", handle="@sub_race_snap", default_visual_style_id="legacy_default")
        init_db.add(channel)
        init_db.commit()

        video = Video(
            channel_id=channel.id,
            topic="Same Snap Sub Race",
            visual_style_id="stickman_tactical_explainer",
            visual_style_version=1,
            visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
        )
        init_db.add(video)
        init_db.commit()
        _populate_mock_completed_pipeline(init_db, video)
        vid_id = video.id

    video_dir = tmp_path / str(vid_id)
    video_dir.mkdir(parents=True, exist_ok=True)
    captioned_canon = video_dir / "final_video_captioned.mp4"

    old_temp = video_dir / "temp_old_subtitles.mp4"
    old_temp.write_bytes(b"OLD_SUB_GEN1_CONTENT")

    event_old_committed = threading.Event()
    event_resume_old = threading.Event()
    old_promoted_result: list[str] = []

    def old_post_commit_hook():
        event_old_committed.set()
        event_resume_old.wait(timeout=5)

    def run_old_subtitles():
        with SessionLocalTest() as old_session:
            promoted = atomic_promote_file(
                temp_path=old_temp,
                final_path=captioned_canon,
                session=old_session,
                video_id=vid_id,
                expected_hash=stickman_hash,
                expected_style_id="stickman_tactical_explainer",
                expected_execution_id="sub_exec_001",
                step_name="subtitles",
                _post_commit_hook=old_post_commit_hook,
            )
            old_promoted_result.append(promoted)

    t_old = threading.Thread(target=run_old_subtitles)
    t_old.start()

    assert event_old_committed.wait(timeout=5), "Old subtitles did not reach post-commit hook"

    # A newer subtitles with the SAME visual style and snapshot runs and completes
    new_temp = video_dir / "temp_new_subtitles.mp4"
    new_temp.write_bytes(b"NEW_SUB_GEN2_CONTENT")
    new_promoted_result: list[str] = []
    with SessionLocalTest() as new_session:
        promoted = atomic_promote_file(
            temp_path=new_temp,
            final_path=captioned_canon,
            session=new_session,
            video_id=vid_id,
            expected_hash=stickman_hash,
            expected_style_id="stickman_tactical_explainer",
            expected_execution_id="sub_exec_002",
            step_name="subtitles",
        )
        new_promoted_result.append(promoted)

    assert captioned_canon.read_bytes() == b"NEW_SUB_GEN2_CONTENT"
    assert new_promoted_result[0] == str(captioned_canon)

    # Resume the old subtitles
    event_resume_old.set()
    t_old.join(timeout=5)
    assert not t_old.is_alive()

    # The canonical captioned file MUST NOT have been overwritten by the older subtitles!
    assert captioned_canon.read_bytes() == b"NEW_SUB_GEN2_CONTENT"
    # Old subtitles returned its immutable generation file, leaving canonical convenience file untouched
    assert old_promoted_result and old_promoted_result[0] != str(captioned_canon)
    assert "generations" in old_promoted_result[0]
    assert Path(old_promoted_result[0]).read_bytes() == b"OLD_SUB_GEN1_CONTENT"


def test_normal_publication_updates_canonical_files(tmp_path: Path, monkeypatch):
    """Scenario 7: Normal publication ensuring current authoritative render and subtitles can update canonical files."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.db import atomic_promote_file

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    stickman_hash = canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump())

    file_db = tmp_path / "test_normal_pub.db"
    file_engine = create_engine(
        f"sqlite:///{file_db}?timeout=30",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(file_engine)
    SessionLocalTest = sessionmaker(bind=file_engine, expire_on_commit=False)

    with SessionLocalTest() as db:
        user = User(email="normal_pub@example.com", password_hash="hash", full_name="Normal Pub")
        db.add(user)
        db.commit()

        channel = Channel(user_id=user.id, name="Normal Pub Channel", handle="@normal_pub", default_visual_style_id="legacy_default")
        db.add(channel)
        db.commit()

        video = Video(
            channel_id=channel.id,
            topic="Normal Publication",
            visual_style_id="stickman_tactical_explainer",
            visual_style_version=1,
            visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
        )
        db.add(video)
        db.commit()
        _populate_mock_completed_pipeline(db, video)
        vid_id = video.id

    video_dir = tmp_path / str(vid_id)
    video_dir.mkdir(parents=True, exist_ok=True)
    final_canon = video_dir / "final_video.mp4"
    captioned_canon = video_dir / "final_video_captioned.mp4"

    # 1. Authoritative Render
    render_temp = video_dir / "temp_render.mp4"
    render_temp.write_bytes(b"NORMAL_RENDER_CONTENT")
    with SessionLocalTest() as db:
        promoted_render = atomic_promote_file(
            temp_path=render_temp,
            final_path=final_canon,
            session=db,
            video_id=vid_id,
            expected_hash=stickman_hash,
            expected_style_id="stickman_tactical_explainer",
            expected_execution_id="render_norm_001",
            step_name="render",
        )
    assert promoted_render == str(final_canon)
    assert final_canon.is_file()
    assert final_canon.read_bytes() == b"NORMAL_RENDER_CONTENT"

    # 2. Authoritative Subtitles
    sub_temp = video_dir / "temp_sub.mp4"
    sub_temp.write_bytes(b"NORMAL_SUBTITLES_CONTENT")
    with SessionLocalTest() as db:
        promoted_sub = atomic_promote_file(
            temp_path=sub_temp,
            final_path=captioned_canon,
            session=db,
            video_id=vid_id,
            expected_hash=stickman_hash,
            expected_style_id="stickman_tactical_explainer",
            expected_execution_id="sub_norm_001",
            step_name="subtitles",
        )
    assert promoted_sub == str(captioned_canon)
    assert captioned_canon.is_file()
    assert captioned_canon.read_bytes() == b"NORMAL_SUBTITLES_CONTENT"


def test_late_older_worker_cannot_reclaim_ownership_after_newer_job_claimed_step(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    """Scenario 8: A late older worker with an expired or superseded queue lease cannot reclaim ownership
    after a newer queue job has claimed the step. Current jobs publish normally.
    """
    from datetime import datetime, timezone, timedelta
    from app.models import PipelineJob
    from app.steps.images import _claim_step_ownership, _promote_image_bytes, _persist_partial, _build_output_payload
    from app.images.mock import MockImageProvider
    import uuid

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    stickman_hash = canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump())

    video = Video(
        channel_id=test_channel.id,
        topic="Queue Lease Invalidation Test",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    _populate_mock_completed_pipeline(db_session, video)

    img_step = next(s for s in video.steps if s.name == "images")
    img_step.status = "running"
    db_session.commit()

    now = datetime.now(timezone.utc)

    # 1. Job 1 is scheduled and claimed by Worker 1
    lease_1 = uuid.uuid4()
    job_1 = PipelineJob(
        video_id=video.id,
        step_name="images",
        status="running",
        worker_id="worker_001",
        lease_token=lease_1,
        lease_expires_at=now + timedelta(seconds=60),
        scheduled_at=now - timedelta(seconds=10),
        started_at=now - timedelta(seconds=5),
        last_heartbeat_at=now - timedelta(seconds=5),
    )
    db_session.add(job_1)
    db_session.commit()
    db_session.refresh(job_1)

    # Worker 1 claims step ownership
    payload_1 = _build_output_payload(
        items=[],
        warnings=[],
        provider=MockImageProvider(),
        total_scenes=2,
        is_complete=False,
        visual_style_id="stickman_tactical_explainer",
        visual_style_snapshot_hash=stickman_hash,
        execution_id="exec_job_1",
        job_id=str(job_1.id),
        lease_token=str(lease_1),
    )
    claimed_1 = _claim_step_ownership(
        step=img_step,
        payload=payload_1,
        initial_snapshot_hash=stickman_hash,
        initial_style_id="stickman_tactical_explainer",
        execution_id="exec_job_1",
        job_id=job_1.id,
        lease_token=lease_1,
    )
    assert claimed_1 is True

    # 2. Worker 1's lease expires / worker crashes, and Job 2 is scheduled and claimed by Worker 2
    lease_2 = uuid.uuid4()
    job_2 = PipelineJob(
        video_id=video.id,
        step_name="images",
        status="running",
        worker_id="worker_002",
        lease_token=lease_2,
        lease_expires_at=now + timedelta(seconds=60),
        scheduled_at=now,
        started_at=now,
        last_heartbeat_at=now,
    )
    # Mark old job cancelled/superseded
    job_1.status = "cancelled"
    job_1.lease_token = None
    job_1.lease_expires_at = None
    db_session.add(job_2)
    db_session.commit()
    db_session.refresh(job_2)

    # Worker 2 claims step ownership
    payload_2 = _build_output_payload(
        items=[],
        warnings=[],
        provider=MockImageProvider(),
        total_scenes=2,
        is_complete=False,
        visual_style_id="stickman_tactical_explainer",
        visual_style_snapshot_hash=stickman_hash,
        execution_id="exec_job_2",
        job_id=str(job_2.id),
        lease_token=str(lease_2),
    )
    claimed_2 = _claim_step_ownership(
        step=img_step,
        payload=payload_2,
        initial_snapshot_hash=stickman_hash,
        initial_style_id="stickman_tactical_explainer",
        execution_id="exec_job_2",
        job_id=job_2.id,
        lease_token=lease_2,
    )
    assert claimed_2 is True

    # 3. Worker 1 wakes up late and attempts to reclaim ownership with its cancelled/expired lease
    late_claim = _claim_step_ownership(
        step=img_step,
        payload=payload_1,
        initial_snapshot_hash=stickman_hash,
        initial_style_id="stickman_tactical_explainer",
        execution_id="exec_job_1",
        job_id=job_1.id,
        lease_token=lease_1,
    )
    # Late claim must be rejected!
    assert late_claim is False

    # Worker 1 attempts to promote an image file
    dest_img = tmp_path / str(video.id) / "images" / "scene_01_lease_test.png"
    promoted_1 = _promote_image_bytes(
        dest_path=dest_img,
        image_bytes=b"WORKER1_EXPIRED_DATA",
        step=img_step,
        initial_snapshot_hash=stickman_hash,
        initial_style_id="stickman_tactical_explainer",
        execution_id="exec_job_1",
        job_id=job_1.id,
        lease_token=lease_1,
    )
    assert promoted_1 is False
    assert not dest_img.exists()

    # Worker 1 attempts to persist partial progress
    _persist_partial(
        step=img_step,
        payload={
            "images": {"items": [{"index": 1, "status": "ok"}], "complete": False},
            "meta": {
                "visual_style_id": "stickman_tactical_explainer",
                "visual_style_snapshot_hash": stickman_hash,
                "execution_id": "exec_job_1",
                "job_id": str(job_1.id),
                "lease_token": str(lease_1),
            },
        },
        initial_snapshot_hash=stickman_hash,
        initial_style_id="stickman_tactical_explainer",
        execution_id="exec_job_1",
        job_id=job_1.id,
        lease_token=lease_1,
    )
    db_session.refresh(img_step)
    # Step output must NOT have been overwritten by Worker 1
    assert img_step.output["meta"]["execution_id"] == "exec_job_2"
    assert img_step.output["meta"]["job_id"] == str(job_2.id)

    # 4. Worker 2 promotes its image and persists partial progress normally
    promoted_2 = _promote_image_bytes(
        dest_path=dest_img,
        image_bytes=b"WORKER2_VALID_DATA",
        step=img_step,
        initial_snapshot_hash=stickman_hash,
        initial_style_id="stickman_tactical_explainer",
        execution_id="exec_job_2",
        job_id=job_2.id,
        lease_token=lease_2,
    )
    assert promoted_2 is True
    assert dest_img.read_bytes() == b"WORKER2_VALID_DATA"

    _persist_partial(
        step=img_step,
        payload={
            "images": {"items": [{"index": 1, "status": "ok"}], "complete": False},
            "meta": {
                "visual_style_id": "stickman_tactical_explainer",
                "visual_style_snapshot_hash": stickman_hash,
                "execution_id": "exec_job_2",
                "job_id": str(job_2.id),
                "lease_token": str(lease_2),
            },
        },
        initial_snapshot_hash=stickman_hash,
        initial_style_id="stickman_tactical_explainer",
        execution_id="exec_job_2",
        job_id=job_2.id,
        lease_token=lease_2,
    )
    db_session.refresh(img_step)
    assert img_step.output["meta"]["execution_id"] == "exec_job_2"
    assert img_step.output["meta"]["job_id"] == str(job_2.id)
    assert len(img_step.output["images"]["items"]) == 1


def test_image_worker_pre_promotion_pause_cannot_overwrite_newer_image(tmp_path: Path, monkeypatch):
    """Scenario 9: An old image worker pauses immediately before publication and loses its lease.
    A new worker claims the step and publishes. The old worker resumes and cannot overwrite the newer image.
    """
    import threading
    from datetime import datetime, timezone, timedelta
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.models import PipelineJob
    from app.steps.images import _claim_step_ownership, _promote_image_bytes, _build_output_payload
    from app.images.mock import MockImageProvider
    import uuid

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    stickman_hash = canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump())

    file_db = tmp_path / "test_image_pre_promo_pause.db"
    file_engine = create_engine(
        f"sqlite:///{file_db}?timeout=30",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(file_engine)
    SessionLocalTest = sessionmaker(bind=file_engine, expire_on_commit=False)

    with SessionLocalTest() as init_db:
        user = User(email="img_pause_user@example.com", password_hash="hash", full_name="Img Pause User")
        init_db.add(user)
        init_db.commit()

        channel = Channel(user_id=user.id, name="Img Pause Channel", handle="@img_pause", default_visual_style_id="legacy_default")
        init_db.add(channel)
        init_db.commit()

        video = Video(
            channel_id=channel.id,
            topic="Image Pre-Promotion Pause Test",
            visual_style_id="stickman_tactical_explainer",
            visual_style_version=1,
            visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
        )
        init_db.add(video)
        init_db.commit()
        _populate_mock_completed_pipeline(init_db, video)
        vid_id = video.id

        now = datetime.now(timezone.utc)
        lease_1 = uuid.uuid4()
        job_1 = PipelineJob(
            video_id=video.id,
            step_name="images",
            status="running",
            worker_id="worker_001",
            lease_token=lease_1,
            lease_expires_at=now + timedelta(seconds=60),
            scheduled_at=now - timedelta(seconds=10),
            started_at=now - timedelta(seconds=5),
            last_heartbeat_at=now - timedelta(seconds=5),
        )
        init_db.add(job_1)
        init_db.commit()
        init_db.refresh(job_1)
        job_1_id = job_1.id

        img_step = next(s for s in video.steps if s.name == "images")
        img_step.status = "running"
        init_db.commit()

        # Worker 1 claims initial ownership
        payload_1 = _build_output_payload(
            items=[],
            warnings=[],
            provider=MockImageProvider(),
            total_scenes=2,
            is_complete=False,
            visual_style_id="stickman_tactical_explainer",
            visual_style_snapshot_hash=stickman_hash,
            execution_id="exec_job_1",
            job_id=str(job_1_id),
            lease_token=str(lease_1),
        )
        claimed_1 = _claim_step_ownership(
            step=img_step,
            payload=payload_1,
            initial_snapshot_hash=stickman_hash,
            initial_style_id="stickman_tactical_explainer",
            execution_id="exec_job_1",
            job_id=job_1_id,
            lease_token=lease_1,
        )
        assert claimed_1 is True

    video_dir = tmp_path / str(vid_id)
    dest_img = video_dir / "images" / "scene_01_target.png"

    event_w1_paused = threading.Event()
    event_resume_w1 = threading.Event()
    w1_promoted_result: list[bool] = []

    def w1_pre_promote_hook():
        event_w1_paused.set()
        event_resume_w1.wait(timeout=5)

    def run_worker_1():
        with SessionLocalTest() as session_w1:
            v1 = session_w1.get(Video, vid_id)
            step_w1 = next(s for s in v1.steps if s.name == "images")
            res = _promote_image_bytes(
                dest_path=dest_img,
                image_bytes=b"WORKER_1_STALE_DATA",
                step=step_w1,
                initial_snapshot_hash=stickman_hash,
                initial_style_id="stickman_tactical_explainer",
                execution_id="exec_job_1",
                job_id=job_1_id,
                lease_token=lease_1,
                _pre_promote_hook=w1_pre_promote_hook,
            )
            w1_promoted_result.append(res)

    t_w1 = threading.Thread(target=run_worker_1)
    t_w1.start()

    assert event_w1_paused.wait(timeout=5), "Worker 1 did not reach pre-promotion hook"

    # Now Worker 1 is paused right before filesystem promotion.
    # Worker 1's lease expires; Job 2 is scheduled and claimed by Worker 2.
    with SessionLocalTest() as session_w2:
        j1 = session_w2.get(PipelineJob, job_1_id)
        j1.status = "cancelled"
        j1.lease_token = None
        j1.lease_expires_at = None

        lease_2 = uuid.uuid4()
        now = datetime.now(timezone.utc)
        job_2 = PipelineJob(
            video_id=vid_id,
            step_name="images",
            status="running",
            worker_id="worker_002",
            lease_token=lease_2,
            lease_expires_at=now + timedelta(seconds=60),
            scheduled_at=now,
            started_at=now,
            last_heartbeat_at=now,
        )
        session_w2.add(job_2)
        session_w2.commit()
        session_w2.refresh(job_2)
        job_2_id = job_2.id

        v2 = session_w2.get(Video, vid_id)
        step_w2 = next(s for s in v2.steps if s.name == "images")

        # Worker 2 claims step ownership
        payload_2 = _build_output_payload(
            items=[],
            warnings=[],
            provider=MockImageProvider(),
            total_scenes=2,
            is_complete=False,
            visual_style_id="stickman_tactical_explainer",
            visual_style_snapshot_hash=stickman_hash,
            execution_id="exec_job_2",
            job_id=str(job_2_id),
            lease_token=str(lease_2),
        )
        claimed_2 = _claim_step_ownership(
            step=step_w2,
            payload=payload_2,
            initial_snapshot_hash=stickman_hash,
            initial_style_id="stickman_tactical_explainer",
            execution_id="exec_job_2",
            job_id=job_2_id,
            lease_token=lease_2,
        )
        assert claimed_2 is True

        # Worker 2 publishes its authoritative image
        promoted_w2 = _promote_image_bytes(
            dest_path=dest_img,
            image_bytes=b"WORKER_2_AUTHORITATIVE_DATA",
            step=step_w2,
            initial_snapshot_hash=stickman_hash,
            initial_style_id="stickman_tactical_explainer",
            execution_id="exec_job_2",
            job_id=job_2_id,
            lease_token=lease_2,
        )
        assert promoted_w2 is True
        assert dest_img.read_bytes() == b"WORKER_2_AUTHORITATIVE_DATA"

    # Resume Worker 1
    event_resume_w1.set()
    t_w1.join(timeout=5)
    assert not t_w1.is_alive()

    # Worker 1's promotion MUST have been rejected under the publication lock!
    assert w1_promoted_result == [False]
    # The destination file MUST NOT have been overwritten by Worker 1!
    assert dest_img.read_bytes() == b"WORKER_2_AUTHORITATIVE_DATA"


def test_concurrent_image_executions_receive_distinct_generation_paths(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    """Scenario 10: Concurrent image executions receive distinct generation paths on disk and never collide."""
    from app.steps.images import generate_images_step
    from app.images.mock import MockImageProvider

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = Video(
        channel_id=test_channel.id,
        topic="Distinct Generation Paths Test",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    _populate_mock_completed_pipeline(db_session, video)

    sb_output = next(s for s in video.steps if s.name == "storyboard").output
    img_step = next(s for s in video.steps if s.name == "images")
    img_step.status = "running"
    db_session.commit()

    # Execution 1
    img_step._execution_id = "exec_alpha_111"
    res1 = generate_images_step(video=video, prev={"storyboard": sb_output}, step=img_step, image_provider=MockImageProvider())
    file1 = res1["images"]["items"][0]["file"]

    # Execution 2 (simulating a concurrent execution from another worker before first output committed)
    img_step.output = None
    img_step._execution_id = "exec_beta_222"
    res2 = generate_images_step(video=video, prev={"storyboard": sb_output}, step=img_step, image_provider=MockImageProvider())
    file2 = res2["images"]["items"][0]["file"]

    # The generation paths MUST be distinct
    assert file1 != file2
    assert "generations" in file1
    assert "generations" in file2
    assert "exec_alp" in file1
    assert "exec_bet" in file2

    # Both generation files MUST exist independently on disk
    path1 = tmp_path / file1
    path2 = tmp_path / file2
    assert path1.is_file()
    assert path2.is_file()


def test_normal_image_generation_and_downstream_rendering_uses_published_paths(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    """Scenario 11: Normal image generation and downstream render resolve authoritative image paths correctly."""
    from app.steps.images import generate_images_step
    from app.steps.render import generate_render_step
    from app.images.mock import MockImageProvider
    from app.voiceover.mock import create_mock_wav
    from app.routers.videos import get_scene_image
    from unittest.mock import MagicMock
    import subprocess

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = Video(
        channel_id=test_channel.id,
        topic="Downstream Render Resolution Test",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    _populate_mock_completed_pipeline(db_session, video)

    video_dir = tmp_path / str(video.id)
    audio_dir = video_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    audio_file = audio_dir / "narration_abc12345.wav"
    audio_file.write_bytes(create_mock_wav(6.0, 24000))

    vo_step = next(s for s in video.steps if s.name == "voiceover")
    vo_step.output["voiceover"]["audio_file"] = str(audio_file.relative_to(tmp_path))
    vo_step.output["audio_file"] = str(audio_file.relative_to(tmp_path))
    db_session.commit()

    sb_output = next(s for s in video.steps if s.name == "storyboard").output
    img_step = next(s for s in video.steps if s.name == "images")
    img_step.status = "running"
    db_session.commit()

    # 1. Generate images step
    img_output = generate_images_step(video=video, prev={"storyboard": sb_output}, step=img_step, image_provider=MockImageProvider())
    assert img_output["images"]["complete"] is True
    item0_file = img_output["images"]["items"][0]["file"]
    assert (tmp_path / item0_file).is_file()

    # 2. Test get_scene_image endpoint serves the image
    resp = get_scene_image(index=1, video=video)
    assert resp.status_code == 200
    assert resp.path == str((tmp_path / item0_file).resolve())

    # 3. Downstream render executes using the published image paths
    def mock_subprocess_run(cmd, *args, **kwargs):
        for arg in cmd:
            arg_str = str(arg)
            if arg_str.endswith(".mp4"):
                p = Path(arg_str)
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b"dummy_rendered_mp4_bytes")
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)
    monkeypatch.setattr(
        "app.steps.render._probe_video_file",
        MagicMock(return_value={
            "streams": [
                {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p", "width": 1080, "height": 1920, "duration": 6.0},
                {"codec_type": "audio", "codec_name": "aac", "duration": 6.0},
            ],
            "format": {"duration": "6.0", "size": "10000"},
        }),
    )

    r_step = next(s for s in video.steps if s.name == "render")
    r_step.status = "running"
    db_session.commit()

    render_output = generate_render_step(
        video=video,
        prev={
            "storyboard": sb_output,
            "images": img_output,
            "voiceover": vo_step.output,
        },
        step=r_step,
    )
    assert render_output["render"]["scenes_rendered"] >= 1
    assert Path(render_output["render"]["video_path"]).is_file()


def test_image_commit_failure_preserves_previous_authoritative_output(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    """Scenario 12: Database commit failure during partial persistence preserves previous authoritative output untouched."""
    from app.steps.images import _persist_partial
    from unittest.mock import patch

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    stickman_hash = canonical_snapshot_hash(STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump())

    video = Video(
        channel_id=test_channel.id,
        topic="Commit Failure Test",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    _populate_mock_completed_pipeline(db_session, video)

    img_step = next(s for s in video.steps if s.name == "images")
    img_step.status = "running"
    orig_output = {
        "images": {"items": [{"index": 1, "status": "ok", "file": "authoritative_img.png"}], "complete": False},
        "meta": {"visual_style_snapshot_hash": stickman_hash, "execution_id": "auth_exec_001"},
    }
    img_step.output = orig_output
    db_session.commit()

    # Attempt to persist newer partial output with a simulated commit error
    with patch.object(db_session, "commit", side_effect=RuntimeError("Simulated DB Disk Full")):
        _persist_partial(
            step=img_step,
            payload={
                "images": {"items": [{"index": 1, "status": "ok", "file": "corrupted_img.png"}], "complete": False},
                "meta": {"visual_style_snapshot_hash": stickman_hash, "execution_id": "auth_exec_001"},
            },
            initial_snapshot_hash=stickman_hash,
            initial_style_id="stickman_tactical_explainer",
            execution_id="auth_exec_001",
        )

    db_session.rollback()
    db_session.refresh(img_step)

    # Previous authoritative output MUST remain completely intact!
    assert img_step.output["images"]["items"][0]["file"] == "authoritative_img.png"
    assert img_step.output["meta"]["execution_id"] == "auth_exec_001"


def test_pure_metadata_image_publication_no_canonical_files_created(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    """Scenario 13: Images are published purely via metadata and immutable generation paths with no canonical mirrors."""
    from app.steps.images import generate_images_step
    from app.images.mock import MockImageProvider

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = Video(
        channel_id=test_channel.id,
        topic="Pure Metadata Image Test",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    _populate_mock_completed_pipeline(db_session, video)

    sb_output = next(s for s in video.steps if s.name == "storyboard").output
    img_step = next(s for s in video.steps if s.name == "images")
    img_step.status = "running"
    db_session.commit()

    res = generate_images_step(video=video, prev={"storyboard": sb_output}, step=img_step, image_provider=MockImageProvider())
    assert res["images"]["complete"] is True

    # Generation files must exist in generations directory
    for item in res["images"]["items"]:
        file_path = item["file"]
        assert "generations" in file_path
        assert "canonical_file" not in item
        assert (tmp_path / file_path).is_file()

    # The images directory itself must NOT contain any canonical files (only the generations/ subdirectory)
    images_dir = tmp_path / str(video.id) / "images"
    direct_files = [f for f in images_dir.iterdir() if f.is_file()]
    assert len(direct_files) == 0, f"Expected no direct canonical files in images/, found: {direct_files}"


def test_image_cache_reuse_uses_metadata_not_unreferenced_canonical_files(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    """Scenario 14: Cache reuse strictly validates metadata pointer and disk existence; ignores unreferenced canonical files."""
    from app.steps.images import generate_images_step, compute_image_cache_key
    from app.images.mock import MockImageProvider

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = Video(
        channel_id=test_channel.id,
        topic="Cache Reuse Validation Test",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    _populate_mock_completed_pipeline(db_session, video)

    sb_output = next(s for s in video.steps if s.name == "storyboard").output
    img_step = next(s for s in video.steps if s.name == "images")
    img_step.status = "running"
    db_session.commit()

    # Run initial generation
    res1 = generate_images_step(video=video, prev={"storyboard": sb_output}, step=img_step, image_provider=MockImageProvider())
    assert res1["images"]["complete"] is True
    first_item = res1["images"]["items"][0]
    first_file = first_item["file"]
    assert first_item["reused"] is False

    # Second run with same step output: metadata pointer exists and file exists -> REUSED
    res2 = generate_images_step(video=video, prev={"storyboard": sb_output}, step=img_step, image_provider=MockImageProvider())
    assert res2["images"]["items"][0]["reused"] is True
    assert res2["images"]["items"][0]["file"] == first_file

    # Now simulate unreferenced canonical file on disk, but metadata pointer missing or deleted
    img_step.output = None
    db_session.commit()

    # Place a rogue canonical file matching prompt hash
    prompt_text = res1["images"]["items"][0]["prompt"]
    full_hash = compute_image_cache_key(
        prompt=prompt_text,
        provider_name="mock",
        model="mock-canvas",
        size="1080x1920",
        quality="standard",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    hash8 = full_hash[:8]
    rogue_canon = tmp_path / str(video.id) / "images" / f"scene_01_{hash8}.png"
    rogue_canon.parent.mkdir(parents=True, exist_ok=True)
    rogue_canon.write_bytes(b"ROGUE_UNREFERENCED_CANONICAL_BYTES")

    # Run generation: must NOT reuse the rogue unreferenced canonical file!
    res3 = generate_images_step(video=video, prev={"storyboard": sb_output}, step=img_step, image_provider=MockImageProvider())
    item3 = res3["images"]["items"][0]
    assert item3["reused"] is False
    assert item3["file"] != str(rogue_canon.relative_to(tmp_path))
    assert "generations" in item3["file"]
    assert (tmp_path / item3["file"]).read_bytes() != b"ROGUE_UNREFERENCED_CANONICAL_BYTES"


def test_legacy_image_metadata_compatibility_preserved(db_session: Session, test_channel: Channel, tmp_path: Path, monkeypatch):
    """Scenario 15: Legacy video images with non-generation paths stored in metadata are safely preserved, reused, and served."""
    from app.steps.images import generate_images_step
    from app.images.mock import MockImageProvider
    from app.routers.videos import get_scene_image

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = Video(
        channel_id=test_channel.id,
        topic="Legacy Image Compat Test",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    db_session.add(video)
    db_session.commit()
    _populate_mock_completed_pipeline(db_session, video)

    sb_output = next(s for s in video.steps if s.name == "storyboard").output
    img_step = next(s for s in video.steps if s.name == "images")
    img_step.status = "running"
    db_session.commit()

    # 1. Run generation to establish valid step items
    res1 = generate_images_step(video=video, prev={"storyboard": sb_output}, step=img_step, image_provider=MockImageProvider())
    assert res1["images"]["complete"] is True
    item1 = res1["images"]["items"][0]

    # 2. Setup legacy file directly in images/ and update DB metadata to point to it
    legacy_rel_file = f"{video.id}/images/legacy_scene_01.png"
    legacy_abs_file = tmp_path / legacy_rel_file
    legacy_abs_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_abs_file.write_bytes(b"LEGACY_COMMITTED_IMAGE_DATA")

    import copy
    new_output = copy.deepcopy(img_step.output)
    new_output["images"]["items"][0]["file"] = legacy_rel_file
    img_step.output = new_output
    db_session.add(img_step)
    db_session.commit()

    # 3. API endpoint get_scene_image serves legacy file correctly
    resp = get_scene_image(index=1, video=video)
    assert resp.status_code == 200
    assert resp.path == str(legacy_abs_file.resolve())

    # 4. generate_images_step recognizes committed legacy file in metadata and reuses it safely
    res2 = generate_images_step(video=video, prev={"storyboard": sb_output}, step=img_step, image_provider=MockImageProvider())
    reused_it = res2["images"]["items"][0]
    assert reused_it["reused"] is True
    assert reused_it["file"] == legacy_rel_file






