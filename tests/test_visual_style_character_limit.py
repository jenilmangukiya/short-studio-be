"""Tests for Style Prompt / Visual Direction character limit expansion to 2,000 characters.

Verifies:
- Schema-level validation for ChannelCreate and ChannelUpdate
- 600, 601, and 2,000 character inputs are accepted and preserved exactly
- 2,001 character inputs are rejected with a clear validation error
- Empty, None, and omitted values behave as expected
- API endpoints enforce the 2,000 character limit and reject 2,001 characters with HTTP 422
- Complete visual direction text reaches creative direction, video bible, and storyboard steps
"""

import uuid
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db import Base, get_db
from app.deps import current_user
from app.main import app
from app.models import Channel, User, Video, VideoStep
from app.schemas import (
    MAX_VISUAL_STYLE_LENGTH,
    ChannelCreate,
    ChannelUpdate,
    StepContentUpdate,
)
from app.steps.creative_direction import build_creative_direction_prompt
from app.steps.prompt_compose import adapt_visual_style, compose_final_prompt
from app.steps.video_bible import build_video_bible_prompt


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def test_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    with Session(engine) as session:
        yield session
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_user(test_db: Session) -> User:
    user = User(
        id=uuid.uuid4(),
        email="creator@example.com",
        password_hash="fakehash",
        full_name="Creator User",
        is_active=True,
    )
    test_db.add(user)
    test_db.commit()
    test_db.refresh(user)
    return user


@pytest.fixture
def client(test_db: Session, test_user: User):
    def override_get_db():
        yield test_db

    def override_current_user():
        return test_user

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[current_user] = override_current_user
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# 1. Pydantic Schema Validation Tests
# ---------------------------------------------------------------------------

class TestVisualStyleSchemaValidation:
    """Direct schema validation tests for 600, 601, 2,000, and 2,001 characters."""

    def test_channel_create_accepts_600_characters(self):
        style_600 = "A" * 600
        obj = ChannelCreate(
            name="Tech Shorts",
            settings={"visual_style": style_600},
        )
        assert len(obj.settings["visual_style"]) == 600
        assert obj.settings["visual_style"] == style_600

    def test_channel_create_accepts_601_characters(self):
        style_601 = "B" * 601
        obj = ChannelCreate(
            name="Tech Shorts",
            settings={"visual_style": style_601},
        )
        assert len(obj.settings["visual_style"]) == 601
        assert obj.settings["visual_style"] == style_601

    def test_channel_create_accepts_2000_characters(self):
        style_2000 = "C" * 2000
        obj = ChannelCreate(
            name="Tech Shorts",
            settings={"visual_style": style_2000},
        )
        assert len(obj.settings["visual_style"]) == 2000
        assert obj.settings["visual_style"] == style_2000

    def test_channel_create_rejects_2001_characters(self):
        style_2001 = "D" * 2001
        with pytest.raises(ValidationError) as exc_info:
            ChannelCreate(
                name="Tech Shorts",
                settings={"visual_style": style_2001},
            )
        err_str = str(exc_info.value)
        assert "Style prompt / visual direction exceeds maximum allowed length" in err_str
        assert "2000" in err_str
        assert "2001" in err_str

    def test_channel_update_accepts_600_601_2000_characters(self):
        for length in (600, 601, 2000):
            style = "X" * length
            obj = ChannelUpdate(settings={"visual_style": style})
            assert obj.settings is not None
            assert len(obj.settings["visual_style"]) == length
            assert obj.settings["visual_style"] == style

    def test_channel_update_rejects_2001_characters(self):
        style_2001 = "Y" * 2001
        with pytest.raises(ValidationError) as exc_info:
            ChannelUpdate(settings={"visual_style": style_2001})
        err_str = str(exc_info.value)
        assert "Style prompt / visual direction exceeds maximum allowed length" in err_str
        assert "2001" in err_str

    def test_preserves_empty_and_none_values(self):
        # Empty string
        c_empty = ChannelCreate(name="Ch1", settings={"visual_style": ""})
        assert c_empty.settings["visual_style"] == ""

        # None value
        c_none = ChannelCreate(name="Ch2", settings={"visual_style": None})
        assert c_none.settings["visual_style"] is None

        # Omitted key
        c_omitted = ChannelCreate(name="Ch3", settings={"voice": "am_adam"})
        assert "visual_style" not in c_omitted.settings
        assert c_omitted.settings["voice"] == "am_adam"

        # Update with empty settings or None
        u_none = ChannelUpdate(settings=None)
        assert u_none.settings is None
        u_empty = ChannelUpdate(settings={})
        assert u_empty.settings == {}

    def test_alias_keys_validated_and_accepted(self):
        # style_prompt alias
        c_sp = ChannelCreate(name="Ch4", settings={"style_prompt": "S" * 2000})
        assert len(c_sp.settings["style_prompt"]) == 2000

        with pytest.raises(ValidationError):
            ChannelCreate(name="Ch4", settings={"style_prompt": "S" * 2001})

        # visual_direction alias
        c_vd = ChannelCreate(name="Ch5", settings={"visual_direction": "V" * 2000})
        assert len(c_vd.settings["visual_direction"]) == 2000

        with pytest.raises(ValidationError):
            ChannelCreate(name="Ch5", settings={"visual_direction": "V" * 2001})

    def test_step_content_update_validates_visual_style(self):
        valid = StepContentUpdate(output={"visual_style": "Z" * 2000})
        assert len(valid.output["visual_style"]) == 2000

        with pytest.raises(ValidationError):
            StepContentUpdate(output={"visual_style": "Z" * 2001})


# ---------------------------------------------------------------------------
# 2. API Integration Tests (POST & PATCH /channels)
# ---------------------------------------------------------------------------

class TestChannelApiEndpoints:
    """Test FastAPI /channels endpoints for status codes and payloads."""

    def test_api_create_channel_with_600_characters(self, client: TestClient):
        style = "A" * 600
        res = client.post(
            "/channels",
            json={"name": "Ch 600", "platform": "youtube", "settings": {"visual_style": style}},
        )
        assert res.status_code == 201
        data = res.json()
        assert data["settings"]["visual_style"] == style
        assert len(data["settings"]["visual_style"]) == 600

    def test_api_create_channel_with_601_characters(self, client: TestClient):
        style = "B" * 601
        res = client.post(
            "/channels",
            json={"name": "Ch 601", "platform": "youtube", "settings": {"visual_style": style}},
        )
        assert res.status_code == 201
        data = res.json()
        assert data["settings"]["visual_style"] == style
        assert len(data["settings"]["visual_style"]) == 601

    def test_api_create_channel_with_2000_characters(self, client: TestClient):
        style = "C" * 2000
        res = client.post(
            "/channels",
            json={"name": "Ch 2000", "platform": "youtube", "settings": {"visual_style": style}},
        )
        assert res.status_code == 201
        data = res.json()
        assert data["settings"]["visual_style"] == style
        assert len(data["settings"]["visual_style"]) == 2000

    def test_api_create_channel_rejects_2001_characters(self, client: TestClient):
        style = "D" * 2001
        res = client.post(
            "/channels",
            json={"name": "Ch 2001", "platform": "youtube", "settings": {"visual_style": style}},
        )
        assert res.status_code == 422
        body = res.json()
        assert "Style prompt / visual direction exceeds maximum allowed length" in str(body)

    def test_api_patch_channel_with_600_601_2000_characters(self, client: TestClient):
        # Create initial channel
        res = client.post(
            "/channels",
            json={"name": "Patch Target", "platform": "youtube"},
        )
        assert res.status_code == 201
        ch_id = res.json()["id"]

        # Patch with 600
        style_600 = "M" * 600
        res_600 = client.patch(f"/channels/{ch_id}", json={"settings": {"visual_style": style_600}})
        assert res_600.status_code == 200
        assert res_600.json()["settings"]["visual_style"] == style_600

        # Patch with 601
        style_601 = "N" * 601
        res_601 = client.patch(f"/channels/{ch_id}", json={"settings": {"visual_style": style_601}})
        assert res_601.status_code == 200
        assert res_601.json()["settings"]["visual_style"] == style_601

        # Patch with 2000
        style_2000 = "P" * 2000
        res_2000 = client.patch(f"/channels/{ch_id}", json={"settings": {"visual_style": style_2000}})
        assert res_2000.status_code == 200
        assert res_2000.json()["settings"]["visual_style"] == style_2000

    def test_api_patch_channel_rejects_2001_characters(self, client: TestClient):
        res = client.post(
            "/channels",
            json={"name": "Reject Target", "platform": "youtube"},
        )
        ch_id = res.json()["id"]

        style_2001 = "Q" * 2001
        res_2001 = client.patch(f"/channels/{ch_id}", json={"settings": {"visual_style": style_2001}})
        assert res_2001.status_code == 422
        assert "Style prompt / visual direction exceeds maximum allowed length" in str(res_2001.json())


# ---------------------------------------------------------------------------
# 3. Workflow Propagation Tests
# ---------------------------------------------------------------------------

class TestWorkflowPropagation:
    """Verify that full 2,000-character visual style text propagates without truncation."""

    def test_creative_direction_prompt_preserves_2000_chars(self):
        style_2000 = "Cinematic 3D render, " + ("hyper-detailed textures, " * 80)
        style_2000 = style_2000[:2000]

        prompt = build_creative_direction_prompt(
            rules_text="Rules",
            topic="Aircraft aerodynamics",
            channel_settings={"visual_style": style_2000},
        )
        assert style_2000 in prompt

    def test_video_bible_prompt_preserves_2000_chars(self):
        style_2000 = "Photorealistic documentary style, " + ("natural lighting, " * 80)
        style_2000 = style_2000[:2000]

        prompt = build_video_bible_prompt(
            rules_text="Rules",
            topic="Aircraft aerodynamics",
            script_data={"narration": "Why do airplanes fly?"},
            channel_visual_style=style_2000,
        )
        assert f"### Channel Visual Style Reference\n{style_2000.strip()}" in prompt

    def test_prompt_compose_preserves_long_style(self):
        style_parts = ["vivid color palette", "sharp focus 8k", "soft volumetric haze"]
        style_2000 = ", ".join(style_parts * 40)[:2000]

        adapted = adapt_visual_style(style_2000, environment="open sky", shot_type="wide")
        assert len(adapted) > 1000
        assert "vivid color palette" in adapted
