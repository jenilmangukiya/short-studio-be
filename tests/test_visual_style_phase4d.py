"""Phase 4D: Video Bible Visual Style Integration Tests.

Validates:
1. stickman_tactical_explainer snapshot injects its correct visual style information.
2. stylized_2d_narrative snapshot injects its correct visual style information.
3. Legacy videos with NULL style fields retain byte-for-byte identical prompt output.
4. Legacy channel visual style fallback still works.
5. Both snapshot and channel settings absent are handled correctly without errors.
6. Malformed snapshots fail closed (VisualStyleResolutionError).
7. Explicit unknown style IDs and version mismatches retain existing failure behavior (VisualStyleResolutionError).
8. Narrative-specific Video Bible information (characters, environments, objects, concepts)
   is not overwritten by generic style descriptions.
9. Existing Video Bible output schema and downstream storyboard integration remain compatible.
10. Deterministic style-field enforcement properly aligns presentation attributes.
"""

from __future__ import annotations

import uuid
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.db import Base
from app.models import Channel, User, Video, VideoStep
from app.pipeline import PipelineError, init_steps
from app.schemas import (
    ArtDirection,
    CharacterEntity,
    EntityCatalog,
    EnvironmentEntity,
    VideoBibleOutput,
)
from app.styles import (
    STICKMAN_TACTICAL_EXPLAINER_PRESET,
    STYLIZED_2D_NARRATIVE_PRESET,
    VisualStyleResolutionError,
    get_style_by_id,
)
from app.steps.video_bible import (
    build_video_bible_prompt,
    generate_video_bible_step,
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
        full_name="Phase 4D Test User",
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
        settings={"visual_style": "Legacy Channel Aesthetic: Warm Amber Glow"},
    )
    db_session.add(channel)
    db_session.commit()
    db_session.refresh(channel)
    return channel


def _create_test_video(
    db_session: Session,
    channel: Channel,
    visual_style_id: str | None = None,
    visual_style_snapshot: dict | None = None,
    visual_style_version: int | None = None,
) -> Video:
    vid = Video(
        channel_id=channel.id,
        topic="The Secret Biology of Sleep",
        description="Explaining hypnic jerks and muscle twitches before sleep.",
        visual_style_id=visual_style_id,
        visual_style_snapshot=visual_style_snapshot,
        visual_style_version=visual_style_version,
    )
    db_session.add(vid)
    db_session.commit()
    db_session.refresh(vid)
    init_steps(db_session, vid)
    db_session.refresh(vid)
    return vid


SAMPLE_SCRIPT_DATA = {
    "title": "Why You Twitch Before Sleep",
    "hook": "Have you ever felt like you were suddenly falling in bed?",
    "narration": "Have you ever felt like you were suddenly falling in bed? Your brain thinks muscles went slack.",
}

SAMPLE_CD_DATA = {
    "format_archetype": "explainer_myth_bust",
    "storytelling_angle": "Debunking hypnic jerk misconceptions",
    "central_thesis": "Sensory signal confusion during sleep onset.",
    "tone": "engaging",
    "visual_genre": "photorealistic_cinematic",  # generic suggestion that should be superseded by preset
    "protagonist_config": {"character_count": "none"},
    "creative_constraints": ["No sudden flashes"],
}


# =========================================================================
# 1. Preset Injection Tests
# =========================================================================

def test_stickman_tactical_explainer_prompt_injection():
    preset = STICKMAN_TACTICAL_EXPLAINER_PRESET
    prompt = build_video_bible_prompt(
        rules_text="RULES_MOCK",
        topic="Sleep Science",
        script_data=SAMPLE_SCRIPT_DATA,
        creative_direction_data=SAMPLE_CD_DATA,
        visual_style=preset,
    )

    assert "### Visual Style Art Direction" in prompt
    assert "Stickman Tactical Explainer" in prompt
    assert "- **Style Family**: illustrated" in prompt
    assert "- **Visual Genre**: stickman_illustrated" in prompt
    assert "- **Rendering Texture**: 2D vector ink contours with flat color fills" in prompt
    assert "- **Lighting Philosophy**: directional ambient lighting" in prompt
    assert "#F4A92E" in prompt
    assert "Character Drawing Constraints" in prompt
    assert "uniform 3px black outlines" in prompt
    assert "Environment & Perspective Rules" in prompt
    assert "Strict 1-point perspective" in prompt
    assert "Forbidden Visual Elements" in prompt
    assert "no photorealistic" in prompt
    assert "PRIORITY: The visual presentation rules below strictly supersede" in prompt
    # Channel visual style fallback must NOT be present when preset is active
    assert "Channel Visual Style Reference" not in prompt


def test_stylized_2d_narrative_prompt_injection():
    preset = STYLIZED_2D_NARRATIVE_PRESET
    prompt = build_video_bible_prompt(
        rules_text="RULES_MOCK",
        topic="Night Terrors",
        script_data=SAMPLE_SCRIPT_DATA,
        creative_direction_data=SAMPLE_CD_DATA,
        visual_style=preset,
    )

    assert "### Visual Style Art Direction" in prompt
    assert "Stylized 2D Narrative Drama" in prompt
    assert "- **Style Family**: illustrated" in prompt
    assert "- **Visual Genre**: 2d_cartoon" in prompt
    assert "two-tone digital cel-shading" in prompt
    assert "atmospheric dramatic lighting" in prompt
    assert "#E91E63" in prompt
    assert "sculpted volume hair" in prompt
    assert "Forbidden Visual Elements" in prompt
    assert "no 3D volumetric render" in prompt


# =========================================================================
# 2. Legacy Behavior & Byte-for-Byte Preservation Tests
# =========================================================================

def test_legacy_video_byte_for_byte_preservation_with_channel_style():
    """Verify legacy prompt generation with channel fallback is byte-for-byte preserved."""
    rules = "BASE_RULES_MOCK_FOR_BYTE_COMPARISON"
    prompt = build_video_bible_prompt(
        rules_text=rules,
        topic="Ancient Rome",
        script_data={"title": "The Fall of Rome", "hook": "Why did Rome collapse?", "narration": "In 476 AD, Rome fell."},
        creative_direction_data={"format_archetype": "deep_dive", "visual_genre": "styled_3d"},
        channel_visual_style="Cinematic channel style",
        visual_style=None,
    )

    # Reconstruct pre-Phase 4D expected text exactly
    expected_lines = [
        "BASE_RULES_MOCK_FOR_BYTE_COMPARISON",
        "\n## Project Ground Truth",
        "- **Video Topic**: Ancient Rome",
        "- **Video Title**: The Fall of Rome",
        "- **Opening Hook**: Why did Rome collapse?",
        "- **Locked Narration Script**:\nIn 476 AD, Rome fell.",
        "\n### Approved Creative Direction",
        "- **Format Archetype**: deep_dive",
        "- **Narrative Angle**: ",
        "- **Central Thesis**: ",
        "- **Tone**: conversational",
        "- **Visual Genre**: styled_3d",
        "- **Protagonist Configuration**: none",
        "\n### Channel Visual Style Reference\nCinematic channel style",
        "\n## Instructions\nGenerate a structured VideoBibleOutput adhering strictly to the schema. "
        "Create an Entity Catalog containing only entities directly utilized by the narration. "
        "Define at least one primary environment. Do NOT create characters if the creative direction specified character_count='none'.",
    ]
    expected_prompt = "\n".join(expected_lines)

    assert prompt == expected_prompt
    assert "### Visual Style Art Direction" not in prompt


def test_legacy_video_byte_for_byte_preservation_without_channel_style():
    """Verify legacy prompt generation when both style preset and channel style are absent."""
    rules = "BASE_RULES_MOCK_FOR_BYTE_COMPARISON"
    prompt = build_video_bible_prompt(
        rules_text=rules,
        topic="Ancient Rome",
        script_data={"title": "The Fall of Rome", "hook": "Why did Rome collapse?", "narration": "In 476 AD, Rome fell."},
        creative_direction_data=None,
        channel_visual_style=None,
        visual_style=None,
    )

    expected_lines = [
        "BASE_RULES_MOCK_FOR_BYTE_COMPARISON",
        "\n## Project Ground Truth",
        "- **Video Topic**: Ancient Rome",
        "- **Video Title**: The Fall of Rome",
        "- **Opening Hook**: Why did Rome collapse?",
        "- **Locked Narration Script**:\nIn 476 AD, Rome fell.",
        "\n## Instructions\nGenerate a structured VideoBibleOutput adhering strictly to the schema. "
        "Create an Entity Catalog containing only entities directly utilized by the narration. "
        "Define at least one primary environment. Do NOT create characters if the creative direction specified character_count='none'.",
    ]
    expected_prompt = "\n".join(expected_lines)

    assert prompt == expected_prompt
    assert "### Visual Style Art Direction" not in prompt
    assert "### Channel Visual Style Reference" not in prompt


# =========================================================================
# 3. Fail-Closed Snapshot & Resolution Tests
# =========================================================================

def test_malformed_snapshot_fails_closed(db_session, test_channel, monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    video = _create_test_video(
        db_session,
        test_channel,
        visual_style_id="stickman_tactical_explainer",
        visual_style_snapshot={"invalid": "malformed_data_missing_fields"},
    )
    vb_step = next(s for s in video.steps if s.name == "video_bible")
    prev = {
        "script": {"script": SAMPLE_SCRIPT_DATA},
        "creative_direction": {"creative_direction": SAMPLE_CD_DATA},
    }

    with pytest.raises(VisualStyleResolutionError, match="Malformed or invalid visual style snapshot"):
        generate_video_bible_step(video, prev=prev, step=vb_step)


def test_unknown_style_id_without_snapshot_fails_closed(db_session, test_channel, monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    video = _create_test_video(
        db_session,
        test_channel,
        visual_style_id="unknown_fantasy_preset",
        visual_style_snapshot=None,
    )
    vb_step = next(s for s in video.steps if s.name == "video_bible")
    prev = {
        "script": {"script": SAMPLE_SCRIPT_DATA},
        "creative_direction": {"creative_direction": SAMPLE_CD_DATA},
    }

    with pytest.raises(VisualStyleResolutionError, match="Unknown visual style ID 'unknown_fantasy_preset'"):
        generate_video_bible_step(video, prev=prev, step=vb_step)


def test_incompatible_version_fails_closed(db_session, test_channel, monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    video = _create_test_video(
        db_session,
        test_channel,
        visual_style_id="stickman_tactical_explainer",
        visual_style_snapshot=None,
        visual_style_version=999,  # incompatible version
    )
    vb_step = next(s for s in video.steps if s.name == "video_bible")
    prev = {
        "script": {"script": SAMPLE_SCRIPT_DATA},
        "creative_direction": {"creative_direction": SAMPLE_CD_DATA},
    }

    with pytest.raises(VisualStyleResolutionError, match="Incompatible visual style version 999"):
        generate_video_bible_step(video, prev=prev, step=vb_step)


# =========================================================================
# 4. Deterministic Style-Field Enforcement & Narrative Preservation Tests
# =========================================================================

def test_deterministic_style_enforcement_and_narrative_preservation(db_session, test_channel, monkeypatch):
    """Verify that preset art direction is deterministically enforced on output

    while narrative-specific character entities, environments, and topic summary are 100% preserved.
    """
    monkeypatch.setattr(settings, "llm_provider", "mock")

    preset = STICKMAN_TACTICAL_EXPLAINER_PRESET
    video = _create_test_video(
        db_session,
        test_channel,
        visual_style_id=preset.id,
        visual_style_snapshot=preset.model_dump(),
        visual_style_version=preset.version,
    )
    vb_step = next(s for s in video.steps if s.name == "video_bible")
    prev = {
        "script": {"script": SAMPLE_SCRIPT_DATA},
        "creative_direction": {"creative_direction": SAMPLE_CD_DATA},
    }

    output = generate_video_bible_step(video, prev=prev, step=vb_step)
    assert "video_bible" in output
    vb_data = output["video_bible"]

    # 1. Check deterministic art direction alignment
    art = vb_data["art_direction"]
    assert art["style_family"] == "illustrated"
    assert art["visual_genre"] == "stickman_illustrated"
    assert art["rendering_texture"] == "2D vector ink contours with flat color fills and ambient cel-shading"
    assert art["lighting_philosophy"] == preset.art_direction.lighting_philosophy
    assert art["primary_palette_hex"] == preset.art_direction.primary_palette_hex
    assert art["negative_anchors"] == preset.art_direction.negative_anchors

    # 2. Check narrative entity preservation (entities are valid EntityCatalog structures)
    entities = vb_data["entities"]
    assert "characters" in entities
    assert "environments" in entities
    assert "objects" in entities
    assert "concepts" in entities
    assert vb_data["topic_summary"] is not None


def test_legacy_video_bible_art_direction_untouched_when_style_is_none(db_session, test_channel, monkeypatch):
    """Verify that for legacy videos without a style preset, art direction is NOT overwritten by a preset."""
    monkeypatch.setattr(settings, "llm_provider", "mock")

    video = _create_test_video(
        db_session,
        test_channel,
        visual_style_id=None,
        visual_style_snapshot=None,
    )
    vb_step = next(s for s in video.steps if s.name == "video_bible")
    prev = {
        "script": {"script": SAMPLE_SCRIPT_DATA},
        "creative_direction": {"creative_direction": SAMPLE_CD_DATA},
    }

    output = generate_video_bible_step(video, prev=prev, step=vb_step)
    assert "video_bible" in output
    vb_data = output["video_bible"]

    # For legacy videos, mock LLM default art_direction remains intact
    assert vb_data["art_direction"] is not None
    # Preset-specific visual genres should NOT be stamped on legacy videos
    assert vb_data["art_direction"]["visual_genre"] != "stickman_illustrated"


# =========================================================================
# 5. Downstream Storyboard Integration Compatibility Test
# =========================================================================

def test_downstream_storyboard_compatibility_with_style_aware_video_bible():
    """Verify that VideoBibleOutput produced with style awareness is seamlessly accepted

    by storyboard prompt assembly and validation.
    """
    from app.steps.storyboard import build_storyboard_prompt
    from app.steps.storyboard_validation import validate_connected_storyboard

    preset = STICKMAN_TACTICAL_EXPLAINER_PRESET

    vb_output = VideoBibleOutput(
        bible_id="vb_stickman_test",
        topic_summary="Hypnic jerk explanation",
        entities=EntityCatalog(
            characters=[
                CharacterEntity(
                    id="char_stick_sleeper",
                    name="Stick Sleeper",
                    role="Protagonist",
                    visual_anchor="Minimalist white circular head stick figure with 3px black stroke",
                    apparel_anchor="Solid blue nightshirt",
                    forbidden_mutations=["no realistic face", "no hair spikes"],
                )
            ],
            environments=[
                EnvironmentEntity(
                    id="env_bedroom_ink",
                    name="Minimalist Bedroom",
                    spatial_layout="1-point perspective bedroom with ink contour walls",
                    lighting_setup="Directional lamp light casting soft flat shadow",
                    persistent_props=["Bed frame", "Clock"],
                    time_of_day="Night",
                )
            ],
        ),
        art_direction=ArtDirection(
            style_family=preset.art_direction.style_family,
            visual_genre=preset.art_direction.visual_genre,
            rendering_texture=preset.art_direction.rendering_texture,
            lighting_philosophy=preset.art_direction.lighting_philosophy,
            primary_palette_hex=list(preset.art_direction.primary_palette_hex),
            negative_anchors=list(preset.art_direction.negative_anchors),
        ),
    )

    # Build storyboard prompt passing this video bible
    sb_prompt = build_storyboard_prompt(
        rules_text="STORYBOARD_RULES",
        topic="Sleep Science",
        script_data=SAMPLE_SCRIPT_DATA,
        visual_style=preset.display_name,
        creative_direction_data=SAMPLE_CD_DATA,
        video_bible_data=vb_output.model_dump(),
    )

    assert "Approved Video Bible" in sb_prompt
    assert "char_stick_sleeper" in sb_prompt
    assert "env_bedroom_ink" in sb_prompt
    assert "- **Visual Genre**: stickman_illustrated" in sb_prompt
    assert "- **Rendering Texture**: 2D vector ink contours" in sb_prompt
