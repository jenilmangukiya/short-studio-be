"""Tests for Phase 2: Creative Direction and Video Bible implementation."""
import json
import uuid
from contextlib import contextmanager
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.db import Base
from app.models import Channel, User, Video, VideoStep
from app.pipeline import (
    PipelineError,
    approve_step,
    claim_next_step,
    execute,
    init_steps,
)
from app.schemas import (
    ArtDirection,
    CharacterEntity,
    CreativeDirectionOutput,
    EmotionalBeat,
    EntityCatalog,
    EnvironmentEntity,
    ProtagonistConfig,
    VideoBibleOutput,
)
from app.steps.creative_direction import (
    build_creative_direction_prompt,
    generate_creative_direction_step,
    load_prompt_template as load_cd_prompt,
    validate_creative_direction,
)
from app.steps.video_bible import (
    build_video_bible_prompt,
    generate_video_bible_step,
    load_prompt_template as load_vb_prompt,
    validate_video_bible,
)
from app.steps.script import (
    ScriptOutput,
    build_script_prompt,
    validate_script_soft_checks,
)


@pytest.fixture
def db_session():
    """Isolated in-memory SQLite session for tests."""
    test_engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(test_engine)
    with Session(test_engine) as session:
        yield session


def _create_test_video(session: Session, topic="Why Do We Feel Like We're Falling Asleep?", steps: list[str] | None = None) -> Video:
    user = User(email=f"tester-{uuid.uuid4().hex[:6]}@example.com", password_hash="fake")
    session.add(user)
    session.flush()

    channel = Channel(
        user_id=user.id,
        name="SleepScience",
        niche="Science & Curiosity",
        settings={"require_approval": True, "strict_approval": True},
    )
    session.add(channel)
    session.flush()

    video = Video(
        channel_id=channel.id,
        topic=topic,
        description="Explaining hypnic jerks and the brain's sensory misinterpretations.",
    )
    session.add(video)
    session.flush()
    init_steps(session, video, steps=steps)
    return video


# =========================================================================
# 1. Creative Direction Unit Tests
# =========================================================================

def test_load_creative_direction_prompt_template():
    rules_text, prompt_version = load_cd_prompt()
    assert len(rules_text) > 100
    assert prompt_version.startswith("cd-rules-")
    assert "Format Archetypes" in rules_text or "archetype" in rules_text.lower()


def test_build_creative_direction_prompt():
    prompt = build_creative_direction_prompt(
        rules_text="Rules here",
        topic="The Sleep Mystery",
        description="Detailed description",
        channel_name="SleepDoc",
        channel_niche="Neuroscience",
        channel_settings={"visual_style": "cinematic_dark"},
        revision_instruction="Make it more surreal",
        previous_creative_direction={"format_archetype": "deep_dive_case"},
    )
    assert "The Sleep Mystery" in prompt
    assert "SleepDoc" in prompt
    assert "Neuroscience" in prompt
    assert "REVISION MODE" in prompt
    assert "Make it more surreal" in prompt


def test_creative_direction_validation_soft_checks():
    # Complete valid output -> no warnings
    valid_cd = CreativeDirectionOutput(
        format_archetype="explainer_myth_bust",
        storytelling_angle="Debunking the falling myth",
        central_thesis="Hypnic jerks are benign misfires.",
        tone="engaging",
        visual_genre="photorealistic_cinematic",
        protagonist_config=ProtagonistConfig(character_count="none"),
        emotional_progression=[
            EmotionalBeat(phase="hook", target_emotion="Curiosity", visual_intent="Macro close-up"),
            EmotionalBeat(phase="complication", target_emotion="Intrigue", visual_intent="Sudden tilt"),
            EmotionalBeat(phase="resolution", target_emotion="Revelation", visual_intent="Wide calm lighting"),
        ],
        creative_constraints=["No horror jump scares", "Keep medical terms simple"],
    )
    warnings = validate_creative_direction(valid_cd)
    assert len(warnings) == 0

    # Incomplete protagonist config -> warning
    bad_protagonist = CreativeDirectionOutput(
        format_archetype="explainer_myth_bust",
        storytelling_angle="Angle",
        central_thesis="Thesis",
        tone="tone",
        visual_genre="styled_3d",
        protagonist_config=ProtagonistConfig(character_count="single", archetype_summary=None),
        emotional_progression=[],
        creative_constraints=[],
    )
    warnings2 = validate_creative_direction(bad_protagonist)
    assert any("archetype summary" in w for w in warnings2)
    assert any("emotional progression" in w.lower() for w in warnings2)
    assert any("creative constraints" in w.lower() for w in warnings2)


def test_generate_creative_direction_step_mock(db_session, monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    video = _create_test_video(db_session)
    step = video.steps[0]

    output = generate_creative_direction_step(video, prev={}, step=step)
    assert "creative_direction" in output
    assert "meta" in output
    assert output["meta"]["task"] == "creative_direction"
    cd = CreativeDirectionOutput.model_validate(output["creative_direction"])
    assert cd.format_archetype is not None


def test_generate_creative_direction_missing_topic(db_session, monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    video = _create_test_video(db_session)
    video.topic = ""
    step = video.steps[0]

    with pytest.raises(PipelineError, match="topic is required"):
        generate_creative_direction_step(video, prev={}, step=step)


# =========================================================================
# 2. Video Bible Unit Tests
# =========================================================================

def test_load_video_bible_prompt_template():
    rules_text, prompt_version = load_vb_prompt()
    assert len(rules_text) > 100
    assert prompt_version.startswith("vb-rules-")
    assert "Entity Catalog" in rules_text or "entity" in rules_text.lower()


def test_build_video_bible_prompt():
    prompt = build_video_bible_prompt(
        rules_text="Bible rules",
        topic="Falling Asleep",
        script_data={"narration": "Wait, why does your body jerk awake?", "title": "Falling Asleep", "hook": "Wait, why?"},
        creative_direction_data={"format_archetype": "explainer_myth_bust", "storytelling_angle": "Hypnic reflex"},
        previous_bible={"bible_id": "vb_1"},
        revision_instruction="Add an antique brass alarm clock to objects",
    )
    assert "Bible rules" in prompt
    assert "Wait, why does your body jerk awake?" in prompt
    assert "explainer_myth_bust" in prompt
    assert "REVISION MODE" in prompt
    assert "antique brass alarm clock" in prompt


def test_video_bible_precondition_missing_script(db_session, monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    video = _create_test_video(db_session)
    step = video.steps[2]  # video_bible

    with pytest.raises(PipelineError, match="Script step must be completed"):
        generate_video_bible_step(video, prev={}, step=step)


def test_generate_video_bible_step_mock(db_session, monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    video = _create_test_video(db_session)
    step = video.steps[2]  # video_bible

    prev = {
        "creative_direction": {
            "creative_direction": {
                "format_archetype": "explainer_myth_bust",
                "storytelling_angle": "Debunking hypnic jerk misconceptions",
                "central_thesis": "Sensory signal confusion during sleep onset.",
                "tone": "engaging",
                "visual_genre": "photorealistic_cinematic",
                "protagonist_config": {"character_count": "none"},
                "emotional_progression": [],
                "creative_constraints": ["No sudden flashes"],
            }
        },
        "script": {
            "script": {
                "title": "Why You Twitch Before Sleep",
                "hook": "Have you ever felt like you were suddenly falling in bed?",
                "narration": "Have you ever felt like you were suddenly falling in bed? Your brain thinks muscles went slack.",
            }
        },
    }

    output = generate_video_bible_step(video, prev=prev, step=step)
    assert "video_bible" in output
    assert "meta" in output
    assert output["meta"]["task"] == "video_bible"
    vb = VideoBibleOutput.model_validate(output["video_bible"])
    assert vb.entities is not None
    assert vb.art_direction is not None


def test_video_bible_validation_soft_checks():
    # Protagonist count > 0 but 0 characters defined -> warning
    cd_dict = {
        "protagonist_config": {"character_count": "single"},
    }
    vb = VideoBibleOutput.model_validate({
        "bible_id": "vb_test",
        "topic_summary": "Sleep science exploration",
        "entities": {
            "characters": [],
            "environments": [],
            "objects_and_props": [],
            "concepts_and_diagrams": [],
        },
        "art_direction": {
            "style_family": "cinematic",
            "visual_genre": "photorealistic_cinematic",
            "lighting_philosophy": "dim moody amber",
            "primary_palette_hex": ["#1A1A24"],
            "rendering_texture": "film grain",
        },
    })
    warnings = validate_video_bible(vb, cd_dict)
    assert any("creative direction specified characters" in w.lower() for w in warnings)
    assert any("no environments defined" in w.lower() for w in warnings)


# =========================================================================
# 3. Script Epistemic Calibration & Creative Direction Ingestion Tests
# =========================================================================

def test_script_build_prompt_includes_creative_direction():
    cd_data = {
        "format_archetype": "investigative_mystery",
        "storytelling_angle": "The evolutionary origin of the twitch",
        "central_thesis": "An ancient primate tree-falling reflex.",
        "tone": "gripping and scientific",
        "creative_constraints": ["Emphasize scientific uncertainty", "No pseudoscientific cures"],
    }
    prompt = build_script_prompt(
        rules_text="Write a short video script.",
        topic="Falling Reflex",
        description="Explaining hypnic jerks",
        creative_direction_data=cd_data,
    )
    assert "Approved Creative Direction" in prompt
    assert "investigative_mystery" in prompt
    assert "An ancient primate tree-falling reflex" in prompt
    assert "Emphasize scientific uncertainty" in prompt


def test_script_epistemic_calibration_soft_checks():
    # Overclaiming / dogmatic language flags soft warnings without raising exceptions
    script_with_dogmatic_claim = ScriptOutput(
        title="The Sleep Twitch Secret",
        hook="Why do you twitch when falling asleep?",
        narration=(
            "Why do you twitch when falling asleep? It is a proven fact that aliens control brain frequencies "
            "during your transition into deep sleep. While traditional neurologists claim muscles simply relax too fast, "
            "this discovery completely rewrites everything we thought we understood about human consciousness and dreams."
        ),
        angle="myth vs reality",
        claims_to_verify=["Alien brain frequency control"],
        hashtags=["#science", "#sleep"],
    )
    warnings = validate_script_soft_checks(script_with_dogmatic_claim, None, None)
    assert any("epistemic calibration" in w.lower() or "dogmatic" in w.lower() for w in warnings)


# =========================================================================
# 4. Multi-Step Pipeline Sequential Execution
# =========================================================================

def test_eight_step_sequential_creative_script_bible_flow(db_session, monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    video = _create_test_video(db_session, topic="Why Do Cats Purr?")

    @contextmanager
    def _mock_session_local():
        yield db_session

    monkeypatch.setattr("app.pipeline.SessionLocal", _mock_session_local)

    # 1. Claim & Execute Creative Direction
    step0 = claim_next_step(db_session, video.id)
    assert step0.name == "creative_direction"
    execute(video.id, "creative_direction", run_all=False)
    db_session.refresh(video)
    assert video.steps[0].status == "awaiting_approval"
    assert "creative_direction" in video.steps[0].output
    approve_step(db_session, video, "creative_direction")
    assert video.steps[0].status == "done"

    # 2. Claim & Execute Script
    step1 = claim_next_step(db_session, video.id)
    assert step1.name == "script"
    execute(video.id, "script", run_all=False)
    db_session.refresh(video)
    assert video.steps[1].status == "awaiting_approval"
    assert "script" in video.steps[1].output
    approve_step(db_session, video, "script")
    assert video.steps[1].status == "done"

    # 3. Claim & Execute Video Bible
    step2 = claim_next_step(db_session, video.id)
    assert step2.name == "video_bible"
    execute(video.id, "video_bible", run_all=False)
    db_session.refresh(video)
    assert video.steps[2].status == "awaiting_approval"
    assert "video_bible" in video.steps[2].output
    vb_out = VideoBibleOutput.model_validate(video.steps[2].output["video_bible"])
    assert vb_out.entities is not None
    approve_step(db_session, video, "video_bible")
    assert video.steps[2].status == "done"


# =========================================================================
# 5. Legacy 6-Step Compatibility
# =========================================================================

def test_legacy_six_step_script_executes_without_creative_direction(db_session, monkeypatch):
    """Ensure legacy 6-step videos still run their script step cleanly when creative_direction is absent."""
    monkeypatch.setattr(settings, "llm_provider", "mock")

    legacy_step_names = ["script", "storyboard", "voiceover", "images", "render", "subtitles"]
    video = _create_test_video(db_session, topic="Legacy Ancient Wonders", steps=legacy_step_names)

    @contextmanager
    def _mock_session_local():
        yield db_session

    monkeypatch.setattr("app.pipeline.SessionLocal", _mock_session_local)

    step0 = claim_next_step(db_session, video.id)
    assert step0.name == "script"
    execute(video.id, "script", run_all=False)
    db_session.refresh(video)
    assert video.steps[0].status == "awaiting_approval"
    assert "script" in video.steps[0].output
