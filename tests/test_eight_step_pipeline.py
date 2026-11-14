"""Tests for Phase 1: 8-Step Pipeline State Machine & Legacy 6-Step Compatibility."""
import uuid
from contextlib import contextmanager
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.db import Base
from app.models import Channel, User, Video, VideoStep
from app.pipeline import (
    LEGACY_STEPS,
    STEPS,
    PipelineError,
    approve_step,
    claim_next_step,
    execute,
    init_steps,
    rerun_step,
)
from app.schemas import (
    CreativeDirectionOutput,
    EntityCatalog,
    StoryboardBeat,
    VideoBibleOutput,
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


def _create_test_video(session: Session, require_approval: bool = True, steps: list[str] | None = None) -> Video:
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

    video = Video(
        channel_id=channel.id,
        topic="Why Bridges Don't Collapse",
        description="Explaining bridge mechanics and load distribution",
    )
    session.add(video)
    session.flush()

    init_steps(session, video, steps=steps)
    return video


def test_eight_step_pipeline_initialization(db_session):
    """New videos initialize with all 8 steps in deterministic order with position 0..7."""
    video = _create_test_video(db_session)
    assert len(video.steps) == 8
    expected_steps = [
        "creative_direction",
        "script",
        "video_bible",
        "storyboard",
        "images",
        "voiceover",
        "render",
        "subtitles",
    ]
    for idx, expected_name in enumerate(expected_steps):
        assert video.steps[idx].name == expected_name
        assert video.steps[idx].position == idx
        assert video.steps[idx].status == "pending"
    assert video.status == "in_progress"


def test_eight_step_execution_and_gatekeeping(db_session, monkeypatch):
    """Test sequential execution, revisions, and approval gates across the 8-step pipeline."""
    video = _create_test_video(db_session, require_approval=True)

    @contextmanager
    def _mock_session_local():
        yield db_session

    monkeypatch.setattr("app.pipeline.SessionLocal", _mock_session_local)

    # 1. First step claimed must be creative_direction
    step0 = claim_next_step(db_session, video.id)
    assert step0.name == "creative_direction"
    assert step0.status == "running"

    # Execute creative_direction
    execute(video.id, "creative_direction", run_all=False)
    db_session.refresh(video)

    assert video.steps[0].status == "awaiting_approval"
    assert "creative_direction" in video.steps[0].output
    # Validate payload against Pydantic schema
    parsed_cd = CreativeDirectionOutput.model_validate(video.steps[0].output["creative_direction"])
    assert parsed_cd.format_archetype == "explainer_myth_bust"
    assert video.steps[0].revisions_count == 1

    # 2. Cannot claim next step until step0 is approved
    with pytest.raises(PipelineError, match="awaiting approval"):
        claim_next_step(db_session, video.id)

    # Approve creative_direction
    approve_step(db_session, video, "creative_direction")
    assert video.steps[0].status == "done"

    # 3. Next step claimed must be script
    step1 = claim_next_step(db_session, video.id)
    assert step1.name == "script"
    assert step1.status == "running"


def test_eight_step_rerun_invalidation(db_session):
    """Rerunning a step marks only downstream steps stale while preserving upstream steps."""
    video = _create_test_video(db_session, require_approval=True)

    # Mark first 4 steps done
    for idx in range(4):
        video.steps[idx].status = "done"
        video.steps[idx].output = {f"step_{idx}": True}
    video.output_path = "/storage/test/final.mp4"
    db_session.commit()

    # Rerun step 1: script
    rerun_step(db_session, video, "script", prompt="Make the hook more punchy")

    # Upstream step 0 (creative_direction) must remain done
    assert video.steps[0].status == "done"

    # Step 1 (script) must become pending with user_prompt set
    assert video.steps[1].status == "pending"
    assert video.steps[1].user_prompt == "Make the hook more punchy"

    # Downstream steps (video_bible pos 2, storyboard pos 3) must become stale
    assert video.steps[2].status == "stale"
    assert video.steps[3].status == "stale"
    # Unreached steps remain pending
    assert video.steps[4].status == "pending"
    assert video.output_path is None


def test_legacy_six_step_video_compatibility(db_session, monkeypatch):
    """Existing 6-step videos continue executing from their current steps without errors."""
    video = _create_test_video(db_session, require_approval=True, steps=LEGACY_STEPS)
    assert len(video.steps) == 6
    assert [s.name for s in video.steps] == LEGACY_STEPS

    # First step on legacy video is script
    step0 = claim_next_step(db_session, video.id)
    assert step0.name == "script"
    assert step0.status == "running"

    # Mark script done and storyboard done
    video.steps[0].status = "done"
    video.steps[0].output = {"script": {"narration": "Legacy script text"}}
    video.steps[1].status = "done"
    video.steps[1].output = {"storyboard": {"scenes": []}}
    db_session.commit()

    # Next step on legacy video is images (pos 2)
    next_step = claim_next_step(db_session, video.id)
    assert next_step.name == "images"
    assert next_step.status == "running"
    next_step.status = "done"
    db_session.commit()

    # Rerunning script on legacy video invalidates only downstream 6-step items
    rerun_step(db_session, video, "script", prompt="Rewrite script")
    assert video.steps[0].status == "pending"
    assert video.steps[1].status == "stale"
    assert video.steps[2].status == "stale"


def test_schemas_instantiation():
    """Verify that all new Phase 1 Pydantic models instantiate and validate correctly."""
    cd = CreativeDirectionOutput(
        format_archetype="explainer_myth_bust",
        storytelling_angle="why sleep fall occurs",
        central_thesis="Hypnic jerk is a motor transition twitch",
        tone="conversational",
        visual_genre="styled_3d",
        protagonist_config={"character_count": "single", "archetype_summary": "Alex the sleeper"},
        emotional_progression=[{"phase": "hook", "target_emotion": "curiosity", "visual_intent": "dark bedroom"}],
        creative_constraints=["Keep in bedroom"],
    )
    assert cd.visual_genre == "styled_3d"
    assert cd.protagonist_config.character_count == "single"

    vb = VideoBibleOutput(
        bible_id="bible_123",
        topic_summary="Hypnic jerk explanation",
        entities=EntityCatalog(
            characters=[{
                "id": "char_alex",
                "name": "Alex",
                "role": "Sleeper",
                "visual_anchor": "Young adult with tousled black hair",
                "apparel_anchor": "Charcoal hoodie",
                "forbidden_mutations": ["no glasses"],
            }],
            environments=[{
                "id": "env_bedroom",
                "name": "Bedroom",
                "spatial_layout": "Platform bed",
                "lighting_setup": "Amber salt lamp 2700K",
                "persistent_props": ["amber lamp"],
                "time_of_day": "night",
            }],
        ),
    )
    assert len(vb.entities.characters) == 1
    assert vb.entities.characters[0].id == "char_alex"
    assert len(vb.entities.environments) == 1

    beat = StoryboardBeat(
        beat_id="beat_01",
        scene_index=1,
        order=1,
        narration_slice="Why do we fall",
        duration_est_seconds=2.5,
        entity_bindings={"character_id": "char_alex", "environment_id": "env_bedroom"},
        transition_from_previous="establishing",
        continuity_intent="Opening establishing shot",
        camera={"shot_type": "medium_close_up", "camera_angle": "high_angle", "focal_point_description": "face on pillow", "movement_type": "push_in"},
        character_state={"pose_description": "lying on side", "expression_description": "drowsy", "action_in_progress": "blinking"},
        foley_cues=[{"foley_id": "sfx_clock", "offset_seconds": 0.2, "relative_volume": 0.5}],
    )
    assert beat.transition_from_previous == "establishing"
    assert beat.entity_bindings.character_id == "char_alex"
    assert len(beat.foley_cues) == 1
