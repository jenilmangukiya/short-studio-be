"""Tests for Phase 3: Connected Storyboard, Video Bible Entity Binding,

Prompt Composition, Continuity Validation, and Legacy Compatibility.
"""

from contextlib import contextmanager
import uuid
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.db import Base
from app.models import Channel, User, Video, VideoStep
from app.pipeline import (
    approve_step,
    claim_next_step,
    execute,
    init_steps,
)
from app.schemas import (
    ArtDirection,
    CameraDirection,
    CharacterEntity,
    CharacterState,
    ConceptEntity,
    CreativeDirectionOutput,
    EntityBindings,
    EntityCatalog,
    EnvironmentEntity,
    ObjectEntity,
    ProtagonistConfig,
    StoryboardBeat,
    StoryboardOutput,
    VideoBibleOutput,
)
from app.steps.prompt_compose import (
    COMPOSITION_RULES,
    MAX_FINAL_PROMPT_LENGTH,
    compose_final_prompt,
    recompute_final_prompts,
)
from app.steps.storyboard import (
    SceneInput,
    build_storyboard_prompt,
    generate_storyboard_step,
    split_script_for_mock,
)
from app.steps.storyboard_validation import validate_connected_storyboard


@pytest.fixture
def db_session():
    """Isolated in-memory SQLite session for tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _create_sample_bible():
    """Construct a representative Video Bible for the hypnic jerk topic."""
    return VideoBibleOutput(
        bible_id="vb_sleep_001",
        topic_summary="Neurological and evolutionary mechanics of hypnic jerks.",
        entities=EntityCatalog(
            characters=[
                CharacterEntity(
                    id="char_alex",
                    name="Alex",
                    role="protagonist",
                    visual_anchor="A 28-year-old lean man with short messy dark brown hair, subtle evening stubble",
                    apparel_anchor="charcoal gray soft cotton crewneck t-shirt",
                    forbidden_mutations=["hair color must remain dark brown", "no spectacles"],
                )
            ],
            environments=[
                EnvironmentEntity(
                    id="env_bedroom_night",
                    name="Cozy Modern Bedroom at Night",
                    spatial_layout="Minimalist modern bedroom, slate blue linen bedding, sheer curtains with soft city moonlight filtering through, bedside wooden table with warm amber nightlight glow",
                    lighting_setup="Low-key chiaroscuro, warm amber bedside glow against cool deep blue moonlight",
                    persistent_props=["bedside table", "slate blue pillows"],
                    time_of_day="night",
                )
            ],
            objects=[
                ObjectEntity(
                    id="obj_bedside_clock",
                    name="Minimalist Wooden LED Clock",
                    visual_description="Dark walnut block with soft warm-amber digital numerals displaying 11:42 PM",
                    material_and_finish="Matte walnut grain texture",
                )
            ],
            concepts=[
                ConceptEntity(
                    id="cpt_sleep_signal_diagram",
                    name="Dual-Circuit Motor Control Flow",
                    concept_explained="Motor cortex vs brainstem sleep gate competitive inhibition",
                    visual_analogy="Minimalist glowing schematic showing competition between Motor Cortex and Brainstem sleep gate against dark technical slate",
                    graphical_elements=["cyan wake vector", "violet sleep inhibition vector"],
                )
            ],
        ),
        art_direction=ArtDirection(
            style_family="cinematic",
            visual_genre="photorealistic_cinematic",
            lighting_philosophy="High-contrast evening chiaroscuro; warm amber focal light contrasting against deep midnight blue shadows",
            primary_palette_hex=["#0D1321", "#1D2D44", "#3E5C76", "#F4D06F"],
            rendering_texture="Subtle 35mm film grain, anamorphic shallow depth of field (f/1.8), organic dust motes in light beams",
            negative_anchors=["cartoon illustration", "bright daylight", "monsters", "harsh white fluorescent lighting", "low resolution artifacts"],
        ),
    )


def _create_sample_creative_direction(character_count="single"):
    return CreativeDirectionOutput(
        format_archetype="explainer_myth_bust",
        storytelling_angle="The brain misinterpreting muscle relaxation as a free fall",
        central_thesis="Hypnic jerks are benign sensorimotor signal glitches during sleep onset.",
        tone="curious, scientifically grounded, reassuring",
        visual_genre="photorealistic_cinematic",
        protagonist_config=ProtagonistConfig(
            character_count=character_count,
            archetype_summary="Relatable sleeper experiencing the sensation" if character_count != "none" else None,
        ),
        emotional_progression=[],
        creative_constraints=["No jump scares", "Do not depict medical seizures"],
    )


def _create_sample_script():
    return {
        "title": "Why You Feel Like Falling Before Sleep",
        "hook": "Have you ever felt like you were suddenly falling in bed?",
        "narration": (
            "Have you ever felt like you were suddenly falling in bed? "
            "You are drifting off when your body violently jerks awake. "
            "This universal sensation is called a hypnic jerk. "
            "Inside your brain, two competing motor systems are fighting for control. "
            "As your muscles relax too quickly, the brainstem misinterprets the sudden drop in tension as an actual physical fall. "
            "So it fires an emergency jolt to protect you before you safely return to sleep."
        ),
    }


# =========================================================================
# 1. Connected Prompt Composition Tests
# =========================================================================

def test_compose_prompt_physical_scene_with_character_and_environment():
    """Physical bedroom scene includes character, pose, environment, and bound props."""
    bible = _create_sample_bible()
    cd = _create_sample_creative_direction(character_count="single")

    scene = {
        "entity_bindings": {
            "character_id": "char_alex",
            "environment_id": "env_bedroom_night",
            "object_ids": ["obj_bedside_clock"],
            "concept_id": None,
        },
        "camera": {
            "shot_type": "medium_close_up",
            "camera_angle": "low_angle",
            "focal_point_description": "Alex's eyelids fluttering shut on pillow",
            "movement_type": "push_in",
        },
        "character_state": {
            "pose_description": "Lying supine in bed resting head on slate-blue pillow",
            "expression_description": "Drowsy, relaxed eyelids half-closed",
            "action_in_progress": "Drifting off to sleep",
        },
        "transition_from_previous": "establishing",
    }

    prompt = compose_final_prompt(scene, video_bible=bible, creative_direction=cd)

    # 1. Art direction and genre present
    assert "Photorealistic cinematic visual style" in prompt
    assert "35mm film grain" in prompt

    # 2. Bound character present with attire and state
    assert "Alex" in prompt
    assert "charcoal gray soft cotton crewneck t-shirt" in prompt
    assert "Lying supine in bed" in prompt
    assert "Drowsy, relaxed" in prompt

    # 3. Bound environment present
    assert "Cozy Modern Bedroom at Night" in prompt
    assert "slate blue linen bedding" in prompt

    # 4. Bound prop present
    assert "Minimalist Wooden LED Clock" in prompt

    # 5. Diagram concept NOT present
    assert "Dual-Circuit Motor Control Flow" not in prompt

    # 6. Camera direction present
    assert "medium close up shot" in prompt
    assert "low angle" in prompt

    # 7. Composition rules and length ceiling
    assert COMPOSITION_RULES in prompt
    assert len(prompt) <= MAX_FINAL_PROMPT_LENGTH


def test_compose_prompt_standalone_conceptual_diagram():
    """Standalone diagram scene includes concept and style, but NO bedroom, NO Alex, NO clock."""
    bible = _create_sample_bible()
    cd = _create_sample_creative_direction(character_count="single")

    diagram_scene = {
        "entity_bindings": {
            "character_id": None,
            "environment_id": None,
            "object_ids": [],
            "concept_id": "cpt_sleep_signal_diagram",
        },
        "camera": {
            "shot_type": "medium",
            "camera_angle": "eye_level",
            "focal_point_description": "Synaptic signal contest between Motor Cortex and Brainstem",
            "movement_type": "static",
        },
        "character_state": None,
        "environment_state_override": None,
        "transition_from_previous": "explanatory_cutaway",
        "visual_purpose": "Explain competing neurological sleep signals",
    }

    prompt = compose_final_prompt(diagram_scene, video_bible=bible, creative_direction=cd)

    # 1. Concept is present
    assert "Dual-Circuit Motor Control Flow" in prompt
    assert "Minimalist glowing schematic" in prompt

    # 2. Cutaway framing context is present
    assert "explanatory cutaway transition" in prompt

    # 3. CRITICAL: Physical environment and human entities are NOT injected
    assert "Alex" not in prompt
    assert "charcoal gray" not in prompt
    assert "bedroom" not in prompt.lower() or "no bedroom" in prompt.lower()
    assert "bedside clock" not in prompt.lower()

    # 4. Color palette and composition rules present
    assert "Color palette:" in prompt
    assert COMPOSITION_RULES in prompt


def test_compose_prompt_character_free_story():
    """When creative direction specifies character_count='none', prompts never include human subjects."""
    bible = _create_sample_bible()
    cd = _create_sample_creative_direction(character_count="none")

    scene = {
        "entity_bindings": {
            "character_id": None,
            "environment_id": "env_bedroom_night",
            "object_ids": ["obj_bedside_clock"],
            "concept_id": None,
        },
        "camera": {
            "shot_type": "close_up",
            "camera_angle": "eye_level",
            "focal_point_description": "Glowing digital numbers on dark walnut nightstand",
        },
        "character_state": None,
        "transition_from_previous": "establishing",
    }

    prompt = compose_final_prompt(scene, video_bible=bible, creative_direction=cd)
    assert "Alex" not in prompt
    assert "Subject: " not in prompt
    assert "Minimalist Wooden LED Clock" in prompt
    assert "Cozy Modern Bedroom at Night" in prompt


# =========================================================================
# 2. Continuity and Story Progression Validation Tests
# =========================================================================

def test_validation_passes_on_coherent_storyboard():
    """Valid connected storyboard produces 0 continuity warnings."""
    bible = _create_sample_bible()
    cd = _create_sample_creative_direction(character_count="single")

    scenes = [
        {
            "index": 1,
            "entity_bindings": {"character_id": "char_alex", "environment_id": "env_bedroom_night"},
            "camera": {"shot_type": "wide", "camera_angle": "eye_level"},
            "transition_from_previous": "establishing",
        },
        {
            "index": 2,
            "entity_bindings": {"character_id": "char_alex", "environment_id": "env_bedroom_night"},
            "camera": {"shot_type": "medium", "camera_angle": "low_angle"},
            "transition_from_previous": "same_scene_reframe",
        },
        {
            "index": 3,
            "entity_bindings": {"character_id": "char_alex", "environment_id": "env_bedroom_night"},
            "camera": {"shot_type": "close_up", "camera_angle": "eye_level"},
            "transition_from_previous": "action_reaction",
        },
        {
            "index": 4,
            "entity_bindings": {"concept_id": "cpt_sleep_signal_diagram"},
            "camera": {"shot_type": "medium", "camera_angle": "eye_level"},
            "transition_from_previous": "explanatory_cutaway",
        },
        {
            "index": 5,
            "entity_bindings": {"character_id": "char_alex", "environment_id": "env_bedroom_night"},
            "camera": {"shot_type": "wide", "camera_angle": "high_angle"},
            "transition_from_previous": "resolution_return",
        },
    ]

    warnings = validate_connected_storyboard(scenes, video_bible=bible, creative_direction=cd)
    assert len(warnings) == 0


def test_validation_detects_unknown_entity_ids():
    """Referencing entity IDs not in Video Bible flags explicit warnings."""
    bible = _create_sample_bible()

    scenes = [
        {
            "index": 1,
            "entity_bindings": {
                "character_id": "char_ghost",
                "environment_id": "env_submarine",
                "object_ids": ["obj_laser_gun"],
                "concept_id": "cpt_unknown_math",
            },
            "camera": {"shot_type": "wide"},
            "transition_from_previous": "establishing",
        }
    ]

    warnings = validate_connected_storyboard(scenes, video_bible=bible)
    assert any("char_ghost" in w for w in warnings)
    assert any("env_submarine" in w for w in warnings)
    assert any("obj_laser_gun" in w for w in warnings)
    assert any("cpt_unknown_math" in w for w in warnings)


def test_validation_detects_unmotivated_location_hop():
    """Switching environments under same_scene_reframe triggers a transition warning."""
    bible = _create_sample_bible()

    # Add a second environment to the catalog
    bible.entities.environments.append(
        EnvironmentEntity(
            id="env_kitchen",
            name="Bright Morning Kitchen",
            spatial_layout="Tile countertops with sunlight",
            lighting_setup="Morning sunlight",
        )
    )

    scenes = [
        {
            "index": 1,
            "entity_bindings": {"environment_id": "env_bedroom_night"},
            "camera": {"shot_type": "wide"},
            "transition_from_previous": "establishing",
        },
        {
            "index": 2,
            # Jumped to kitchen without location_change or cutaway!
            "entity_bindings": {"environment_id": "env_kitchen"},
            "camera": {"shot_type": "medium"},
            "transition_from_previous": "same_scene_reframe",
        },
    ]

    warnings = validate_connected_storyboard(scenes, video_bible=bible)
    assert any("switches environment" in w.lower() for w in warnings)


def test_validation_detects_framing_monotony():
    """Repeating the exact same shot framing 3 times in a row triggers variety warning."""
    bible = _create_sample_bible()

    scenes = [
        {"index": 1, "camera": {"shot_type": "medium"}, "transition_from_previous": "establishing"},
        {"index": 2, "camera": {"shot_type": "medium"}, "transition_from_previous": "same_scene_reframe"},
        {"index": 3, "camera": {"shot_type": "medium"}, "transition_from_previous": "action_reaction"},
    ]

    warnings = validate_connected_storyboard(scenes, video_bible=bible)
    assert any("repeat the same camera shot type 'medium'" in w for w in warnings)


# =========================================================================
# 3. Storyboard Pipeline Step Integration Tests
# =========================================================================

def test_generate_storyboard_step_consumes_all_three_upstream_outputs(db_session, monkeypatch):
    """Storyboard generation successfully ingests script, creative direction, and video bible."""
    monkeypatch.setattr(settings, "llm_provider", "mock")

    user = User(email=f"tester-{uuid.uuid4().hex[:6]}@example.com", password_hash="fake")
    db_session.add(user)
    db_session.flush()

    channel = Channel(user_id=user.id, name="NeuroScience", settings={"visual_style": "Cinematic evening realism"})
    db_session.add(channel)
    db_session.flush()

    video = Video(
        channel_id=channel.id,
        topic="Why Do We Feel Like We're Falling Asleep?",
        description="Hypnic jerk explanation",
    )
    db_session.add(video)
    db_session.flush()
    init_steps(db_session, video)

    step = video.steps[3]  # storyboard step

    bible = _create_sample_bible()
    cd = _create_sample_creative_direction(character_count="single")
    script = _create_sample_script()

    prev = {
        "creative_direction": {"creative_direction": cd.model_dump()},
        "script": {"script": script},
        "video_bible": {"video_bible": bible.model_dump()},
    }

    output = generate_storyboard_step(video, prev=prev, step=step)

    assert "storyboard" in output
    sb = output["storyboard"]
    assert "scenes" in sb
    assert len(sb["scenes"]) >= 5
    assert "beats" in sb
    assert len(sb["beats"]) >= 5

    # Check that scenes have composed prompts and bound entities
    scene1 = sb["scenes"][0]
    assert "final_prompt" in scene1
    assert COMPOSITION_RULES in scene1["final_prompt"]
    assert scene1["entity_bindings"] is not None

    # Check flat StoryboardBeat models
    parsed_beats = [StoryboardBeat.model_validate(b) for b in sb["beats"]]
    assert len(parsed_beats) >= 5
    assert parsed_beats[0].transition_from_previous == "establishing"


def test_legacy_six_step_storyboard_compatibility(db_session, monkeypatch):
    """Legacy 6-step videos without creative_direction or video_bible run smoothly without errors."""
    monkeypatch.setattr(settings, "llm_provider", "mock")

    user = User(email=f"legacy-{uuid.uuid4().hex[:6]}@example.com", password_hash="fake")
    db_session.add(user)
    db_session.flush()

    channel = Channel(user_id=user.id, name="LegacyChannel")
    db_session.add(channel)
    db_session.flush()

    video = Video(channel_id=channel.id, topic="Ancient Bridges")
    db_session.add(video)
    db_session.flush()

    legacy_steps = ["script", "storyboard", "voiceover", "images", "render", "subtitles"]
    init_steps(db_session, video, steps=legacy_steps)

    step = video.steps[1]  # storyboard step in 6-step flow

    prev = {
        "script": {
            "script": {
                "title": "Ancient Stone Bridges",
                "hook": "Why have Roman bridges lasted two thousand years?",
                "narration": "Why have Roman bridges lasted two thousand years? The secret is the keystone and volcanic ash concrete.",
            }
        }
    }

    output = generate_storyboard_step(video, prev=prev, step=step)
    assert "storyboard" in output
    assert len(output["storyboard"]["scenes"]) >= 1
    assert "final_prompt" in output["storyboard"]["scenes"][0]
    assert COMPOSITION_RULES in output["storyboard"]["scenes"][0]["final_prompt"]


# =========================================================================
# 5. Character Prohibition & Scientific Uncertainty Tests
# =========================================================================

def test_character_cannot_be_introduced_when_character_count_none():
    """When character_count='none', introducing a character is flagged and stripped from prompt."""
    bible = _create_sample_bible()
    cd = _create_sample_creative_direction(character_count="none")

    # An invalid scene attempting to bind a character despite character_count='none'
    invalid_scene = {
        "index": 1,
        "entity_bindings": {
            "character_id": "char_alex",
            "environment_id": "env_bedroom_night",
        },
        "character_state": {
            "pose_description": "Sitting upright",
            "expression_description": "Surprised",
            "action_in_progress": "Awakening",
        },
        "camera": {"shot_type": "medium"},
        "transition_from_previous": "establishing",
    }

    # 1. Validation must flag the forbidden character binding and character_state
    warnings = validate_connected_storyboard([invalid_scene], video_bible=bible, creative_direction=cd)
    assert any("character_count='none'" in w for w in warnings)
    assert any("no character is permitted" in w for w in warnings)

    # 2. Prompt composition must strictly strip the character
    prompt = compose_final_prompt(invalid_scene, video_bible=bible, creative_direction=cd)
    assert "Alex" not in prompt
    assert "Sitting upright" not in prompt
    assert "Surprised" not in prompt
    assert "Cozy Modern Bedroom at Night" in prompt


def test_character_not_in_video_bible_rejected():
    """Referencing a character not in the Video Bible entity catalog is flagged and excluded."""
    bible = _create_sample_bible()
    cd = _create_sample_creative_direction(character_count="single")

    scene_with_unregistered_char = {
        "index": 1,
        "entity_bindings": {
            "character_id": "char_unregistered_person",
            "environment_id": "env_bedroom_night",
        },
        "camera": {"shot_type": "close_up"},
        "transition_from_previous": "establishing",
    }

    # 1. Validation flags the nonexistent character ID
    warnings = validate_connected_storyboard([scene_with_unregistered_char], video_bible=bible, creative_direction=cd)
    assert any("char_unregistered_person" in w and "does not exist in the Video Bible" in w for w in warnings)

    # 2. Prompt composition excludes the unregistered character
    prompt = compose_final_prompt(scene_with_unregistered_char, video_bible=bible, creative_direction=cd)
    assert "char_unregistered_person" not in prompt
    assert "Subject: char_unregistered_person" not in prompt


def test_build_storyboard_prompt_enforces_character_prohibition_and_scientific_uncertainty():
    """Prompt builder injects explicit rules for character_count='none' and scientific uncertainty preservation."""
    bible = _create_sample_bible()
    cd_none = _create_sample_creative_direction(character_count="none")
    script = _create_sample_script()

    rules_text = "Base Storyboard Rules"
    prompt_text = build_storyboard_prompt(
        rules_text=rules_text,
        topic="Hypnic Jerks",
        script_data=script,
        visual_style="Cinematic",
        creative_direction_data=cd_none.model_dump(),
        video_bible_data=bible.model_dump(),
    )

    # 1. Strict character prohibition rule injected
    assert "CRITICAL CHARACTER RULE" in prompt_text
    assert "Character count is NONE" in prompt_text
    assert "MUST NOT introduce, invent, or name any protagonist, person, or character" in prompt_text

    # 2. Scientific uncertainty preservation rule injected
    assert "SCIENTIFIC ACCURACY & UNCERTAINTY RULE" in prompt_text
    assert "Preserve scientific uncertainty from the approved script" in prompt_text
    assert "DO NOT turn it into an established, absolute fact" in prompt_text

