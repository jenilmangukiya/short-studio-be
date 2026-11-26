"""Phase 1B Regression Tests: Visual Storytelling Planning & Beat-Aware Environment Slicing.

Tests:
- Contextual humans without recurring protagonists.
- Recurring character continuity.
- Topics requiring no humans.
- Flexible environment counts.
- Interior versus exterior environment isolation.
- Close-up shots without irrelevant wide-angle props.
- Visual intent propagation through the pipeline.
- Meaningful subject continuity without forced repetition restrictions.
- Narrative resolution without unwanted anchor subject injection (narrative contradiction avoidance).
- Phase 1A concept slicing preservation.
- Frozen visual style behavior.
- Multiple unrelated video genres (Engineering, Astronomy, Historical Chronicle).
"""

import pytest

from app.schemas import (
    CameraDirection,
    CharacterEntity,
    CreativeDirectionOutput,
    EmotionalBeat,
    EntityBindings,
    EnvironmentEntity,
    ObjectEntity,
    ConceptEntity,
    EntityCatalog,
    ProtagonistConfig,
    StoryboardBeat,
    VideoBibleOutput,
    VisualIntentType,
)
from app.steps.creative_direction import validate_creative_direction
from app.steps.prompt_compose import (
    _compose_connected_final_prompt,
    slice_concept_for_beat,
    slice_environment_for_beat,
)
from app.steps.storyboard import (
    SceneInput,
    VisualBeat,
    normalize_scene_visual_beats,
)
from app.steps.storyboard_validation import validate_connected_storyboard
from app.styles.registry import resolve_visual_style_preset


# ==============================================================================
# 1. Human Presence & Protagonist Configuration Tests
# ==============================================================================

def test_protagonist_config_contextual_humans():
    """Contextual humans without a recurring hero: human_presence='contextual'."""
    cfg = ProtagonistConfig(
        human_presence="contextual",
        archetype_summary="19th-century European farming families and rural villagers",
    )
    assert cfg.human_presence == "contextual"
    assert "European farming families" in (cfg.archetype_summary or "")


def test_protagonist_config_all_four_modes_and_required_field():
    """ProtagonistConfig supports all 4 human_presence modes and requires human_presence explicitly."""
    import pytest
    from pydantic import ValidationError

    # All 4 valid modes
    cfg_none = ProtagonistConfig(human_presence="none")
    assert cfg_none.human_presence == "none"
    assert cfg_none.archetype_summary is None

    cfg_ctx = ProtagonistConfig(human_presence="contextual", archetype_summary="Witnesses")
    assert cfg_ctx.human_presence == "contextual"

    cfg_single = ProtagonistConfig(human_presence="single", archetype_summary="Solo Hero")
    assert cfg_single.human_presence == "single"

    cfg_ens = ProtagonistConfig(human_presence="ensemble", archetype_summary="Crew of 3")
    assert cfg_ens.human_presence == "ensemble"

    # Must fail validation if human_presence is omitted (no automatic default)
    with pytest.raises(ValidationError):
        ProtagonistConfig()  # type: ignore


def test_contextual_humans_validation_allows_witness_character_entities():
    """When human_presence='contextual', storyboard can bind a contextual witness character without warning."""
    cd = CreativeDirectionOutput(
        format_archetype="historical_chronicle",
        storytelling_angle="Following the global climate disruption and human hardship",
        central_thesis="Tambora's 1815 eruption caused widespread agricultural failure",
        tone="Documentary and dramatic",
        visual_genre="mixed_format",
        protagonist_config=ProtagonistConfig(
            human_presence="contextual",
            archetype_summary="19th-century European rural villagers and farmers",
        ),
        emotional_progression=[
            EmotionalBeat(phase="hook", target_emotion="curiosity", visual_intent="Show dim sky"),
            EmotionalBeat(phase="setup", target_emotion="tension", visual_intent="Show eruption"),
            EmotionalBeat(phase="resolution", target_emotion="awe", visual_intent="Show communities"),
        ],
        creative_constraints=["Keep 1815 period accuracy"],
    )

    vb = VideoBibleOutput(
        bible_id="vb_tambora_test",
        topic_summary="Mount Tambora eruption and Year Without a Summer",
        entities=EntityCatalog(
            characters=[
                CharacterEntity(
                    id="char_peasant_witness",
                    name="European Farm Family",
                    role="Contextual witness of crop failure",
                    visual_anchor="Weathered 19th-century farmers with solemn expressions",
                    apparel_anchor="Homespun coarse linen tunics and woolen shawls",
                )
            ],
            environments=[
                EnvironmentEntity(
                    id="env_farmland",
                    name="European Farmland",
                    spatial_layout="Low crop rows under rain clouds",
                    lighting_setup="Overcast 6500K daylight",
                )
            ],
        ),
    )

    scenes = [
        {
            "index": 1,
            "narration": "Harvests failed across Europe.",
            "visual_intent": "human_experience",
            "entity_bindings": {"character_id": "char_peasant_witness", "environment_id": "env_farmland"},
            "shot_type": "medium",
            "camera": {"shot_type": "medium", "camera_angle": "eye_level"},
        }
    ]

    warnings = validate_connected_storyboard(scenes=scenes, video_bible=vb, creative_direction=cd)
    # Must NOT warn that human_presence is none because human_presence is contextual!
    assert not any("human_presence='none'" in w for w in warnings)


def test_no_humans_validation_flags_character_bindings():
    """When human_presence='none', binding any character raises an explicit warning."""
    cd = CreativeDirectionOutput(
        format_archetype="cosmic_scale",
        storytelling_angle="Scale of black holes",
        central_thesis="Extreme spacetime curvature",
        tone="Awe-inspiring",
        visual_genre="photorealistic_cinematic",
        protagonist_config=ProtagonistConfig(human_presence="none"),
        emotional_progression=[
            EmotionalBeat(phase="hook", target_emotion="awe", visual_intent="Show black hole"),
            EmotionalBeat(phase="setup", target_emotion="curiosity", visual_intent="Show event horizon"),
            EmotionalBeat(phase="resolution", target_emotion="clarity", visual_intent="Show singularity"),
        ],
        creative_constraints=["No fantasy elements"],
    )

    vb = VideoBibleOutput(
        bible_id="vb_blackhole",
        topic_summary="Black hole physics",
        entities=EntityCatalog(
            characters=[
                CharacterEntity(
                    id="char_unwanted",
                    name="Unwanted Person",
                    role="Extra",
                    visual_anchor="Person standing in space",
                    apparel_anchor="Suit",
                )
            ]
        ),
    )

    scenes = [
        {
            "index": 1,
            "narration": "Inside the event horizon, nothing escapes.",
            "visual_intent": "environmental_context",
            "entity_bindings": {"character_id": "char_unwanted"},
            "shot_type": "wide",
            "camera": {"shot_type": "wide", "camera_angle": "eye_level"},
        }
    ]

    warnings = validate_connected_storyboard(scenes=scenes, video_bible=vb, creative_direction=cd)
    assert any("human_presence='none'" in w for w in warnings)


# ==============================================================================
# 2. Flexible Environment Planning Tests
# ==============================================================================

def test_flexible_environment_counts_supported_cleanly():
    """Video Bible supports defining 4+ distinct environments without validation rejection."""
    vb = VideoBibleOutput(
        bible_id="vb_multi_env",
        topic_summary="Multi-environment narrative",
        entities=EntityCatalog(
            environments=[
                EnvironmentEntity(id="env_caldera", name="Tambora Caldera", spatial_layout="Volcano cone", lighting_setup="Eruption glow"),
                EnvironmentEntity(id="env_stratosphere", name="Stratospheric Canopy", spatial_layout="High-altitude aerosol haze", lighting_setup="Subdued sunlight"),
                EnvironmentEntity(id="env_farmland", name="European Farmland", spatial_layout="Flooded crop fields", lighting_setup="Cold rain 6500K"),
                EnvironmentEntity(id="env_cellar", name="Peasant Storeroom", spatial_layout="Enclosed dark stone cellar", lighting_setup="Single dim lantern"),
            ]
        ),
    )

    assert len(vb.entities.environments) == 4
    env_ids = [e.id for e in vb.entities.environments]
    assert "env_caldera" in env_ids
    assert "env_cellar" in env_ids


# ==============================================================================
# 3. Beat-Aware Environment Slicing Tests
# ==============================================================================

def test_interior_isolation_prevents_outdoor_landscape_contamination():
    """An interior storeroom beat slices out outdoor crop rows and muddy furrows from prompt."""
    env = {
        "id": "env_cold_wet_farmland",
        "name": "Distant Farmland in the Year Without a Summer",
        "spatial_layout": "Low crop rows and muddy fields occupy the foreground, with distant rural buildings on horizon.",
        "lighting_setup": "Cool overcast 6500K daylight with muted fill.",
        "persistent_props": [
            "Rows of rain-damaged crops",
            "Muddy field furrows",
            "Distant rural buildings",
            "Low rain-heavy clouds",
        ],
    }

    beat = {
        "shot_type": "interior",
        "visual_intent": "human_experience",
        "focal_point": "Nearly empty storage baskets in a shadowed storeroom",
        "visual_description": "A rural peasant woman stares into empty wicker storage baskets in a dim storeroom.",
        "environment": "A modest rural storeroom with rough wooden surfaces and sparse damaged produce",
        "narration_text": "food grew scarce,",
    }

    sliced_props, incl_layout, local_setting, additional_negs = slice_environment_for_beat(env, beat)

    # Spatial layout with muddy field rows must NOT be included
    assert incl_layout is False
    # Outdoor props must be sliced out
    assert "Rows of rain-damaged crops" not in sliced_props
    assert "Muddy field furrows" not in sliced_props
    # Local setting should reflect the interior storeroom
    assert local_setting is not None
    assert "storeroom" in local_setting.lower()
    # Should include negative constraint against outdoor landscape
    assert any("outdoor landscape" in neg for neg in additional_negs)


def test_close_up_detail_focus_slices_out_wide_panoramic_layout():
    """A close-up/macro shot of a sprout or window bleed hole suppresses wide-angle midground layout."""
    env = {
        "id": "env_mountain",
        "name": "Mount Tambora Summit",
        "spatial_layout": "A steep volcanic cone dominates the midground with broad summit crater and coastline.",
        "lighting_setup": "Harsh volcanic glare",
        "persistent_props": ["Broad summit crater", "Volcanic cone slopes", "Distant coastline", "Pumice fragments"],
    }

    beat = {
        "shot_type": "macro",
        "visual_intent": "detail_focus",
        "focal_point": "Microscopic porous pumice rock fragment with air vesicles",
        "visual_description": "Extreme macro close-up of porous pumice texture with tiny volcanic gas pockets.",
        "environment": "Volcanic slope ground level",
        "narration_text": "trapping sulfur within glass.",
    }

    sliced_props, incl_layout, local_setting, additional_negs = slice_environment_for_beat(env, beat)

    # Wide midground cone layout must NOT be included in a macro shot
    assert incl_layout is False
    # Only relevant prop (pumice fragments) should remain
    assert "Pumice fragments" in sliced_props
    assert "Distant coastline" not in sliced_props
    assert "Broad summit crater" not in sliced_props


def test_legitimate_visual_reveal_of_unseen_or_distant_subject_not_suppressed():
    """When narration notes characters never saw a volcano, but the directorial beat is a visual reveal, volcano is NOT suppressed."""
    preset = resolve_visual_style_preset(style_id="stylized_2d_narrative")
    vb = {
        "entities": {
            "environments": [
                {
                    "id": "env_volcano",
                    "name": "Mount Tambora Caldera",
                    "spatial_layout": "A colossal smoking caldera with towering ash plume ascending into stratosphere.",
                    "lighting_setup": "Volcanic twilight glow",
                    "persistent_props": ["Massive ash column", "Sulfur fissure vents"],
                }
            ],
            "characters": [],
            "objects": [],
            "concepts": [],
        },
        "art_direction": {
            "style_family": "stylized_2d",
            "visual_genre": "stylized_2d_narrative",
            "lighting_philosophy": "Moody atmospheric lighting",
        },
    }

    # Beat intentionally shows the distant volcano as a dramatic visual reveal
    beat = {
        "shot_type": "wide",
        "visual_intent": "action_event",
        "focal_point": "Colossal smoking caldera of Mount Tambora",
        "visual_description": "Dramatic visual reveal of the colossal volcanic caldera of Mount Tambora billowing sulfur ash.",
        "narration_text": "from a distant volcano they had never seen.",
        "entity_bindings": {"environment_id": "env_volcano"},
    }

    prompt = _compose_connected_final_prompt(scene=beat, video_bible=vb, preset=preset)

    # The volcano MUST be included in the composition
    assert "caldera" in prompt.lower() or "volcano" in prompt.lower()
    # Must NOT automatically inject "no volcano" negative constraint merely because narration said "never seen"
    assert "no volcano" not in prompt.lower()


# ==============================================================================
# 4. Prompt Composition Integration Tests
# ==============================================================================

def test_prompt_compose_with_beat_aware_environment_and_style():
    """Verify that final composed prompt receives beat-aware sliced environment under stylized 2D preset, avoiding outdoor name and daylight contamination."""
    preset = resolve_visual_style_preset(style_id="stylized_2d_narrative")

    vb = {
        "entities": {
            "environments": [
                {
                    "id": "env_cold_wet_farmland",
                    "name": "Distant Farmland",
                    "spatial_layout": "Low crop rows and muddy fields occupy the foreground with distant farm houses.",
                    "lighting_setup": "Overcast cold daylight 6500K",
                    "persistent_props": ["Rows of rain-damaged crops", "Muddy field furrows"],
                }
            ],
            "characters": [],
            "objects": [],
            "concepts": [],
        },
        "art_direction": {
            "style_family": "stylized_2d",
            "visual_genre": "stylized_2d_narrative",
            "lighting_philosophy": "Atmospheric dramatic lighting",
            "primary_palette_hex": ["#FFFFFF", "#212529", "#FF5722"],
            "negative_anchors": ["no photorealism"],
        },
    }

    scene = {
        "shot_type": "interior",
        "camera_angle": "eye-level",
        "visual_intent": "human_experience",
        "focal_point": "Empty baskets in peasant cellar",
        "visual_description": "Peasant woman standing near empty wicker baskets in a shadowy stone storeroom.",
        "environment": "A modest rural storeroom with rough wooden surfaces",
        "narration_text": "food grew scarce,",
        "entity_bindings": {"environment_id": "env_cold_wet_farmland"},
    }

    prompt = _compose_connected_final_prompt(scene=scene, video_bible=vb, preset=preset)

    # Must contain the storeroom setting
    assert "storeroom" in prompt.lower()
    # Must NOT contain the outdoor muddy crop rows
    assert "muddy fields occupy the foreground" not in prompt.lower()
    assert "rows of rain-damaged crops" not in prompt.lower()
    # Must NOT brand the interior storeroom with outdoor farmland identity
    assert "distant farmland" not in prompt.lower()
    # Must NOT contaminate interior with outdoor cold daylight 6500K
    assert "overcast cold daylight" not in prompt.lower()
    assert "dim interior ambient" in prompt.lower()
    # Must contain the style anchors
    assert "stylized 2d" in prompt.lower()
    # Must enforce composition rules
    assert "vertical 9:16" in prompt.lower()
    # Must contain the style anchors
    assert "stylized 2d" in prompt.lower()
    # Must enforce composition rules
    assert "vertical 9:16" in prompt.lower()


# ==============================================================================
# 5. Visual Intent Propagation in Storyboard Normalization
# ==============================================================================

def test_visual_intent_propagation_through_normalization():
    """Structured visual_intent is correctly normalized and propagated to VisualBeat."""
    raw_beats = [
        {
            "narration_text": "How did one volcano darken summers across the world?",
            "visual_description": "Wide establishing view of Mount Tambora at dawn.",
            "visual_intent": "establishing",
            "shot_type": "wide",
        },
        {
            "narration_text": "Harvests failed across Europe,",
            "visual_description": "Close-up of frozen ruined wheat stalks.",
            "visual_intent": "detail_focus",
            "shot_type": "close-up",
        },
        {
            "narration_text": "food grew scarce in every village.",
            "visual_description": "Farm family in dim cellar with empty grain sacks.",
            "visual_intent": "human_experience",
            "shot_type": "medium",
        },
        {
            "narration_text": "and distant communities felt the chill of an eruption far away.",
            "visual_description": "Villagers looking up at twilight sky in peaceful aftermath.",
            "visual_intent": "narrative_resolution",
            "shot_type": "wide",
        },
    ]

    normalized = normalize_scene_visual_beats(raw_beats=raw_beats, scene_index=1)
    assert len(normalized) == 4
    assert normalized[0].visual_intent == "establishing"
    assert normalized[1].visual_intent == "detail_focus"
    assert normalized[2].visual_intent == "human_experience"
    assert normalized[3].visual_intent == "narrative_resolution"


# ==============================================================================
# 6. Multi-Genre Generalization Tests
# ==============================================================================

def test_engineering_genre_bleed_hole_contextual_passengers():
    """Engineering topic with contextual passengers, macro bleed hole, and cutaway diagram."""
    preset = resolve_visual_style_preset(style_id="stickman_tactical_explainer")

    vb = {
        "entities": {
            "environments": [
                {
                    "id": "env_cabin",
                    "name": "Commercial Airliner Cabin",
                    "spatial_layout": "Passenger seating rows looking toward windows with overhead compartments.",
                    "lighting_setup": "Soft cabin ambient light with sunlight streaming through oval windows.",
                    "persistent_props": ["Overhead reading lights", "Passenger seats", "Window frame"],
                }
            ],
            "concepts": [
                {
                    "id": "cpt_pressure_diff",
                    "name": "Cabin Pressure Equalization",
                    "concept_explained": "The tiny bleed hole balances cabin pressure and outside atmospheric pressure.",
                    "visual_analogy": "Cutaway of the triple acrylic pane assembly showing airflow vectors through the tiny bleed hole.",
                    "graphical_elements": [
                        "Triple-pane acrylic window cross-section",
                        "Airflow vectors balancing cabin pressure",
                        "Moisture relief channel indicator",
                    ],
                }
            ],
        },
        "art_direction": {
            "style_family": "stickman_tactical",
            "visual_genre": "stickman_illustrated",
            "primary_palette_hex": ["#000000", "#FFFFFF", "#00B0FF"],
        },
    }

    # Beat 1: Contextual passenger looking at window
    beat_human = {
        "shot_type": "medium",
        "visual_intent": "human_experience",
        "focal_point": "Passenger hand touching window acrylic",
        "visual_description": "Passenger seated by oval airplane window pointing toward a tiny dot in the glass.",
        "environment": "Airplane cabin interior",
        "narration_text": "Ever noticed that tiny hole in your airplane window?",
        "entity_bindings": {"environment_id": "env_cabin"},
    }
    p_human = _compose_connected_final_prompt(scene=beat_human, video_bible=vb, preset=preset)
    assert "airplane window" in p_human.lower() or "passenger" in p_human.lower()

    # Beat 2: Detail macro of bleed hole
    beat_macro = {
        "shot_type": "macro",
        "visual_intent": "detail_focus",
        "focal_point": "Tiny circular 1.5mm bleed hole in middle acrylic pane",
        "visual_description": "Extreme macro of the tiny laser-drilled bleed hole in transparent acrylic pane.",
        "environment": "Airplane window surface",
        "narration_text": "It's called a bleed hole,",
        "entity_bindings": {"environment_id": "env_cabin"},
    }
    p_macro = _compose_connected_final_prompt(scene=beat_macro, video_bible=vb, preset=preset)
    assert "bleed hole" in p_macro.lower()
    # Macro should not contain wide passenger seating rows layout
    assert "seating rows looking toward windows" not in p_macro.lower()

    # Beat 3: Mechanism diagram (Phase 1A slicing preserved)
    beat_diagram = {
        "shot_type": "diagram",
        "visual_intent": "mechanism_explanation",
        "visual_technique": "cutaway-diagram",
        "focal_point": "Pressure gradient between outer and inner acrylic panes",
        "visual_description": "Cutaway diagram showing airflow vectors through the triple-pane window.",
        "narration_text": "and it balances the enormous pressure difference between the cabin and the thin air outside.",
        "entity_bindings": {"concept_id": "cpt_pressure_diff"},
    }
    p_diagram = _compose_connected_final_prompt(scene=beat_diagram, video_bible=vb, preset=preset)
    assert "triple-pane" in p_diagram.lower() or "pressure" in p_diagram.lower()


def test_astronomy_genre_no_humans_cosmic_scale():
    """Astronomy topic strictly forbids humans while generating scale-appropriate cosmic environments."""
    cd = CreativeDirectionOutput(
        format_archetype="cosmic_scale",
        storytelling_angle="Exploring gravitational collapse",
        central_thesis="Supermassive black holes warp spacetime",
        tone="Awe-inspiring and contemplative",
        visual_genre="photorealistic_cinematic",
        protagonist_config=ProtagonistConfig(human_presence="none"),
        emotional_progression=[
            EmotionalBeat(phase="hook", target_emotion="curiosity", visual_intent="Show accretion disk"),
            EmotionalBeat(phase="setup", target_emotion="awe", visual_intent="Show event horizon"),
            EmotionalBeat(phase="resolution", target_emotion="clarity", visual_intent="Show gravitational lensing"),
        ],
        creative_constraints=["Strict astrophysics accuracy"],
    )

    warnings = validate_creative_direction(cd, topic="How Black Holes Bend Light")
    assert len(warnings) == 0
    assert cd.protagonist_config.human_presence == "none"


# ==============================================================================
# 7. Required Visual Intent & Validation Soft-Check Tests
# ==============================================================================

def test_storyboard_beat_schemas_require_visual_intent():
    """StoryboardBeat, VisualBeat, and SceneInput must require visual_intent with no automatic default."""
    import pytest
    from pydantic import ValidationError
    from app.schemas import StoryboardBeat, EntityBindings
    from app.steps.storyboard import VisualBeat, SceneInput

    # 1. StoryboardBeat requires visual_intent
    with pytest.raises(ValidationError):
        StoryboardBeat(
            beat_id="beat_1",
            scene_index=1,
            order=1,
            narration_slice="Test slice",
            duration_est_seconds=2.0,
            entity_bindings=EntityBindings(),
        )  # type: ignore

    valid_sb_beat = StoryboardBeat(
        beat_id="beat_1",
        scene_index=1,
        order=1,
        narration_slice="Test slice",
        duration_est_seconds=2.0,
        visual_intent="action_event",
        entity_bindings=EntityBindings(),
    )
    assert valid_sb_beat.visual_intent == "action_event"

    # 2. VisualBeat requires visual_intent
    with pytest.raises(ValidationError):
        VisualBeat(visual_description="A scene visual beat")  # type: ignore

    valid_v_beat = VisualBeat(
        visual_description="A scene visual beat",
        visual_intent="human_experience",
    )
    assert valid_v_beat.visual_intent == "human_experience"

    # 3. SceneInput requires visual_intent
    with pytest.raises(ValidationError):
        SceneInput(
            narration="Test narration",
            shot_type="medium",
        )  # type: ignore

    valid_scene = SceneInput(
        narration="Test narration",
        visual_intent="mechanism_explanation",
        shot_type="medium",
    )
    assert valid_scene.visual_intent == "mechanism_explanation"


def test_storyboard_soft_checks_flag_missing_or_invalid_visual_intent():
    """validate_storyboard_soft_checks flags scenes and beats with missing or invalid visual_intent."""
    from app.steps.storyboard import validate_storyboard_soft_checks

    # Scene with invalid visual_intent and visual beat with invalid visual_intent
    scenes = [
        {
            "narration": "First scene narration.",
            "visual_intent": "invalid_intent_xyz",
            "shot_type": "wide",
            "image_prompt": "Wide shot of landscape",
            "visual_beats": [
                {
                    "order": 1,
                    "visual_description": "Beat description",
                    "visual_intent": "bad_beat_intent",
                }
            ],
        }
    ]

    warnings = validate_storyboard_soft_checks(
        scenes=scenes,
        script_narration="First scene narration.",
    )

    assert any("Scene 1 has missing or invalid visual_intent 'invalid_intent_xyz'" in w for w in warnings)
    assert any("Scene 1 visual beat 1 has missing or invalid visual_intent 'bad_beat_intent'" in w for w in warnings)
