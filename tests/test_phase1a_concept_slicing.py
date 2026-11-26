"""Regression and unit tests for Phase 1A: Concept Slicing & Beat-Specific Prompt Priority."""

import copy
import pytest

from app.schemas import (
    ArtDirection,
    CameraDirection,
    ConceptEntity,
    CreativeDirectionOutput,
    EntityBindings,
    EntityCatalog,
    EnvironmentEntity,
    ProtagonistConfig,
    StoryboardBeat,
    VideoBibleOutput,
)
from app.steps.prompt_compose import (
    compose_final_prompt,
    extract_scarcity_negative_constraints,
    recompute_final_prompts,
    slice_concept_for_beat,
)
from app.styles.adapter import adapt_concept
from app.styles.registry import get_style


def _make_multi_stage_bible() -> VideoBibleOutput:
    """Create a test Video Bible with a multi-stage concept and real-world environments."""
    return VideoBibleOutput(
        bible_id="vb_volcano_climate",
        topic_summary="Mount Tambora 1815 eruption and global cooling mechanism.",
        entities=EntityCatalog(
            characters=[],
            environments=[
                EnvironmentEntity(
                    id="env_tambora",
                    name="Mount Tambora Summit",
                    spatial_layout="Steep volcanic slopes leading to a crater summit beneath an expansive sky",
                    lighting_setup="Volcanic glow contrasting with cold storm sky",
                    persistent_props=["crater rim", "volcanic slopes"],
                    time_of_day="April 1815",
                ),
                EnvironmentEntity(
                    id="env_farmland",
                    name="European Farmland 1816",
                    spatial_layout="Cold muddy crop rows and distant rural cottages beneath heavy overcast skies",
                    lighting_setup="Dim, overcast cool daylight 6500K with soft shadows",
                    persistent_props=["rain-damaged crops", "muddy furrows", "rural barns"],
                    time_of_day="Summer 1816",
                ),
            ],
            objects=[],
            concepts=[
                ConceptEntity(
                    id="concept_eruption_climate_chain",
                    name="Eruption-to-Climate Causal Chain",
                    concept_explained=(
                        "Tambora's eruption injected ash and sulfur gases into the stratosphere. "
                        "Sulfur gases formed sulfate aerosols that reflected incoming solar radiation back into space. "
                        "Reduced solar energy cooled global surface temperatures, triggering unseasonable frost, ruined harvests, and food shortages."
                    ),
                    visual_analogy=(
                        "A hand-drawn cutaway links the erupting volcano to an atmospheric layer, "
                        "then shows solar rays deflecting back to space before resolving into rain-swept blighted fields."
                    ),
                    graphical_elements=[
                        "A simplified Mount Tambora silhouette releasing ash and sulfur gases",
                        "A high atmospheric band carrying small sulfur-derived aerosol particles outward",
                        "Sunlight rays with some reflected back toward space",
                        "A subdued temperature-decline indicator on a curved Earth horizon",
                        "Regional weather symbols showing patchy cold rainfall and cloud cover",
                        "A final causal arrow linking freezing summer rains to damaged crop roots",
                    ],
                )
            ],
        ),
        art_direction=ArtDirection(
            style_family="illustrated",
            visual_genre="2d_cartoon",
            lighting_philosophy="Atmospheric dramatic lighting with motivated key and cool ambient fill",
            primary_palette_hex=["#212529", "#5C3A21", "#E91E63", "#FFE082"],
            rendering_texture="Clean tapered black contour lines with two-tone digital cel-shading",
            negative_anchors=["photorealism", "3D volumetric render", "blurry watercolor"],
        ),
    )


def _make_creative_direction(human_presence: str = "none") -> CreativeDirectionOutput:
    return CreativeDirectionOutput(
        format_archetype="historical_chronicle",
        storytelling_angle="Following the chain from Tambora's eruption to global summer cooling.",
        central_thesis="Volcanic aerosols screened sunlight and caused the 1816 Year Without a Summer.",
        tone="Documentary, dark, awe-struck",
        visual_genre="mixed_format",
        protagonist_config=ProtagonistConfig(human_presence=human_presence),
        emotional_progression=[],
        creative_constraints=["Do not depict an ocean literally disappearing"],
    )


def test_multi_stage_concept_slices_per_beat():
    """Verify that beats referencing a multi-stage concept receive ONLY relevant elements."""
    bible = _make_multi_stage_bible()
    cd = _make_creative_direction()
    preset = get_style("stylized_2d_narrative")

    beat_particles = {
        "visual_description": "In a simplified upper-atmosphere cutaway, gases spread from a small volcano silhouette and become fine aerosol particles.",
        "focal_point": "Fine aerosol particles forming in the high atmospheric layer",
        "narration_text": "As the sulfur gases spread, they formed tiny particles",
        "shot_type": "diagram",
        "visual_technique": "hand-drawn scientific causal-chain diagram",
        "entity_bindings": {"concept_id": "concept_eruption_climate_chain"},
    }

    beat_sunlight = {
        "visual_description": "A cross-section shows incoming solar radiation meeting the aerosol veil, with sunlight rays deflecting back toward space.",
        "focal_point": "Sunlight rays reflecting off the particle layer back into space",
        "narration_text": "that reflected some sunlight back into space.",
        "shot_type": "cross-section",
        "visual_technique": "hand-drawn atmospheric cross-section",
        "entity_bindings": {"concept_id": "concept_eruption_climate_chain"},
    }

    prompt_particles = compose_final_prompt(
        scene=beat_particles,
        video_bible=bible,
        creative_direction=cd,
        preset=preset,
        video_topic="The Ocean That Disappeared Overnight",
    )

    prompt_sunlight = compose_final_prompt(
        scene=beat_sunlight,
        video_bible=bible,
        creative_direction=cd,
        preset=preset,
        video_topic="The Ocean That Disappeared Overnight",
    )

    # 1. Beat A (particles) has relevant particle/volcano elements
    assert "sulfur" in prompt_particles.lower() or "particles" in prompt_particles.lower()
    # Beat A must NOT contain later stages: weather symbols, temperature decline, or crop damage
    assert "weather symbols" not in prompt_particles.lower()
    assert "temperature-decline" not in prompt_particles.lower()
    assert "damaged crop roots" not in prompt_particles.lower()

    # 2. Beat B (sunlight) has sunlight rays reflection
    assert "sunlight rays with some reflected back toward space" in prompt_sunlight.lower()
    # Beat B must NOT contain volcano eruption, temperature decline, weather symbols, or crop roots
    assert "weather symbols" not in prompt_sunlight.lower()
    assert "temperature-decline" not in prompt_sunlight.lower()
    assert "damaged crop roots" not in prompt_sunlight.lower()
    assert "releasing ash and sulfur gases" not in prompt_sunlight.lower()


def test_different_beats_receive_distinct_sliced_concept_details():
    """Verify that different beats referencing the same concept get distinct sliced details."""
    bible = _make_multi_stage_bible()
    concept = bible.entities.concepts[0]

    beat_weather = {
        "visual_description": "Regional overview showing separate weather symbols across distinct map territories.",
        "focal_point": "Varied weather symbols across different regional areas",
        "narration_text": "but the effects varied by place and weather.",
        "shot_type": "diagram",
    }

    beat_temp = {
        "visual_description": "A curved Earth horizon displaying a subdued temperature-decline indicator as cooling spreads.",
        "focal_point": "A subdued temperature indicator on the curved Earth horizon",
        "narration_text": "global temperatures dipped",
        "shot_type": "diagram",
    }

    elems_weather, exp_weather, _ = slice_concept_for_beat(concept, beat_weather)
    elems_temp, exp_temp, _ = slice_concept_for_beat(concept, beat_temp)

    assert any("weather symbols" in e.lower() for e in elems_weather)
    assert not any("temperature-decline" in e.lower() for e in elems_weather)

    assert any("temperature-decline" in e.lower() for e in elems_temp)
    assert not any("weather symbols" in e.lower() for e in elems_temp)


def test_beat_not_requiring_diagram_is_not_forced_into_one():
    """When a beat has an environmental shot and environment_id, do NOT force an infographic diagram."""
    bible = _make_multi_stage_bible()
    cd = _make_creative_direction()
    preset = get_style("stylized_2d_narrative")

    landscape_beat = {
        "visual_description": "Sweeping wide view over rain-drenched fields and distant farmhouses under heavy gray clouds.",
        "focal_point": "Rain falling over dark furrowed farmland",
        "narration_text": "and summer felt as if it had vanished beneath clouds and rain.",
        "shot_type": "wide",
        "visual_technique": "atmospheric illustrated landscape",
        "entity_bindings": {
            "environment_id": "env_farmland",
            # Storyboard attached concept_id, but the beat is fundamentally a landscape environment
            "concept_id": "concept_eruption_climate_chain",
        },
    }

    prompt = compose_final_prompt(
        scene=landscape_beat,
        video_bible=bible,
        creative_direction=cd,
        preset=preset,
        video_topic="The Ocean That Disappeared Overnight",
    )

    # Must prioritize environment description
    assert "European Farmland 1816" in prompt
    assert "rain-damaged crops" in prompt
    # Must NOT force the conceptual diagram block into Segment 3
    assert "Illustrated conceptual diagram:" not in prompt
    assert "graphical elements:" not in prompt


def test_scarcity_anti_abundance_negative_constraints():
    """Where narration describes crop failure or scarcity, prompts must prevent contradictory visuals."""
    bible = _make_multi_stage_bible()
    cd = _make_creative_direction()
    preset = get_style("stylized_2d_narrative")

    crop_failure_beat = {
        "visual_description": "A sparse collection of damaged crops lies on a rough surface with signs of spoilage.",
        "focal_point": "The meager damaged harvest",
        "narration_text": "Harvests failed,",
        "shot_type": "top-down",
        "entity_bindings": {"environment_id": "env_farmland"},
    }

    food_scarcity_beat = {
        "visual_description": "Nearly empty storage baskets and bare wooden shelves inside a shadowy storeroom.",
        "focal_point": "Nearly bare baskets and dwindling provisions",
        "narration_text": "food grew scarce,",
        "shot_type": "medium",
        "entity_bindings": {"environment_id": "env_farmland"},
    }

    neutral_beat = {
        "visual_description": "Mount Tambora's crater silhouette looms under a darkening sky.",
        "focal_point": "Crater silhouette",
        "narration_text": "How did one volcano darken summers across the world?",
        "shot_type": "wide",
        "entity_bindings": {"environment_id": "env_tambora"},
    }

    prompt_crops = compose_final_prompt(crop_failure_beat, video_bible=bible, creative_direction=cd, preset=preset)
    prompt_food = compose_final_prompt(food_scarcity_beat, video_bible=bible, creative_direction=cd, preset=preset)
    prompt_neutral = compose_final_prompt(neutral_beat, video_bible=bible, creative_direction=cd, preset=preset)

    # Crop failure constraints present in Avoid section
    assert "no abundant harvest" in prompt_crops
    assert "no healthy vegetables" in prompt_crops
    assert "no overflowing produce" in prompt_crops

    # Food scarcity constraints present in Avoid section
    assert "no abundant food" in prompt_food
    assert "no full baskets" in prompt_food
    assert "no overflowing pantry" in prompt_food

    # Neutral beat does NOT contain scarcity constraints
    assert "no abundant harvest" not in prompt_neutral
    assert "no overflowing produce" not in prompt_neutral
    assert "no full baskets" not in prompt_neutral


def test_no_bedroom_setting_not_leaked_into_non_bedroom_stories():
    """Ensure 'no bedroom setting' is NOT injected into stories having no bedroom context."""
    bible = _make_multi_stage_bible()
    cd = _make_creative_direction()
    preset = get_style("stylized_2d_narrative")

    diagram_beat = {
        "visual_description": "Atmospheric cutaway showing aerosol particles.",
        "narration_text": "particles formed in the upper atmosphere",
        "shot_type": "diagram",
        "entity_bindings": {"concept_id": "concept_eruption_climate_chain"},
    }

    prompt = compose_final_prompt(
        scene=diagram_beat,
        video_bible=bible,
        creative_direction=cd,
        preset=preset,
        video_topic="The Ocean That Disappeared Overnight",
    )

    assert "no bedroom setting" not in prompt


def test_no_bedroom_setting_retained_when_bedroom_context_exists():
    """Ensure 'no bedroom setting' IS retained when the story actually has bedroom/sleep context."""
    bible = VideoBibleOutput(
        bible_id="vb_sleep",
        topic_summary="Neurological sleep onset mechanisms.",
        entities=EntityCatalog(
            characters=[],
            environments=[
                EnvironmentEntity(
                    id="env_bedroom_night",
                    name="Modern Bedroom",
                    spatial_layout="Bed with blue duvet and bedside lamp",
                    lighting_setup="Night chiaroscuro",
                    persistent_props=["bedside clock"],
                )
            ],
            objects=[],
            concepts=[
                ConceptEntity(
                    id="cpt_brain",
                    name="Brain Sleep Gate",
                    concept_explained="Brainstem inhibits motor signals.",
                    visual_analogy="Cyan and violet vectors competing.",
                    graphical_elements=["cyan vector", "violet vector"],
                )
            ],
        ),
        art_direction=ArtDirection(
            style_family="illustrated",
            visual_genre="2d_cartoon",
            lighting_philosophy="Moody dramatic",
            primary_palette_hex=["#000000", "#FFFFFF"],
            rendering_texture="Clean linework",
        ),
    )
    cd = CreativeDirectionOutput(
        format_archetype="explainer_myth_bust",
        storytelling_angle="Why we jerk when falling asleep in bedroom.",
        central_thesis="Hypnic jerks are sleep onset glitches.",
        tone="reassuring",
        visual_genre="styled_3d",
        protagonist_config=ProtagonistConfig(human_presence="none"),
        emotional_progression=[],
        creative_constraints=[],
    )
    preset = get_style("stylized_2d_narrative")

    diagram_beat = {
        "visual_description": "Brainstem motor control signal diagram.",
        "narration_text": "motor signals are inhibited",
        "shot_type": "diagram",
        "entity_bindings": {"concept_id": "cpt_brain"},
    }

    prompt = compose_final_prompt(
        scene=diagram_beat,
        video_bible=bible,
        creative_direction=cd,
        preset=preset,
        video_topic="Why Do We Feel Like Falling Asleep?",
    )

    assert "no bedroom setting" in prompt


def test_video_bible_concept_preserved_immutably():
    """Verify that concept slicing does NOT mutate or alter the original Video Bible concept."""
    bible = _make_multi_stage_bible()
    original_concept = copy.deepcopy(bible.entities.concepts[0])

    beat = {
        "visual_description": "Incoming sunlight reflecting off aerosol particles.",
        "narration_text": "reflecting sunlight back into space",
        "shot_type": "diagram",
        "entity_bindings": {"concept_id": "concept_eruption_climate_chain"},
    }

    _ = compose_final_prompt(beat, video_bible=bible, preset=get_style("stylized_2d_narrative"))

    current_concept = bible.entities.concepts[0]
    assert current_concept.graphical_elements == original_concept.graphical_elements
    assert len(current_concept.graphical_elements) == 6
    assert current_concept.concept_explained == original_concept.concept_explained
    assert current_concept.visual_analogy == original_concept.visual_analogy


def test_atomic_concept_preserves_all_elements():
    """Concepts with <= 2 elements remain intact without aggressive truncation."""
    concept = ConceptEntity(
        id="cpt_lift",
        name="Bernoulli Lift",
        concept_explained="Air travels faster over curved surfaces.",
        visual_analogy="Airflow slicing over an airfoil.",
        graphical_elements=["High pressure vector", "Low pressure vector"],
    )
    beat = {
        "visual_description": "Airfoil showing high pressure below and low pressure above.",
        "narration_text": "creating lift",
    }

    elems, exp, ana = slice_concept_for_beat(concept, beat)
    assert len(elems) == 2
    assert "High pressure vector" in elems
    assert "Low pressure vector" in elems


def test_legacy_prompt_composition_backward_compatibility():
    """Verify legacy prompt composition paths continue to produce expected output."""
    scene = {
        "image_prompt": "A small glossy blue cube with rounded edges on a pedestal.",
        "uses_anchor": False,
        "shot_type": "medium",
        "camera_angle": "eye-level",
        "visual_purpose": "Introduce the blue cube",
        "narration": "This is a blue cube.",
    }

    prompt = compose_final_prompt(
        scene=scene,
        subject_anchor="A glossy blue cube",
        visual_style="Clean 3D render, studio lighting",
    )

    assert "A small glossy blue cube with rounded edges on a pedestal" in prompt
    assert "Clean 3D render, studio lighting" in prompt
    assert "Vertical 9:16" in prompt


def test_compound_analogy_leakage_prevented_in_early_beat():
    """Verify that multi-stage compound analogies do not leak downstream events into earlier beats."""
    bible = _make_multi_stage_bible()
    cd = _make_creative_direction()
    preset = get_style("stylized_2d_narrative")

    # Beat 4_1: Sulfur gases forming tiny particles in the atmospheric layer
    beat_particles = {
        "visual_description": "In a simplified upper-atmosphere cutaway, gases spread from a small volcano silhouette and become fine aerosol particles.",
        "focal_point": "Fine aerosol particles forming in the high atmospheric layer",
        "narration_text": "As the sulfur gases spread, they formed tiny particles",
        "shot_type": "diagram",
        "visual_technique": "hand-drawn scientific causal-chain diagram",
        "entity_bindings": {"concept_id": "concept_eruption_climate_chain"},
    }

    prompt = compose_final_prompt(
        scene=beat_particles,
        video_bible=bible,
        creative_direction=cd,
        preset=preset,
        video_topic="The Ocean That Disappeared Overnight",
    )

    # 1. Must include the clean supporting clause
    assert "links the erupting volcano to an atmospheric layer" in prompt.lower()

    # 2. Must NOT leak downstream events from later stages (solar rays deflecting, blighted fields)
    assert "solar rays deflecting" not in prompt.lower()
    assert "blighted fields" not in prompt.lower()
    assert "damaged crop" not in prompt.lower()
    assert "rain-swept" not in prompt.lower()


def test_empty_beat_narration_prioritizes_visual_description_and_focal_point():
    """Empty or duplicate beat narration must NOT cause wrong concept selection or explanation misattribution."""
    bible = _make_multi_stage_bible()
    concept = bible.entities.concepts[0]

    scene_narration = "With less solar energy reaching the surface, global temperatures dipped, but the effects varied by place and weather."

    # Beat 6_1: Empty beat narration, scene narration passed as context
    beat_cooling = {
        "visual_description": "Fewer rays pass through the atmospheric layer to the curved surface below, suggesting a broad cooling influence.",
        "focal_point": "The reduced sunlight reaching the surface",
        "visual_purpose": "Show the drop in solar energy reaching the ground before shifting to regional differences.",
        "narration": "",  # Empty beat narration
        "scene_narration": scene_narration,
        "shot_type": "diagram",
    }

    # Beat 6_2: Regional weather variation
    beat_regional = {
        "visual_description": "The diagram shifts across distinct regions, showing different combinations of clouds, rain, and cooler light.",
        "focal_point": "Contrasting weather patterns across separate regions",
        "visual_purpose": "Prevent the misconception that cooling was identical everywhere.",
        "narration_text": "but the effects varied by place and weather.",
        "scene_narration": scene_narration,
        "shot_type": "diagram",
    }

    elems_cooling, exp_cooling, ana_cooling = slice_concept_for_beat(concept, beat_cooling)
    elems_regional, exp_regional, ana_regional = slice_concept_for_beat(concept, beat_regional)

    # Beat 6_1 focuses on solar deflection and broad cooling
    assert any("sunlight rays" in e.lower() for e in elems_cooling)
    assert not any("weather symbols" in e.lower() for e in elems_cooling)
    assert exp_cooling is not None
    assert "reduced solar energy" in exp_cooling.lower() or "sulfate aerosols" in exp_cooling.lower()

    # Beat 6_2 focuses on regional weather variation
    assert any("weather symbols" in e.lower() for e in elems_regional)
    assert not any("sunlight rays" in e.lower() for e in elems_regional)


def test_useful_concept_overlap_preserved_across_connected_beats():
    """Verify connected sequential beats can share relevant elements for scientific continuity."""
    bible = _make_multi_stage_bible()
    concept = bible.entities.concepts[0]

    beat_sunlight_layer = {
        "visual_description": "Incoming solar rays strike the high atmospheric aerosol veil, deflecting back to space.",
        "focal_point": "Sunlight rays reflecting off the atmospheric layer",
        "shot_type": "cross-section",
    }

    beat_ground_cooling = {
        "visual_description": "Fewer solar rays reach the Earth surface beneath the atmospheric veil, depicting dimmed sunlight.",
        "focal_point": "Reduced sunlight rays reaching the ground beneath the atmospheric layer",
        "shot_type": "cross-section",
    }

    elems_a, _, _ = slice_concept_for_beat(concept, beat_sunlight_layer)
    elems_b, _, _ = slice_concept_for_beat(concept, beat_ground_cooling)

    # Both beats genuinely involve sunlight rays and the atmospheric layer
    assert any("sunlight rays" in e.lower() for e in elems_a)
    assert any("sunlight rays" in e.lower() for e in elems_b)


def test_complete_analogy_preserved_when_genuinely_applicable():
    """Unified single-concept analogies without unrelated downstream stages are preserved in full."""
    concept = ConceptEntity(
        id="cpt_parasol",
        name="Stratospheric Parasol",
        concept_explained="Fine particles create an umbrella that bounces solar energy away from Earth.",
        visual_analogy="Like a frosted glass parasol held over the Earth, reflecting sunlight before it can warm the surface.",
        graphical_elements=[
            "Frosted glass parasol shield",
            "Reflected solar beams bouncing off the parasol",
            "Shaded Earth surface below",
        ],
    )

    beat = {
        "visual_description": "A cross-section shows sunlight bouncing off the high aerosol shield over the Earth.",
        "focal_point": "Reflected sunlight beams bouncing away from Earth",
        "narration_text": "reflecting sunlight before it can warm the surface",
    }

    elems, exp, ana = slice_concept_for_beat(concept, beat)
    assert ana == "Like a frosted glass parasol held over the Earth, reflecting sunlight before it can warm the surface."
