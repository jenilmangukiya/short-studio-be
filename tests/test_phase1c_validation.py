"""Comprehensive regression test suite for Phase 1C: Visual Storytelling Validation.

Covers:
1. Canonical visual style (removal of visual_genre, frozen style snapshot single source of truth).
2. Separation of visual intent from presentation (5 visual techniques, no auto-diagram forcing).
3. Concept composition modes (context-only, in-scene explanation, standalone diagram).
4. Spatial and semantic consistency (Check E2 cross-environment landmark detection, multi-location framing).
5. Visual progression and validation (spatial_cutaway, diagram repetition Check F2, intent quality Check I).
6. Negative constraint optimization (planning commentary removal, deduplication, context filtering).
7. False positive prevention for layout guidance in forbidden words validation.
"""

import pytest
from app.schemas import CreativeDirectionOutput, VideoBibleOutput, EntityCatalog
from app.steps.creative_direction import (
    build_creative_direction_prompt,
    filter_conflicting_channel_settings,
)
from app.steps.prompt_compose import (
    compose_final_prompt,
    determine_concept_presentation_mode,
    _extract_world_negative_constraints,
    _normalize_negative_constraint,
)
from app.steps.storyboard import detect_forbidden_prompt_words
from app.steps.storyboard_validation import validate_connected_storyboard
from app.styles.registry import get_style


# ==============================================================================
# 1. Canonical Visual Style Tests
# ==============================================================================

def test_creative_direction_schema_has_no_visual_genre():
    """Verify visual_genre is removed from CreativeDirectionOutput."""
    cd = CreativeDirectionOutput(
        format_archetype="deep_dive",
        storytelling_angle="Distant climate domino effect",
        central_thesis="1815 Tambora eruption caused the 1816 Year Without a Summer",
        tone="cinematic",
        protagonist_config={"human_presence": "none", "archetype_summary": ""},
        emotional_progression=[{"phase": "hook", "target_emotion": "awe", "visual_intent": "landscape"}],
        creative_constraints=["historical accuracy"],
    )
    assert not hasattr(cd, "visual_genre")
    assert "visual_genre" not in cd.model_dump()


def test_filter_conflicting_channel_settings():
    """Verify channel settings strip artistic style instructions while preserving tone/branding."""
    raw_settings = {
        "visual_style": "High-contrast anime cel-shading",
        "aesthetic": "Dark gritty watercolor",
        "tone": "informative and urgent",
        "brand_color": "#FF5500",
        "artistic_direction": "3D photorealistic render",
    }
    filtered = filter_conflicting_channel_settings(raw_settings)
    assert "tone" in filtered
    assert "brand_color" in filtered
    assert "visual_style" not in filtered
    assert "aesthetic" not in filtered
    assert "artistic_direction" not in filtered


def test_build_creative_direction_prompt_incorporates_frozen_preset():
    """Verify creative direction prompt receives active style preset as context."""
    preset = get_style("stylized_2d_narrative")
    prompt = build_creative_direction_prompt(
        rules_text="MOCK_RULES",
        topic="The Ocean That Disappeared Overnight",
        channel_settings={"tone": "dramatic"},
        visual_style_snapshot=preset.model_dump(),
        visual_style_id="stylized_2d_narrative",
    )
    assert "Visual Style Context" in prompt
    assert "stylized_2d_narrative" in prompt
    assert "governed canonical by this frozen visual style preset" in prompt


# ==============================================================================
# 2. Separation of Visual Intent & Concept Composition Modes
# ==============================================================================

@pytest.fixture
def sample_video_bible():
    preset = get_style("stylized_2d_narrative")
    return {
        "bible_id": "bible_phase1c_test",
        "topic_summary": "Tambora eruption and Year Without a Summer",
        "entities": {
            "environments": [
                {
                    "id": "env_sumbawa_caldera",
                    "name": "Tambora Volcanic Caldera",
                    "description": "Massive smoking volcanic caldera surrounded by jagged Indonesian peaks.",
                    "lighting": "Dramatic ash-filtered twilight with red glow",
                    "persistent_props": ["Volcanic crater rim", "Smoking caldera vents", "Sumbawa coastline"],
                },
                {
                    "id": "env_european_farms",
                    "name": "European Farmland",
                    "description": "Rolling agricultural fields with a period stone farmhouse and stone walls.",
                    "lighting": "Cold muted overcast daylight beneath a thin atmospheric veil",
                    "persistent_props": ["Period stone farmhouse", "Dry-stone walls", "Damaged wheat fields"],
                },
            ],
            "characters": [
                {
                    "id": "char_farmer",
                    "name": "European Farmer",
                    "visual_summary": "Weathered 19th-century subsistence farmer in worn linen smock.",
                    "forbidden_mutations": ["no modern clothes", "no tractors"],
                }
            ],
            "concepts": [
                {
                    "id": "concept_aerosol_veil",
                    "name": "Stratospheric Aerosol Veil",
                    "concept_explained": "Sulfur dioxide gas converted into sulfate aerosols that scattered solar radiation.",
                    "visual_analogy": "A thin reflective atmospheric layer scattering incoming sunlight back into space.",
                    "graphical_elements": [
                        "Incoming sunlight rays",
                        "Reflective stratospheric aerosol band",
                        "Deflected light rays bouncing back to space",
                        "Diminished sunlight reaching earth",
                    ],
                }
            ],
            "objects": [],
            "machines": [],
        },
        "art_direction": {
            "style_family": preset.art_direction.style_family,
            "visual_genre": preset.art_direction.visual_genre,
            "rendering_texture": preset.art_direction.rendering_texture,
            "lighting_philosophy": preset.art_direction.lighting_philosophy,
            "primary_palette_hex": preset.default_palette_hex,
            "negative_anchors": list(preset.negative_prompt_anchors or []),
        },
    }


def test_concept_mode_1_context_only(sample_video_bible):
    """Mode 1: Context only — bound concept informs factual context without graphical diagram overlay."""
    scene = {
        "visual_intent": "environmental_context",
        "visual_technique": "illustrated scientific landscape",
        "shot_type": "wide",
        "transition": "environmental_continuation",
        "entity_bindings": {
            "environment_id": "env_european_farms",
            "concept_id": "concept_aerosol_veil",
            "character_id": None,
        },
        "visual_description": "Cold dim daylight blankets rolling farmland beneath a subtle atmospheric haze.",
        "focal_point": "Dim sunlight filtering over crop fields",
    }
    mode = determine_concept_presentation_mode(scene, has_bound_concept=True)
    assert mode == "context_only"

    prompt = compose_final_prompt(
        scene=scene,
        video_bible=sample_video_bible,
        preset=get_style("stylized_2d_narrative"),
    )
    # Environment is rendered
    assert "European Farmland" in prompt
    assert "Period stone farmhouse" in prompt
    # No diagram phrasing or graphical element callouts
    assert "Illustrated conceptual diagram:" not in prompt
    assert "graphical elements:" not in prompt
    assert "Visual analogy:" not in prompt


def test_concept_mode_2_in_scene_explanation(sample_video_bible):
    """Mode 2: In-scene explanation — thematic visual cues appear directly inside the environment."""
    scene = {
        "visual_intent": "mechanism_explanation",
        "visual_technique": "in_scene_cue",
        "shot_type": "medium_wide",
        "transition": "explanatory_cutaway",
        "entity_bindings": {
            "environment_id": "env_european_farms",
            "concept_id": "concept_aerosol_veil",
            "character_id": "char_farmer",
        },
        "visual_description": "A farmer inspects withered crops as faint scattered sunbeams illuminate the haze above.",
        "focal_point": "Scattered sunbeams caught in the atmospheric haze",
    }
    mode = determine_concept_presentation_mode(scene, has_bound_concept=True)
    assert mode == "in_scene"

    prompt = compose_final_prompt(
        scene=scene,
        video_bible=sample_video_bible,
        preset=get_style("stylized_2d_narrative"),
    )
    # Both character and environment are rendered
    assert "European Farmer" in prompt
    assert "European Farmland" in prompt
    # In-scene cue is appended without turning into a standalone diagram
    assert "Thematic in-scene cue:" in prompt
    assert "Illustrated conceptual diagram:" not in prompt


def test_concept_mode_3_standalone_diagram(sample_video_bible):
    """Mode 3: Standalone diagram — dedicated conceptual visualization when explicitly specified."""
    scene = {
        "visual_intent": "mechanism_explanation",
        "visual_technique": "diagram",
        "shot_type": "diagram",
        "transition": "explanatory_cutaway",
        "entity_bindings": {
            "environment_id": None,
            "concept_id": "concept_aerosol_veil",
            "character_id": None,
        },
        "visual_description": "Schematic flow showing aerosol particles scattering solar radiation.",
        "focal_point": "Deflection of incoming solar rays",
    }
    mode = determine_concept_presentation_mode(scene, has_bound_concept=True)
    assert mode == "diagram"

    prompt = compose_final_prompt(
        scene=scene,
        video_bible=sample_video_bible,
        preset=get_style("stylized_2d_narrative"),
    )
    assert "Illustrated conceptual diagram: Stratospheric Aerosol Veil" in prompt
    assert "Visual analogy:" in prompt


def test_transition_and_mechanism_intent_do_not_force_diagram(sample_video_bible):
    """Verify that mechanism_explanation and explanatory_cutaway do NOT automatically force diagram mode."""
    scene = {
        "visual_intent": "mechanism_explanation",
        "visual_technique": "cinematic_illustration",
        "shot_type": "close_up",
        "transition": "explanatory_cutaway",
        "entity_bindings": {
            "environment_id": "env_european_farms",
            "concept_id": "concept_aerosol_veil",
            "character_id": None,
        },
        "visual_description": "Close physical view of frost crystals forming on fragile wheat heads in mid-summer.",
        "focal_point": "Frost crystals on wheat stems",
    }
    mode = determine_concept_presentation_mode(scene, has_bound_concept=True)
    assert mode == "context_only"

    prompt = compose_final_prompt(
        scene=scene,
        video_bible=sample_video_bible,
        preset=get_style("stylized_2d_narrative"),
    )
    assert "Illustrated conceptual diagram:" not in prompt
    assert "European Farmland" in prompt


# ==============================================================================
# 3. Spatial and Semantic Consistency Validation
# ==============================================================================

def test_check_e2_detects_remote_landmark_contradiction(sample_video_bible):
    """Check E2 flags physically placing a remote volcano landmark in a European farm environment."""
    scenes = [
        {
            "index": 1,
            "environment": "European Farmland",
            "entity_bindings": {"environment_id": "env_european_farms"},
            "visual_intent": "narrative_resolution",
            "visual_technique": "cinematic_illustration",
            "shot_type": "wide",
            "transition": "resolution_return",
            "visual_description": "A distant smoking volcanic caldera sits on the horizon beyond the European stone farmhouse.",
            "focal_point": "Distant volcanic caldera over farm fields",
            "image_prompt": "European farmland with volcanic caldera vents visible on the horizon.",
        }
    ]
    warnings = validate_connected_storyboard(scenes, sample_video_bible)
    assert any("[Contradiction]" in w and "Caldera" in w and "env_sumbawa_caldera" in w for w in warnings)


def test_scene_11_contradiction_from_video_fa7aa9ba():
    """Verify that Scene 11 from video fa7aa9ba is detected as a [Contradiction] in Check E2.

    Scene 11 is bound to env_distant_farms ('Rural Farmland in Europe and North America'),
    but describes 'Distant Tambora beneath a sky whose light reaches faraway land' and
    'faint volcanic silhouette on the far horizon' from env_tambora_sumbawa ('Mount Tambora, Sumbawa')
    without multi-location framing.
    """
    video_bible_fixture = {
        "entities": {
            "environments": [
                {
                    "id": "env_tambora_sumbawa",
                    "name": "Mount Tambora, Sumbawa",
                    "description": "Indonesian volcanic island setting with towering peak and caldera vents.",
                    "persistent_props": ["Volcanic peak", "Caldera", "Tambora slope", "Indonesian coastline"],
                },
                {
                    "id": "env_distant_farms",
                    "name": "Rural Farmland in Europe and North America",
                    "description": "Rolling countryside and open farm fields under an unseasonably cold summer sky.",
                    "persistent_props": ["Stone barn", "Fencing", "Farmhouse", "Crop fields", "Stone wall"],
                },
            ],
            "characters": [],
            "concepts": [],
            "objects": [],
        }
    }
    scenes = [
        {
            "index": 11,
            "narration": "Thousands of miles away, in the summer of 1816, snow fell in New England and crops froze in the fields.",
            "visual_intent": "environmental_context",
            "visual_technique": "cinematic_illustration",
            "shot_type": "wide",
            "transition": "location_change",
            "entity_bindings": {"environment_id": "env_distant_farms"},
            "visual_description": "Wide shot of a European or North American farm under cold summer skies, with a faint volcanic silhouette on the far horizon beyond rolling fields.",
            "focal_point": "Distant Tambora beneath a sky whose light reaches faraway land",
            "image_prompt": "European farmland with cold summer skies and distant Tambora silhouette.",
        }
    ]
    warnings = validate_connected_storyboard(scenes, video_bible_fixture)

    # 1. Validator detects it
    contradictions = [w for w in warnings if w.startswith("[Contradiction]")]
    assert len(contradictions) == 1

    # 2. Severity is [Contradiction] (actionable, non-advisory)
    assert "[Contradiction]" in contradictions[0]
    assert "env_distant_farms" in contradictions[0]
    assert "Tambora" in contradictions[0]
    assert "env_tambora_sumbawa" in contradictions[0]

    # 3. Exact feedback supplied to retry mechanism
    assert contradictions[0] == (
        "[Contradiction] Scene 11 is bound to environment 'env_distant_farms', but its visual description or focal point "
        "places remote landmark 'Tambora' from environment 'env_tambora_sumbawa' directly into the local scene "
        "without explicit multi-location framing (e.g. split-screen, in-scene map, or global view)."
    )


def test_shared_generic_features_do_not_trigger_contradiction():
    """Verify generic objects and landscape features (stone walls, fields, trees, mud, barn) do not trigger false positive contradictions."""
    video_bible_fixture = {
        "entities": {
            "environments": [
                {
                    "id": "env_village",
                    "name": "Mountain Village",
                    "persistent_props": ["Stone wall", "Timber fence", "Barn", "Mud path", "Field"],
                },
                {
                    "id": "env_farmland",
                    "name": "Valley Farmland",
                    "persistent_props": ["Dry-stone wall", "Crops", "Trees", "Pasture", "Wooden cart"],
                },
            ]
        }
    }
    scenes = [
        {
            "index": 1,
            "environment": "Valley Farmland",
            "entity_bindings": {"environment_id": "env_farmland"},
            "visual_intent": "environmental_context",
            "visual_technique": "cinematic_illustration",
            "shot_type": "wide",
            "transition": "establishing",
            "visual_description": "A stone wall lines the mud path beside an old timber fence and open field.",
            "focal_point": "Stone wall along mud path",
        }
    ]
    warnings = validate_connected_storyboard(scenes, video_bible_fixture)
    assert not any("[Contradiction]" in w for w in warnings)


def test_legitimate_multi_location_compositions_preserved():
    """Verify split-screen, in-scene map, and global/orbital view avoid contradiction warnings."""
    video_bible_fixture = {
        "entities": {
            "environments": [
                {
                    "id": "env_tambora",
                    "name": "Mount Tambora, Sumbawa",
                    "persistent_props": ["Tambora caldera", "Volcano peak"],
                },
                {
                    "id": "env_europe",
                    "name": "European Farmhouse",
                    "persistent_props": ["Farmhouse parlor", "Stone hearth"],
                },
            ]
        }
    }
    # In-scene map composition
    scene_map = {
        "index": 1,
        "environment": "European Farmhouse",
        "entity_bindings": {"environment_id": "env_europe"},
        "visual_intent": "detail_focus",
        "visual_technique": "in_scene_cue",
        "shot_type": "medium",
        "transition": "establishing",
        "visual_description": "In-scene map of Tambora and the Indonesian archipelago hanging on the rustic parlor wall.",
        "focal_point": "Framed in-scene map of Tambora",
    }
    # Global orbital view
    scene_orbital = {
        "index": 2,
        "environment": "European Farmhouse",
        "entity_bindings": {"environment_id": "env_europe"},
        "visual_intent": "establishing",
        "visual_technique": "cinematic_illustration",
        "shot_type": "wide",
        "transition": "location_change",
        "visual_description": "Global view from space showing aerosol cloud spreading from Tambora across continents toward Europe.",
        "focal_point": "Global view of aerosol cloud",
    }
    warnings = validate_connected_storyboard([scene_map, scene_orbital], video_bible_fixture)
    assert not any("[Contradiction]" in w for w in warnings)


def test_check_e2_permits_explicit_multi_location_framing(sample_video_bible):
    """Check E2 allows multi-location compositions such as split-screens, cutaways, and maps."""
    scenes = [
        {
            "index": 1,
            "environment": "European Farmland",
            "entity_bindings": {"environment_id": "env_european_farms"},
            "visual_intent": "narrative_resolution",
            "visual_technique": "split_screen_composite",
            "shot_type": "split",
            "transition": "resolution_return",
            "visual_description": "Split-screen composition contrasting the smoking caldera vents in Sumbawa on the left with frost-covered European fields on the right.",
            "focal_point": "Side-by-side contrast between caldera and cold farmland",
            "image_prompt": "Split-screen dual view of Indonesian volcano and European farmland.",
        }
    ]
    warnings = validate_connected_storyboard(scenes, sample_video_bible)
    assert not any("Check E2" in w for w in warnings)


def test_spatial_cutaway_is_valid_transition():
    """Verify spatial_cutaway is recognized as a valid location transition."""
    scenes = [
        {
            "index": 1,
            "environment": "Location A",
            "entity_bindings": {"environment_id": "env_a"},
            "visual_intent": "environmental_context",
            "visual_technique": "cinematic_illustration",
            "shot_type": "wide",
            "transition": "spatial_cutaway",
            "visual_description": "Overview of location A.",
            "focal_point": "Center",
            "image_prompt": "Clean landscape of location A.",
        }
    ]
    entities = {"environments": [{"id": "env_a", "name": "Location A"}]}
    warnings = validate_connected_storyboard(scenes, {"entities": entities})
    assert not any("invalid location transition" in w.lower() for w in warnings)


# ==============================================================================
# 4. Progression & Anti-Repetition Validation
# ==============================================================================

def test_check_f2_detects_consecutive_diagram_repetition():
    """Check F2 flags redundant consecutive diagram beats as advisory."""
    scenes = [
        {
            "index": i,
            "environment": f"Env {i}",
            "entity_bindings": {"concept_id": "same_concept"},
            "visual_intent": "mechanism_explanation",
            "visual_technique": "diagram",
            "shot_type": "diagram",
            "transition": "none",
            "visual_description": "Static technical diagram tracking identical schematic.",
            "focal_point": "Identical schematic flow",
            "image_prompt": "Technical schematic.",
        }
        for i in range(1, 4)
    ]
    warnings = validate_connected_storyboard(scenes, {"entities": {"environments": []}})
    assert any("[Advisory]" in w and "consecutive diagrams" in w for w in warnings)


def test_diagram_sequential_progression_allowed():
    """Verify 3 consecutive diagrams explaining sequential stages of a process pass Check F2 without warnings."""
    scenes = [
        {
            "index": 1,
            "entity_bindings": {"concept_id": "concept_stage_1"},
            "visual_intent": "mechanism_explanation",
            "visual_technique": "diagram",
            "shot_type": "diagram",
            "transition": "explanatory_cutaway",
            "visual_description": "Initial sulfur dioxide injection into the stratosphere.",
            "focal_point": "Stratospheric injection stage",
        },
        {
            "index": 2,
            "entity_bindings": {"concept_id": "concept_stage_2"},
            "visual_intent": "mechanism_explanation",
            "visual_technique": "diagram",
            "shot_type": "diagram",
            "transition": "explanatory_cutaway",
            "visual_description": "Chemical conversion of sulfur dioxide into sulfate aerosols.",
            "focal_point": "Chemical conversion phase",
        },
        {
            "index": 3,
            "entity_bindings": {"concept_id": "concept_stage_3"},
            "visual_intent": "mechanism_explanation",
            "visual_technique": "diagram",
            "shot_type": "diagram",
            "transition": "explanatory_cutaway",
            "visual_description": "Solar radiation reflection and global cooling mechanism.",
            "focal_point": "Solar backscattering stage",
        },
    ]
    warnings = validate_connected_storyboard(scenes, {"entities": {"environments": []}})
    assert not any("consecutive diagrams" in w for w in warnings)


def test_check_i_validates_opening_and_resolution_intent():
    """Check I flags weak opening hooks and abrupt ending intents with advisory feedback."""
    scenes = [
        {
            "index": 1,
            "environment": "Room",
            "entity_bindings": {},
            "visual_intent": "routine_setup",  # Weak intent for opening beat
            "visual_technique": "cinematic_illustration",
            "shot_type": "medium",
            "transition": "none",
            "visual_description": "A man sits at a desk.",
            "focal_point": "Desk",
            "image_prompt": "A man sits at a desk.",
        },
        {
            "index": 2,
            "environment": "Room",
            "entity_bindings": {},
            "visual_intent": "mechanism_explanation",  # Ending without closure/resolution
            "visual_technique": "cinematic_illustration",
            "shot_type": "wide",
            "transition": "none",
            "visual_description": "Explaining the gear mechanics.",
            "focal_point": "Gears",
            "image_prompt": "Gears in motion.",
        },
    ]
    warnings = validate_connected_storyboard(scenes, {"entities": {"environments": []}})
    assert any("[Advisory] Scene 1 visual_intent is 'routine_setup'" in w for w in warnings)
    assert any("[Advisory] Scene 2 visual_intent is 'mechanism_explanation'" in w for w in warnings)


# ==============================================================================
# 5. Negative Constraint Optimization & Deduplication
# ==============================================================================

def test_extract_world_negative_constraints_filters_planning_commentary():
    """Verify editorial, layout, and narrative planning instructions are filtered out of negative constraints."""
    art_dir = {"negative_anchors": ["photorealism", "blurry watercolor"]}
    cd = {
        "creative_constraints": [
            "Do not depict an ocean literally vanishing or imply that an earthquake in 1819 triggered Tambora’s eruption; treat the disappearing ocean as a metaphorical hook.",
            "Do not crowd the lower caption-safe area.",
            "Never show modern clothing or modern technology.",
        ]
    }
    negs = _extract_world_negative_constraints(art_dir=art_dir, creative_direction=cd, beat_context_text="Farmland crops")
    # Planning-only sentence with semicolons and narrative instructions is omitted
    assert not any("treat as a metaphorical hook" in n for n in negs)
    assert not any("caption-safe" in n for n in negs)
    # Genuine visual constraints are kept
    assert any("modern clothing" in n for n in negs)


def test_extract_world_negative_constraints_general_purpose():
    """Verify negative constraints preserve era/historical and style constraints without hardcoded topic keywords."""
    art_dir = {"negative_anchors": ["photorealism"]}
    cd = {
        "creative_constraints": [
            "Never depict modern clothing or modern technology",
            "Do not show modern vehicles or power lines",
            "Editorial note: ensure pacing remains urgent; do not resolve conflict too early",
        ]
    }
    negs = _extract_world_negative_constraints(art_dir=art_dir, creative_direction=cd, beat_context_text="European wheat field")
    # Planning-only sentence with semicolon is excluded
    assert not any("urgent" in n.lower() for n in negs)
    # Genuine visual constraints are kept
    assert any("modern clothing" in n.lower() for n in negs)
    assert any("modern vehicles" in n.lower() for n in negs)


def test_standalone_diagram_avoids_registered_environments(sample_video_bible):
    """Verify standalone conceptual diagrams derive negative setting rules from registered environment names."""
    scene = {
        "visual_intent": "mechanism_explanation",
        "visual_technique": "diagram",
        "shot_type": "diagram",
        "transition": "explanatory_cutaway",
        "entity_bindings": {
            "environment_id": None,
            "concept_id": "concept_aerosol_veil",
            "character_id": None,
        },
        "visual_description": "Schematic flow showing aerosol particles scattering solar radiation.",
        "focal_point": "Deflection of incoming solar rays",
    }
    prompt = compose_final_prompt(
        scene=scene,
        video_bible=sample_video_bible,
        preset=get_style("stylized_2d_narrative"),
    )
    # Avoids settings derived directly from registered environments: "Tambora Volcanic Caldera" and "European Farmland"
    prompt_lower = prompt.lower()
    assert "no tambora volcanic caldera setting" in prompt_lower or "no european farmland setting" in prompt_lower
    # Does not have ad-hoc "no bedroom setting" since no bedroom exists in this video's environments
    assert "no bedroom setting" not in prompt_lower


def test_canonical_visual_techniques_enforced():
    """Verify visual_technique is strictly validated against the 5 canonical values."""
    from app.steps.storyboard import ALLOWED_VISUAL_TECHNIQUES, validate_storyboard_soft_checks

    assert set(ALLOWED_VISUAL_TECHNIQUES) == {
        "cinematic_illustration",
        "physical_close_up",
        "in_scene_cue",
        "cutaway",
        "diagram",
    }

    # Non-canonical technique triggers warning in validate_storyboard_soft_checks
    scenes = [
        {
            "narration": "The world changed completely in one single explosive night.",
            "visual_intent": "establishing",
            "visual_technique": "arbitrary_free_text_technique",
            "shot_type": "wide",
            "image_prompt": "Wide landscape view.",
        }
    ]
    warnings = validate_storyboard_soft_checks(scenes, "The world changed completely in one single explosive night.")
    assert any("invalid visual_technique 'arbitrary_free_text_technique'" in w for w in warnings)


def test_normalize_negative_constraint_deduplication():
    """Verify constraint normalization collapses semantically redundant variations."""
    c1 = "Do not use photorealistic or 3D-rendered imagery"
    c2 = "photorealism"
    norm1 = _normalize_negative_constraint(c1)
    norm2 = _normalize_negative_constraint(c2)
    assert "photoreal" in norm1 and "photoreal" in norm2


# ==============================================================================
# 6. False-Positive Prevention in Forbidden Words
# ==============================================================================

def test_detect_forbidden_prompt_words_ignores_layout_guidance():
    """Verify negative layout guidance like 'no text' and 'caption-safe' does not trigger false positives."""
    text = (
        "Vertical 9:16 framing, main subject in upper third, calm uncluttered lower third, "
        "no text, no letters, no logos, leave caption-safe space."
    )
    violations = detect_forbidden_prompt_words(text)
    assert violations == []

    # But genuine forbidden words (e.g. asking to draw text or words) are still flagged
    bad_text = "A sign with bold text reading DANGER in glowing letters."
    violations_bad = detect_forbidden_prompt_words(bad_text)
    assert len(violations_bad) > 0
