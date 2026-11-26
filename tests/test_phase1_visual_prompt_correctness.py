"""Regression tests for Phase 1: Visual Prompt & Storyboard Correctness.

Covers:
1. Visual-style prompts do not contain unrelated concrete example scenes or props from style descriptions.
2. General lighting and style identity are preserved after cleaning the descriptions.
3. Current storyboard entity catalog fields are correctly included in generated prompts.
4. Legacy entity data remains supported where compatibility is necessary.
5. World/era constraints reach image prompts even when character_id is absent.
6. Character-specific constraints remain character-specific and do not leak into unrelated scenes.
7. Existing legacy visual-style behavior remains unchanged.
"""

from __future__ import annotations

import re
from app.steps.prompt_compose import (
    COMPOSITION_RULES,
    compose_final_prompt,
)
from app.steps.storyboard import build_storyboard_prompt
from app.styles.registry import (
    LEGACY_DEFAULT_PRESET,
    STICKMAN_TACTICAL_EXPLAINER_PRESET,
    STYLIZED_2D_NARRATIVE_PRESET,
    get_style,
)


# ==============================================================================
# 1. Concrete Example Leakage Prevention
# ==============================================================================

def test_visual_style_prompts_do_not_contain_concrete_example_leakage():
    """Verify that image prompts do not leak bedroom lamps, stormy windows, smartphones,

    or other concrete props from style drawing rules.
    """
    preset = get_style("stylized_2d_narrative")
    scene = {
        "index": 1,
        "visual_description": "Mount Tambora erupts beneath a dark sky, ash climbing into the atmosphere.",
        "camera": {"shot_type": "wide", "camera_angle": "low_angle", "focal_point_description": "Erupting volcano summit"},
        "entity_bindings": {"character_id": None, "environment_id": "env_volcano", "concept_id": None, "object_ids": []},
    }
    bible = {
        "art_direction": {
            "style_family": "illustrated",
            "visual_genre": "2d_cartoon",
            "primary_palette_hex": ["#FFFFFF", "#212529", "#FF5722"],
            "lighting_philosophy": "Atmospheric dramatic lighting with strong warm and cool contrast",
            "negative_anchors": ["photorealism", "3D CGI"],
        },
        "entities": {
            "environments": [
                {
                    "id": "env_volcano",
                    "name": "Mount Tambora Coastline",
                    "spatial_layout": "Steep volcanic cone rising above open sea",
                    "lighting_setup": "Volcanic orange glow against charcoal ash shadows",
                }
            ],
        },
    }

    prompt = compose_final_prompt(scene, video_bible=bible, preset=preset)

    # Specific leaked objects that must NEVER appear in the prompt
    banned_leaks = [
        "bedroom",
        "table lamp",
        "stormy exterior window",
        "smartphone",
        "alert displays",
        "vehicle wreckage",
        "backpacks",
        "ticking clocks",
        "heartbeat waveforms",
        "hoodies",
        "door peephole",
    ]
    for leak in banned_leaks:
        assert leak not in prompt.lower(), f"Leaked concrete object '{leak}' found in prompt: {prompt}"

    # Verify no parenthetical illustrative e.g. clauses remain in the prompt
    assert "(e.g." not in prompt.lower()
    assert "(for example" not in prompt.lower()


def test_stickman_prompt_does_not_contain_server_or_weapon_examples():
    """Verify stickman prompt does not leak servers, weapons, or water jugs from rules."""
    preset = get_style("stickman_tactical_explainer")
    scene = {
        "index": 1,
        "visual_description": "A hiker looking at a forest map.",
        "camera": {"shot_type": "medium", "camera_angle": "eye_level", "focal_point_description": "Map in hands"},
        "entity_bindings": {"character_id": None, "environment_id": None, "concept_id": None, "object_ids": []},
    }
    bible = {
        "art_direction": {
            "style_family": "illustrated",
            "visual_genre": "stickman_illustrated",
            "primary_palette_hex": ["#FFFFFF", "#1A1A1A"],
            "negative_anchors": ["photorealism"],
        },
        "entities": {},
    }

    prompt = compose_final_prompt(scene, video_bible=bible, preset=preset)

    banned = ["servers with grid panels", "weapons with bold outlines", "water jugs", "windows, lamps, candles"]
    for b in banned:
        assert b not in prompt.lower(), f"Leaked example '{b}' found in prompt: {prompt}"


# ==============================================================================
# 2. General Lighting and Style Identity Preservation
# ==============================================================================

def test_general_lighting_and_style_identity_preserved():
    """Verify that general lighting aesthetics and visual identity are preserved after cleaning."""
    style_2d = STYLIZED_2D_NARRATIVE_PRESET
    assert "stylized 2D digital cartoon animation art" in style_2d.permanent_prompt_anchors
    assert "clean bold outlines" in style_2d.permanent_prompt_anchors
    assert "two-tone cel-shading" in style_2d.permanent_prompt_anchors
    # "modern" removed to prevent anachronistic bias, while preserving graphic novel aesthetic
    assert "expressive graphic novel character aesthetic" in style_2d.permanent_prompt_anchors
    assert "expressive modern graphic novel" not in style_2d.permanent_prompt_anchors

    # General lighting direction preserved without concrete bedroom/lamp examples
    lighting = style_2d.rules.lighting_style
    assert "dramatic lighting" in lighting.lower()
    assert "warm and cool" in lighting.lower()
    assert "bedroom" not in lighting.lower()
    assert "lamp" not in lighting.lower()

    stickman = STICKMAN_TACTICAL_EXPLAINER_PRESET
    assert "clean 2D animated stickman illustration style" in stickman.permanent_prompt_anchors
    assert "directional ambient lighting" in stickman.rules.lighting_style.lower()
    assert "windows, lamps, candles" not in stickman.rules.lighting_style.lower()


# ==============================================================================
# 3. Current Storyboard Entity Catalog Schema Fields Included
# ==============================================================================

def test_current_storyboard_entity_catalog_fields_reach_prompt():
    """Verify that canonical schema fields (objects, concepts, visual_anchor, spatial_layout) reach the prompt."""
    rules_text = "Storyboard planning guidelines"
    script_data = {
        "title": "Tambora 1815",
        "hook": "How could an ocean vanish?",
        "narration": "How could an ocean vanish? Mount Tambora darkened skies.",
        "angle": "what goes wrong without it",
    }
    bible = {
        "entities": {
            "characters": [
                {
                    "id": "char_farmer",
                    "name": "Rural Farmer",
                    "visual_anchor": "Weathered 30s farmer with rounded head and dark-brown hair",
                    "apparel_anchor": "worn dark-brown coat and cream linen shirt",
                    "forbidden_mutations": ["No modern clothing"],
                }
            ],
            "environments": [
                {
                    "id": "env_farmland",
                    "name": "Distant Farmland",
                    "spatial_layout": "Open rural fields extending toward dim clouded sky",
                }
            ],
            "objects": [
                {
                    "id": "obj_crops",
                    "name": "Struggling Crops",
                    "visual_description": "Low uneven rows of frost-damaged wheat",
                    "material_and_finish": "two-tone cel-shaded muted olive foliage",
                }
            ],
            "concepts": [
                {
                    "id": "concept_aerosols",
                    "name": "Sulfur Aerosol Veil",
                    "visual_analogy": "High-altitude stippled particle veil reflecting incoming sun rays",
                    "concept_explained": "Atmospheric particles reflect sunlight back to space",
                }
            ],
        },
        "art_direction": {
            "visual_genre": "2d_cartoon",
            "lighting_philosophy": "Atmospheric dramatic lighting",
            "rendering_texture": "Clean tapered lines with cel-shading",
            "primary_palette_hex": ["#FFFFFF", "#212529"],
            "negative_anchors": ["photorealism"],
        },
    }

    prompt = build_storyboard_prompt(
        rules_text=rules_text,
        topic="The Ocean That Disappeared",
        script_data=script_data,
        visual_style="Stylized 2D Narrative Drama",
        video_bible_data=bible,
    )

    # 1. Objects section must be present and contain the canonical object
    assert "- **Objects & Props**:" in prompt
    assert "obj_crops" in prompt
    assert "Struggling Crops" in prompt
    assert "Low uneven rows of frost-damaged wheat" in prompt
    assert "two-tone cel-shaded muted olive foliage" in prompt

    # 2. Concepts section must be present and contain the canonical concept
    assert "- **Concepts & Diagrams**:" in prompt
    assert "concept_aerosols" in prompt
    assert "Sulfur Aerosol Veil" in prompt
    assert "High-altitude stippled particle veil reflecting incoming sun rays" in prompt

    # 3. Characters must include visual anchor and apparel (not None)
    assert "char_farmer" in prompt
    assert "Rural Farmer" in prompt
    assert "Weathered 30s farmer with rounded head and dark-brown hair" in prompt
    assert "worn dark-brown coat and cream linen shirt" in prompt
    assert "(None)" not in prompt

    # 4. Environments must include spatial layout (not None)
    assert "env_farmland" in prompt
    assert "Distant Farmland - Open rural fields extending toward dim clouded sky" in prompt
    assert "- None" not in prompt


# ==============================================================================
# 4. Legacy Entity Data Compatibility Supported
# ==============================================================================

def test_legacy_entity_data_remains_supported():
    """Verify that legacy dictionary keys (objects_and_props, concepts_and_diagrams, visual_summary, description)

    continue to work without errors or dropped data.
    """
    rules_text = "Storyboard planning guidelines"
    script_data = {"title": "Legacy Test", "narration": "Sample narration text."}
    legacy_bible = {
        "entities": {
            "characters": [
                {
                    "id": "char_old",
                    "name": "Legacy Sleeper",
                    "visual_summary": "Young sleeper resting on pillow",
                    "clothing_and_style": "slate-blue pajamas",
                }
            ],
            "environments": [
                {
                    "id": "env_old",
                    "name": "Legacy Bedroom",
                    "description": "Cozy dimly lit bedroom with wooden furniture",
                }
            ],
            "objects_and_props": [
                {
                    "id": "obj_old",
                    "name": "Analog Clock",
                    "visual_details": "Brass ticking clock on bedside table",
                }
            ],
            "concepts_and_diagrams": [
                {
                    "id": "concept_old",
                    "name": "Brainwave Transition",
                    "diagram_style": "Sinusoidal waveform shifting from alpha to theta",
                }
            ],
        },
        "art_direction": {"visual_genre": "styled_3d", "negative_anchors": []},
    }

    prompt = build_storyboard_prompt(
        rules_text=rules_text,
        topic="Legacy Sleep Test",
        script_data=script_data,
        visual_style="Clean 3D Studio",
        video_bible_data=legacy_bible,
    )

    # Legacy keys must be resolved and formatted
    assert "obj_old" in prompt
    assert "Analog Clock (Brass ticking clock on bedside table)" in prompt
    assert "concept_old" in prompt
    assert "Brainwave Transition (Sinusoidal waveform shifting from alpha to theta)" in prompt
    assert "char_old" in prompt
    assert "Legacy Sleeper (Young sleeper resting on pillow, wearing slate-blue pajamas)" in prompt
    assert "env_old" in prompt
    assert "Legacy Bedroom - Cozy dimly lit bedroom with wooden furniture" in prompt
    assert "(None)" not in prompt
    assert "- None" not in prompt


# ==============================================================================
# 5. World and Era Constraints Reach Prompts When Character Is Absent
# ==============================================================================

def test_world_era_constraints_reach_image_prompts_when_character_absent():
    """Verify that historical and world negative constraints reach image prompts on scenes without character_id."""
    preset = get_style("stylized_2d_narrative")
    scene = {
        "index": 7,
        "visual_description": "Cold, dim fields under overcast sky lead toward horizon, showing crop damage.",
        "camera": {"shot_type": "wide", "camera_angle": "eye_level", "focal_point_description": "Frost-bitten crops"},
        "entity_bindings": {
            "character_id": None,  # Character is absent
            "environment_id": "env_farmland",
            "concept_id": None,
            "object_ids": [],
        },
    }
    bible = {
        "art_direction": {
            "style_family": "illustrated",
            "visual_genre": "2d_cartoon",
            "primary_palette_hex": ["#FFFFFF", "#212529"],
            "negative_anchors": [
                "photorealism",
                "3D volumetric render",
                "no modern clothing or technology",
                "no modern buildings",
            ],
            "negative_prompt_tokens": "literal disappearing ocean, modern vehicles, airplanes",
        },
        "entities": {
            "characters": [
                {
                    "id": "char_farmer",
                    "name": "Rural Farmer",
                    "visual_anchor": "Farmer with dark coat",
                    "apparel_anchor": "cream shirt",
                    "forbidden_mutations": ["Never change the brown coat", "Never change dark-brown hair"],
                }
            ],
            "environments": [
                {
                    "id": "env_farmland",
                    "name": "Distant Farmland",
                    "spatial_layout": "Rural fields under overcast sky",
                    "lighting_setup": "Dim cool daylight",
                }
            ],
        },
    }
    cd = {
        "creative_constraints": [
            "Never depict modern clothing, accessories, or technology",
            "Never depict the ocean literally disappearing",
            "State the eruption year accurately as 1815",
            "Keep the farmer fictional and representative",
        ],
    }

    prompt = compose_final_prompt(scene, video_bible=bible, creative_direction=cd, preset=preset)

    # 1. World and era negative constraints MUST be in the Avoid: section
    assert "Avoid:" in prompt
    avoid_segment = prompt.split("Avoid:")[1]
    assert "no modern clothing or technology" in avoid_segment or "modern clothing" in avoid_segment
    assert "no modern buildings" in avoid_segment or "modern buildings" in avoid_segment
    assert "modern vehicles" in avoid_segment
    assert "airplanes" in avoid_segment
    assert "literal disappearing ocean" in avoid_segment or "ocean literally disappearing" in avoid_segment

    # 2. Editorial script instructions must NOT leak into the prompt
    assert "State the eruption year" not in prompt
    assert "fictional and representative" not in prompt

    # 3. Character-specific mutations must NOT be present (character is absent)
    assert "brown coat" not in avoid_segment
    assert "dark-brown hair" not in avoid_segment
    assert "Rural Farmer" not in prompt


# ==============================================================================
# 6. Character-Specific Constraints Remain Character-Specific
# ==============================================================================

def test_character_specific_constraints_do_not_leak_into_unrelated_scenes():
    """Verify character-specific mutations are applied ONLY when that character is bound."""
    preset = get_style("stylized_2d_narrative")
    bible = {
        "art_direction": {
            "style_family": "illustrated",
            "visual_genre": "2d_cartoon",
            "primary_palette_hex": ["#FFFFFF", "#212529"],
            "negative_anchors": ["photorealism", "no modern clothing"],
        },
        "entities": {
            "characters": [
                {
                    "id": "char_hero",
                    "name": "Hero",
                    "visual_anchor": "Hero with silver hair",
                    "apparel_anchor": "crimson velvet cape",
                    "forbidden_mutations": ["Never change the crimson velvet cape", "Never add eyeglasses"],
                },
                {
                    "id": "char_villain",
                    "name": "Villain",
                    "visual_anchor": "Tall figure with obsidian armor",
                    "apparel_anchor": "spiked obsidian armor",
                    "forbidden_mutations": ["Never remove the spiked helmet"],
                },
            ],
            "environments": [
                {
                    "id": "env_castle",
                    "name": "Castle Hall",
                    "spatial_layout": "Stone corridor with torchlight",
                    "lighting_setup": "Warm torchlight",
                }
            ],
        },
    }

    # Scene 1: Bound to char_hero
    scene_hero = {
        "index": 1,
        "visual_description": "Hero stands in the castle corridor.",
        "camera": {"shot_type": "medium", "camera_angle": "eye_level", "focal_point_description": "Hero standing"},
        "entity_bindings": {"character_id": "char_hero", "environment_id": "env_castle", "concept_id": None, "object_ids": []},
    }

    # Scene 2: No character bound (environment vista)
    scene_empty = {
        "index": 2,
        "visual_description": "Wide empty castle corridor stretching into darkness.",
        "camera": {"shot_type": "wide", "camera_angle": "eye_level", "focal_point_description": "Corridor perspective"},
        "entity_bindings": {"character_id": None, "environment_id": "env_castle", "concept_id": None, "object_ids": []},
    }

    # Scene 3: Bound to char_villain
    scene_villain = {
        "index": 3,
        "visual_description": "Villain emerges from the shadows.",
        "camera": {"shot_type": "medium", "camera_angle": "low_angle", "focal_point_description": "Villain stance"},
        "entity_bindings": {"character_id": "char_villain", "environment_id": "env_castle", "concept_id": None, "object_ids": []},
    }

    prompt_hero = compose_final_prompt(scene_hero, video_bible=bible, preset=preset)
    prompt_empty = compose_final_prompt(scene_empty, video_bible=bible, preset=preset)
    prompt_villain = compose_final_prompt(scene_villain, video_bible=bible, preset=preset)

    # 1. Global world constraint "no modern clothing" is present in ALL THREE scenes
    assert "no modern clothing" in prompt_hero
    assert "no modern clothing" in prompt_empty
    assert "no modern clothing" in prompt_villain

    # 2. Hero mutations appear ONLY in Scene 1
    assert "crimson velvet cape" in prompt_hero
    assert "crimson velvet cape" not in prompt_empty
    assert "crimson velvet cape" not in prompt_villain

    # 3. Villain mutations appear ONLY in Scene 3
    assert "spiked helmet" in prompt_villain
    assert "spiked helmet" not in prompt_hero
    assert "spiked helmet" not in prompt_empty


# ==============================================================================
# 7. Legacy Visual-Style Behavior Preserved
# ==============================================================================

def test_existing_legacy_visual_style_behavior_remains_unchanged():
    """Verify that Clean 3D Studio (legacy_default) and standalone concepts preserve exact historical behavior."""
    preset = LEGACY_DEFAULT_PRESET
    assert preset.id == "legacy_default"
    assert "Clean 3D render" in preset.permanent_prompt_anchors
    assert "soft studio lighting" in preset.permanent_prompt_anchors
    assert "smooth matte materials" in preset.permanent_prompt_anchors

    scene = {
        "index": 1,
        "visual_description": "A smooth 3D silicon cube floating in abstract space.",
        "camera": {"shot_type": "close-up", "camera_angle": "eye_level", "focal_point_description": "Silicon cube"},
        "entity_bindings": {"character_id": None, "environment_id": None, "concept_id": None, "object_ids": []},
    }
    bible = {
        "art_direction": {
            "style_family": "cinematic",
            "visual_genre": "styled_3d",
            "rendering_texture": "smooth matte materials",
            "lighting_philosophy": "soft studio lighting",
            "primary_palette_hex": ["#FFFFFF", "#E0E0E0"],
            "negative_anchors": ["harsh shadows", "2D flat vector"],
        },
        "entities": {},
    }

    prompt = compose_final_prompt(scene, video_bible=bible, preset=preset)

    assert "Styled 3d visual style" in prompt
    assert "smooth matte materials" in prompt
    assert "Negative constraints:" in prompt
    assert "no harsh shadows" in prompt
    assert "no 2D flat vector" in prompt
    assert COMPOSITION_RULES in prompt
