"""Parity regression tests verifying the legacy preset reproduces the existing prompt output exactly."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from app.steps.prompt_compose import compose_final_prompt
from app.styles import (
    DEFAULT_STYLE_ID,
    ReferenceMetadata,
    VisualStylePreset,
    get_style,
    list_styles,
)

FIXTURES_PATH = Path(__file__).resolve().parent / "fixtures" / "legacy_prompts_golden.json"


@pytest.fixture(scope="module")
def golden_baselines() -> dict[str, dict]:
    """Load genuine unmodified baseline outputs captured from the production codebase."""
    assert FIXTURES_PATH.exists(), f"Golden baselines file missing at {FIXTURES_PATH}"
    with open(FIXTURES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def legacy_preset() -> VisualStylePreset:
    return get_style("legacy_default")


# ==============================================================================
# Golden Parity Tests: Character-for-Character Equality Against Baseline
# ==============================================================================

def test_golden_parity_studio_closeup(golden_baselines, legacy_preset):
    """Case 1: Studio Close-up (smartphone, no environment)."""
    case = golden_baselines["studio_closeup"]
    actual = compose_final_prompt(
        scene=case["scene"],
        visual_style=", ".join(legacy_preset.permanent_prompt_anchors),
    )
    assert actual == case["expected_prompt"]
    assert "pastel gradient background" in actual
    assert "shallow depth of field" in actual


def test_golden_parity_wide_environment(golden_baselines, legacy_preset):
    """Case 2: Wide shot in specific environment (strips pastel gradient & shallow DOF)."""
    case = golden_baselines["wide_environment"]
    actual = compose_final_prompt(
        scene=case["scene"],
        visual_style=", ".join(legacy_preset.permanent_prompt_anchors),
    )
    assert actual == case["expected_prompt"]
    assert "pastel gradient background" not in actual
    assert "shallow depth of field" not in actual


def test_golden_parity_cutaway_diagram(golden_baselines, legacy_preset):
    """Case 3: Technical cutaway diagram (strips shallow DOF, preserves studio lighting)."""
    case = golden_baselines["cutaway_diagram"]
    actual = compose_final_prompt(
        scene=case["scene"],
        visual_style=", ".join(legacy_preset.permanent_prompt_anchors),
    )
    assert actual == case["expected_prompt"]
    assert "shallow depth of field" not in actual


def test_golden_parity_aerial_cityscape(golden_baselines, legacy_preset):
    """Case 4: Aerial shot of highway landscape (strips pastel and shallow DOF)."""
    case = golden_baselines["aerial_cityscape"]
    actual = compose_final_prompt(
        scene=case["scene"],
        visual_style=", ".join(legacy_preset.permanent_prompt_anchors),
    )
    assert actual == case["expected_prompt"]
    assert "pastel gradient background" not in actual
    assert "shallow depth of field" not in actual


def test_golden_parity_character_with_anchor(golden_baselines, legacy_preset):
    """Case 5: Character with subject anchor and uses_anchor=True."""
    case = golden_baselines["character_with_anchor"]
    actual = compose_final_prompt(
        scene=case["scene"],
        subject_anchor=case["subject_anchor"],
        visual_style=", ".join(legacy_preset.permanent_prompt_anchors),
    )
    assert actual == case["expected_prompt"]
    assert "Captain Miller" in actual


def test_golden_parity_connected_video_bible_beat(golden_baselines):
    """Case 6: Connected Video Bible prompt composition."""
    case = golden_baselines["connected_video_bible_beat"]
    actual = compose_final_prompt(
        scene=case["scene"],
        video_bible=case["video_bible"],
    )
    assert actual == case["expected_prompt"]
    assert "Styled 3d visual style" in actual
    assert "no 2D sketches" in actual


def test_golden_parity_conceptual_diagram_beat(golden_baselines):
    """Case 7: Standalone conceptual diagram beat (sanitizes bedroom/sleeper references)."""
    case = golden_baselines["conceptual_diagram_beat"]
    actual = compose_final_prompt(
        scene=case["scene"],
        video_bible=case["video_bible"],
    )
    assert actual == case["expected_prompt"]
    assert "Visual diagram concept: Sleep Onset" in actual
    assert "no bedroom setting" in actual


# ==============================================================================
# Registry and Schema Adherence Tests
# ==============================================================================

def test_registry_contains_all_three_presets():
    styles = list_styles()
    style_ids = [s.id for s in styles]
    assert "legacy_default" in style_ids
    assert "stickman_tactical_explainer" in style_ids
    assert "stylized_2d_narrative" in style_ids
    assert len(styles) == 3


def test_registry_get_style_fallback():
    assert get_style(None).id == DEFAULT_STYLE_ID
    assert get_style("").id == DEFAULT_STYLE_ID
    assert get_style("non_existent_preset").id == DEFAULT_STYLE_ID
    assert get_style("stickman_tactical_explainer").id == "stickman_tactical_explainer"
    assert get_style("STYLIZED_2D_NARRATIVE").id == "stylized_2d_narrative"


def test_presets_conform_to_schema():
    for preset in list_styles():
        assert isinstance(preset, VisualStylePreset)
        assert preset.id
        assert preset.display_name
        assert preset.version >= 1
        assert len(preset.permanent_prompt_anchors) >= 1
        assert preset.rules.character_anatomy
        assert preset.rules.linework_and_outlines
        assert preset.rules.color_and_shading
        assert preset.art_direction.style_family
        assert preset.art_direction.visual_genre


# ==============================================================================
# Reference Metadata Compatibility Logic Tests
# ==============================================================================

def test_reference_metadata_style_role():
    ref = ReferenceMetadata(
        entity_id="global_style",
        role="style",
        asset_path="references/stickman_tactical/style_benchmark.png",
        visual_style_id="stickman_tactical_explainer",
        style_version=1,
    )
    assert ref.is_compatible_with("stickman_tactical_explainer", 1) is True
    assert ref.is_compatible_with("stickman_tactical_explainer", 2) is True  # style role ignores minor version
    assert ref.is_compatible_with("legacy_default", 1) is False
    assert ref.is_compatible_with("stylized_2d_narrative", 1) is False


def test_reference_metadata_semantic_role():
    # Semantic references (CAD drawings, real photos, structural blueprints) are always compatible across styles
    ref = ReferenceMetadata(
        entity_id="obj_mri_scanner",
        role="semantic",
        asset_path="references/objects/mri_schematic.png",
    )
    assert ref.is_compatible_with("legacy_default", 1) is True
    assert ref.is_compatible_with("stickman_tactical_explainer", 1) is True
    assert ref.is_compatible_with("stylized_2d_narrative", 1) is True


def test_reference_metadata_style_specific_role():
    # Style-specific references (e.g. 3D rendered character) must NEVER contaminate another style
    ref_3d = ReferenceMetadata(
        entity_id="char_pilot",
        role="style_specific",
        asset_path="storage/video_123/entities/char_pilot_3d.png",
        visual_style_id="legacy_default",
        style_version=1,
    )
    assert ref_3d.is_compatible_with("legacy_default", 1) is True
    assert ref_3d.is_compatible_with("legacy_default", 2) is False  # Version mismatch rejected
    assert ref_3d.is_compatible_with("stickman_tactical_explainer", 1) is False  # Incompatible style rejected
    assert ref_3d.is_compatible_with("stylized_2d_narrative", 1) is False  # Incompatible style rejected
