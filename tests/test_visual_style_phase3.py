"""Tests for Phase 3: Reference Selection Engine, Entity Adaptation, and GPT Image Edits Integration."""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.images.base import BaseImageProvider
from app.images.exceptions import (
    ImageConfigError,
    ImageContentBlockedError,
    ImageError,
    ImageProviderError,
    ImageRateLimitError,
    ImageTimeoutError,
)
from app.images.mock import MockImageProvider
from app.images.openai import OpenAIImageProvider
from app.images.types import ImageResult
from app.models import Video, VideoStep
from app.schemas import (
    CharacterEntity,
    ConceptEntity,
    EnvironmentEntity,
    ObjectEntity,
)
from app.steps.images import generate_images_step
from app.styles import (
    LEGACY_DEFAULT_PRESET,
    STICKMAN_TACTICAL_EXPLAINER_PRESET,
    STYLIZED_2D_NARRATIVE_PRESET,
    AdaptedEntity,
    ReferenceMetadata,
    ReferenceSelectionEngine,
    adapt_character,
    adapt_concept,
    adapt_environment,
    adapt_object,
    select_scene_references,
)


# ==============================================================================
# 1. Reference Metadata & Compatibility
# ==============================================================================

def test_reference_metadata_legacy_string_normalization():
    """Unadorned string paths are normalized without invented IDs and not assumed style-neutral."""
    norm = ReferenceMetadata.normalize("storage/character_3d.png", default_entity_id="char_1")
    assert norm.entity_id == "char_1"
    assert norm.role is None
    assert norm.asset_path == "storage/character_3d.png"
    assert norm.visual_style_id is None
    assert norm.style_version is None

    # Incompatible with stickman preset: explicit role required
    compat, msg = norm.check_compatibility("stickman_tactical_explainer", active_style_version=1)
    assert not compat
    assert "has no role metadata" in msg
    assert not norm.is_compatible_with("stickman_tactical_explainer", active_style_version=1)

    # Compatible with legacy_default preset
    compat_leg, msg_leg = norm.check_compatibility("legacy_default", active_style_version=1)
    assert compat_leg
    assert "Legacy unannotated reference accepted" in msg_leg
    assert norm.is_compatible_with("legacy_default", active_style_version=1)


def test_reference_metadata_legacy_dict_normalization():
    """Legacy dictionary without role is normalized without inventing role, style ID, or version."""
    raw = {
        "entity_id": "c1",
        "asset_path": "assets/char.png",
        "description": "Original pilot concept",
        "custom_tag": "v1_export",
    }
    norm = ReferenceMetadata.normalize(raw)
    assert norm.entity_id == "c1"
    assert norm.role is None
    assert norm.visual_style_id is None
    assert norm.style_version is None
    assert norm.source_metadata.get("custom_tag") == "v1_export"
    assert norm.description == "Original pilot concept"

    # Incompatible with specialized preset, compatible with legacy_default
    assert not norm.is_compatible_with("stickman_tactical_explainer")
    assert norm.is_compatible_with("legacy_default")


def test_reference_role_compatibility_filtering():
    """Verify role-based compatibility filtering across presets and versions."""
    # 1. Semantic references communicate physical structure; compatible across all styles
    sem_ref = ReferenceMetadata(
        entity_id="char_anatomy",
        role="semantic",
        asset_path="assets/silhouette.png",
    )
    assert sem_ref.is_compatible_with("stickman_tactical_explainer")
    assert sem_ref.is_compatible_with("stylized_2d_narrative")
    assert sem_ref.is_compatible_with("legacy_default")

    # 2. Style references communicate medium; only compatible with own style
    style_ref = ReferenceMetadata(
        entity_id="stickman_benchmark",
        role="style",
        asset_path="benchmarks/stickman.png",
        visual_style_id="stickman_tactical_explainer",
        style_version=1,
    )
    assert style_ref.is_compatible_with("stickman_tactical_explainer")
    assert not style_ref.is_compatible_with("stylized_2d_narrative")
    assert not style_ref.is_compatible_with("legacy_default")

    # 3. Style-specific references require matching visual_style_id and style_version
    spec_ref = ReferenceMetadata(
        entity_id="hero_turnaround",
        role="style_specific",
        asset_path="chars/hero_2d.png",
        visual_style_id="stylized_2d_narrative",
        style_version=1,
    )
    assert spec_ref.is_compatible_with("stylized_2d_narrative", active_style_version=1)
    assert not spec_ref.is_compatible_with("stylized_2d_narrative", active_style_version=2)
    assert not spec_ref.is_compatible_with("stickman_tactical_explainer", active_style_version=1)


# ==============================================================================
# 2. Deterministic Entity Adaptation
# ==============================================================================

def test_adapt_character_preserves_source_and_applies_style_rules():
    """Character adaptation compiles target-style description while keeping source intact."""
    char = CharacterEntity(
        id="c1",
        name="Elena Ramos",
        role="Avionics Engineer",
        visual_anchor="Athletic 30-year-old woman with braided dark hair",
        apparel_anchor="Navy jumpsuit with reflective stripes and tool harness",
        forbidden_mutations=["hair length", "uniform color"],
    )

    # 1. Adapt to Stickman Tactical Explainer
    adapted_stick = adapt_character(char, STICKMAN_TACTICAL_EXPLAINER_PRESET)
    assert adapted_stick.entity_id == "c1"
    assert adapted_stick.entity_type == "character"
    assert "2D stickman illustration style" in adapted_stick.adapted_description
    assert "Elena Ramos (Avionics Engineer)" in adapted_stick.adapted_description
    assert "Athletic 30-year-old woman with braided dark hair" in adapted_stick.adapted_description
    assert "Navy jumpsuit with reflective stripes" in adapted_stick.adapted_description
    assert "clean uniform black outlines" in adapted_stick.adapted_description
    # Source entity remains untouched
    assert char.name == "Elena Ramos"
    assert char.visual_anchor == "Athletic 30-year-old woman with braided dark hair"

    # 2. Adapt to Stylized 2D Narrative
    adapted_2d = adapt_character(char, STYLIZED_2D_NARRATIVE_PRESET)
    assert "stylized 2D graphic illustration style" in adapted_2d.adapted_description
    assert "expressive ink linework, 2-tone cel-shading" in adapted_2d.adapted_description

    # 3. Adapt to Legacy Default
    adapted_legacy = adapt_character(char, LEGACY_DEFAULT_PRESET)
    assert "Clean 3D stylized matte render" in adapted_legacy.adapted_description
    assert "Smooth rounded contours, matte materials" in adapted_legacy.adapted_description


def test_adapt_object_preserves_source_and_applies_style_rules():
    """Object adaptation compiles prop description without modifying original properties."""
    obj = ObjectEntity(
        id="o1",
        name="Turbofan Engine",
        visual_description="High-bypass commercial jet engine with titanium fan blades",
        material_and_finish="Brushed titanium alloy with matte composite cowling",
    )

    stick_obj = adapt_object(obj, STICKMAN_TACTICAL_EXPLAINER_PRESET)
    assert "tactical 2D line-art prop" in stick_obj.adapted_description
    assert "Turbofan Engine" in stick_obj.adapted_description
    assert "titanium fan blades" in stick_obj.adapted_description

    stylized_obj = adapt_object(obj, STYLIZED_2D_NARRATIVE_PRESET)
    assert "stylized 2D graphic novel illustration" in stylized_obj.adapted_description

    legacy_obj = adapt_object(obj, LEGACY_DEFAULT_PRESET)
    assert "Clean 3D matte object" in legacy_obj.adapted_description
    assert "Brushed titanium alloy" in legacy_obj.adapted_description


def test_adapt_environment_and_concept():
    """Environment and Concept adaptation preserve layout, props, and analogies."""
    env = EnvironmentEntity(
        id="e1",
        name="Flight Test Hangar",
        spatial_layout="Wide industrial hangar with high steel rafters and concrete floor",
        lighting_setup="Bright daylight from skylights and halogen work lamps",
        persistent_props=["Yellow gantry crane", "Rolling toolboxes"],
        time_of_day="morning",
    )
    concept = ConceptEntity(
        id="cp1",
        name="Bernoulli Lift",
        concept_explained="Pressure differential creating upward aerodynamic lift",
        visual_analogy="Curved airfoil slicing through parallel airflow streamlines",
        graphical_elements=["High pressure vectors", "Low pressure zone"],
    )

    # Environment under Stickman
    adapted_env = adapt_environment(env, STICKMAN_TACTICAL_EXPLAINER_PRESET)
    assert "1-point perspective architectural line art" in adapted_env.adapted_description
    assert "Flight Test Hangar" in adapted_env.adapted_description
    assert "Yellow gantry crane" in adapted_env.adapted_description

    # Concept under Stickman
    adapted_cp = adapt_concept(concept, STICKMAN_TACTICAL_EXPLAINER_PRESET)
    assert "Tactical schematic visual diagram" in adapted_cp.adapted_description
    assert "Bernoulli Lift" in adapted_cp.adapted_description
    assert "Curved airfoil slicing through parallel airflow streamlines" in adapted_cp.adapted_description
    assert "High pressure vectors" in adapted_cp.adapted_description


# ==============================================================================
# 3. Deterministic Reference Selection Engine
# ==============================================================================

def test_reference_selection_priority_and_budget_enforcement():
    """Engine prioritizes primary entity, secondary entity, and respects budget of 2."""
    engine = ReferenceSelectionEngine(default_budget=2)

    available = [
        ReferenceMetadata(entity_id="c1", role="semantic", asset_path="refs/c1_semantic.png"),
        ReferenceMetadata(entity_id="e1", role="semantic", asset_path="refs/e1_semantic.png"),
        ReferenceMetadata(entity_id="o1", role="semantic", asset_path="refs/o1_semantic.png"),
    ]

    scene = {
        "entity_bindings": {
            "character_id": "c1",
            "environment_id": "e1",
            "object_ids": ["o1"],
        }
    }

    result = engine.select_references(
        scene=scene,
        preset=STICKMAN_TACTICAL_EXPLAINER_PRESET,
        available_references=available,
        budget=2,
    )

    # Exactly 2 references selected
    assert len(result.selected_references) == 2
    selected_entities = [s.entity_id for s in result.selected_references]

    # Primary entity (c1) selected in Tier 1
    assert selected_entities[0] == "c1"
    assert result.selected_references[0].priority_tier == 1

    # Secondary entity (o1 or e1) selected in Tier 2
    assert selected_entities[1] in ("o1", "e1")
    assert result.selected_references[1].priority_tier == 2

    # Third entity rejected with budget_exceeded
    assert len(result.rejected_references) >= 1
    budget_rejected = [r for r in result.rejected_references if r.reason == "budget_exceeded"]
    assert len(budget_rejected) >= 1


def test_reference_selection_style_benchmark_does_not_crowd_out_entities():
    """When 2 entity references fill budget, global style benchmark is NOT forced."""
    engine = ReferenceSelectionEngine(default_budget=2)

    available = [
        ReferenceMetadata(entity_id="c1", role="semantic", asset_path="refs/char.png"),
        ReferenceMetadata(entity_id="e1", role="semantic", asset_path="refs/env.png"),
    ]

    # STICKMAN_TACTICAL_EXPLAINER_PRESET has assets.style_benchmark defined
    scene = {"entity_bindings": {"character_id": "c1", "environment_id": "e1"}}

    result = engine.select_references(
        scene=scene,
        preset=STICKMAN_TACTICAL_EXPLAINER_PRESET,
        available_references=available,
        budget=2,
    )

    assert len(result.selected_references) == 2
    selected_entities = [s.entity_id for s in result.selected_references]
    assert "c1" in selected_entities
    assert "e1" in selected_entities
    assert "style_benchmark" not in selected_entities


def test_reference_selection_includes_style_benchmark_when_capacity_remains():
    """When only 1 entity reference exists, style benchmark fills remaining slot."""
    engine = ReferenceSelectionEngine(default_budget=2)

    available = [
        ReferenceMetadata(entity_id="c1", role="semantic", asset_path="refs/char.png"),
    ]

    scene = {"entity_bindings": {"character_id": "c1"}}

    result = engine.select_references(
        scene=scene,
        preset=STICKMAN_TACTICAL_EXPLAINER_PRESET,
        available_references=available,
        budget=2,
    )

    assert len(result.selected_references) == 2
    selected_entities = [s.entity_id for s in result.selected_references]
    assert selected_entities[0] == "c1"
    assert selected_entities[1] == "style_benchmark"
    assert result.selected_references[1].priority_tier == 3


def test_reference_selection_deduplication():
    """Duplicate asset paths are detected and rejected."""
    engine = ReferenceSelectionEngine(default_budget=2)

    available = [
        ReferenceMetadata(entity_id="c1", role="semantic", asset_path="refs/shared_asset.png"),
        ReferenceMetadata(entity_id="e1", role="semantic", asset_path="refs/shared_asset.png"),
    ]

    scene = {"entity_bindings": {"character_id": "c1", "environment_id": "e1"}}

    result = engine.select_references(
        scene=scene,
        preset=STICKMAN_TACTICAL_EXPLAINER_PRESET,
        available_references=available,
        budget=2,
    )

    assert len(result.selected_references) == 2
    selected_entities = [s.entity_id for s in result.selected_references]
    assert selected_entities[0] == "c1"
    assert selected_entities[1] == "style_benchmark"
    dupe_rejected = [r for r in result.rejected_references if r.reason == "duplicate_path"]
    assert len(dupe_rejected) == 1
    assert dupe_rejected[0].entity_id == "e1"


def test_reference_selection_preset_without_reference_support():
    """Presets with supports_reference_images=False (such as legacy_default) reject all references."""
    engine = ReferenceSelectionEngine(default_budget=2)

    available = [
        ReferenceMetadata(entity_id="c1", role="semantic", asset_path="refs/char.png"),
    ]

    scene = {"entity_bindings": {"character_id": "c1"}}

    result = engine.select_references(
        scene=scene,
        preset=LEGACY_DEFAULT_PRESET,
        available_references=available,
        budget=2,
    )

    assert len(result.selected_references) == 0
    assert len(result.rejected_references) == 1
    assert result.rejected_references[0].reason == "unsupported_by_preset"


def test_reference_selection_multiple_characters_in_one_scene():
    """Engine selects primary character in Tier 1 and secondary character in Tier 2, rejecting tertiary."""
    engine = ReferenceSelectionEngine(default_budget=2)

    available = [
        ReferenceMetadata(entity_id="c1", role="semantic", asset_path="refs/char1.png"),
        ReferenceMetadata(entity_id="c2", role="semantic", asset_path="refs/char2.png"),
        ReferenceMetadata(entity_id="c3", role="semantic", asset_path="refs/char3.png"),
    ]

    scene = {
        "entity_bindings": {
            "character_id": "c1",
            "secondary_character_ids": ["c2", "c3"],
        }
    }

    result = engine.select_references(
        scene=scene,
        preset=STICKMAN_TACTICAL_EXPLAINER_PRESET,
        available_references=available,
        budget=2,
    )

    assert len(result.selected_references) == 2
    assert result.selected_references[0].entity_id == "c1"
    assert result.selected_references[0].priority_tier == 1
    assert result.selected_references[1].entity_id == "c2"
    assert result.selected_references[1].priority_tier == 2

    # Tertiary character and style benchmark rejected with budget_exceeded
    rejected_entities = {r.entity_id: r.reason for r in result.rejected_references}
    assert rejected_entities.get("c3") == "budget_exceeded"
    assert rejected_entities.get("style_benchmark") == "budget_exceeded"


def test_reference_selection_primary_character_plus_machine():
    """Engine selects primary character in Tier 1 and machine in Tier 2."""
    engine = ReferenceSelectionEngine(default_budget=2)

    available = [
        ReferenceMetadata(entity_id="c1", role="semantic", asset_path="refs/char1.png"),
        ReferenceMetadata(entity_id="m1", role="semantic", asset_path="refs/machine1.png"),
    ]

    scene = {
        "entity_bindings": {
            "character_id": "c1",
            "machine_ids": ["m1"],
        }
    }

    result = engine.select_references(
        scene=scene,
        preset=STICKMAN_TACTICAL_EXPLAINER_PRESET,
        available_references=available,
        budget=2,
    )

    assert len(result.selected_references) == 2
    assert result.selected_references[0].entity_id == "c1"
    assert result.selected_references[0].priority_tier == 1
    assert result.selected_references[1].entity_id == "m1"
    assert result.selected_references[1].priority_tier == 2

    # Style benchmark rejected because budget was filled
    bm_rejected = [r for r in result.rejected_references if r.entity_id == "style_benchmark"]
    assert len(bm_rejected) == 1
    assert bm_rejected[0].reason == "budget_exceeded"


def test_reference_selection_primary_object_plus_machine_primary_object_wins():
    """Primary object takes Tier 1, and machine does not displace it; machine takes Tier 2."""
    engine = ReferenceSelectionEngine(default_budget=2)

    available = [
        ReferenceMetadata(entity_id="telescope_optics", role="semantic", asset_path="refs/telescope.png"),
        ReferenceMetadata(entity_id="centrifuge_machine", role="semantic", asset_path="refs/centrifuge.png"),
    ]

    scene = {
        "entity_bindings": {
            "object_ids": ["telescope_optics"],
            "machine_ids": ["centrifuge_machine"],
        }
    }

    result = engine.select_references(
        scene=scene,
        preset=STICKMAN_TACTICAL_EXPLAINER_PRESET,
        available_references=available,
        budget=2,
    )

    assert len(result.selected_references) == 2
    # Primary object wins slot 1 in Tier 1
    assert result.selected_references[0].entity_id == "telescope_optics"
    assert result.selected_references[0].priority_tier == 1
    assert "Primary object prop" in result.selected_references[0].selection_reason

    # Machine takes slot 2 in Tier 2
    assert result.selected_references[1].entity_id == "centrifuge_machine"
    assert result.selected_references[1].priority_tier == 2
    assert "Machine" in result.selected_references[1].selection_reason

    # Benchmark rejected with budget_exceeded
    bm_rejected = [r for r in result.rejected_references if r.entity_id == "style_benchmark"]
    assert len(bm_rejected) == 1
    assert bm_rejected[0].reason == "budget_exceeded"


def test_reference_selection_machine_only_scene_is_tier2():
    """When only a machine is bound (no character, object, or concept), it is assigned Tier 2."""
    engine = ReferenceSelectionEngine(default_budget=2)

    available = [
        ReferenceMetadata(entity_id="jet_engine", role="semantic", asset_path="refs/jet_engine.png"),
    ]

    scene = {
        "entity_bindings": {
            "machine_ids": ["jet_engine"],
        }
    }

    result = engine.select_references(
        scene=scene,
        preset=STICKMAN_TACTICAL_EXPLAINER_PRESET,
        available_references=available,
        budget=2,
    )

    assert len(result.selected_references) == 2
    assert result.selected_references[0].entity_id == "jet_engine"
    assert result.selected_references[0].priority_tier == 2
    assert "Machine" in result.selected_references[0].selection_reason

    # Style benchmark fills remaining slot in Tier 3
    assert result.selected_references[1].entity_id == "style_benchmark"
    assert result.selected_references[1].priority_tier == 3


def test_reference_selection_primary_character_plus_environment():
    """Engine selects primary character in Tier 1 and environment in Tier 2."""
    engine = ReferenceSelectionEngine(default_budget=2)

    available = [
        ReferenceMetadata(entity_id="c1", role="semantic", asset_path="refs/char1.png"),
        ReferenceMetadata(entity_id="env_cockpit", role="semantic", asset_path="refs/cockpit.png"),
    ]

    scene = {
        "entity_bindings": {
            "character_id": "c1",
            "environment_id": "env_cockpit",
        }
    }

    result = engine.select_references(
        scene=scene,
        preset=STICKMAN_TACTICAL_EXPLAINER_PRESET,
        available_references=available,
        budget=2,
    )

    assert len(result.selected_references) == 2
    assert result.selected_references[0].entity_id == "c1"
    assert result.selected_references[0].priority_tier == 1
    assert result.selected_references[1].entity_id == "env_cockpit"
    assert result.selected_references[1].priority_tier == 2

    # Benchmark rejected
    bm_rejected = [r for r in result.rejected_references if r.entity_id == "style_benchmark"]
    assert len(bm_rejected) == 1
    assert bm_rejected[0].reason == "budget_exceeded"


def test_reference_metadata_legacy_variations_and_rejections():
    """Verify all reference metadata variations and diagnostic rejections."""
    # 1. Legacy unannotated string path
    str_ref = ReferenceMetadata.normalize("legacy/cad_drawing.png", default_entity_id="part_x")
    assert str_ref.role is None
    assert str_ref.visual_style_id is None
    # Incompatible with stickman preset
    compat, reason = str_ref.check_compatibility("stickman_tactical_explainer", active_style_version=1)
    assert not compat
    assert "has no role metadata" in reason
    # Compatible with legacy_default preset
    compat_leg, reason_leg = str_ref.check_compatibility("legacy_default", active_style_version=1)
    assert compat_leg
    assert "Legacy unannotated reference accepted" in reason_leg

    # 2. Legacy unannotated dict
    dict_ref = ReferenceMetadata.normalize({"entity_id": "e_raw", "asset_path": "legacy/photo.png"})
    assert dict_ref.role is None
    compat, reason = dict_ref.check_compatibility("stylized_2d_narrative", active_style_version=1)
    assert not compat

    # 3. Explicit semantic metadata is compatible across styles
    sem_ref = ReferenceMetadata(entity_id="hero", role="semantic", asset_path="refs/hero.png")
    compat, _ = sem_ref.check_compatibility("stickman_tactical_explainer", active_style_version=1)
    assert compat
    compat, _ = sem_ref.check_compatibility("stylized_2d_narrative", active_style_version=1)
    assert compat

    # 4. Explicit style benchmark only compatible with own style
    style_ref = ReferenceMetadata(
        entity_id="stick_bm",
        role="style",
        asset_path="refs/stick.png",
        visual_style_id="stickman_tactical_explainer",
        style_version=1,
    )
    compat, _ = style_ref.check_compatibility("stickman_tactical_explainer", active_style_version=1)
    assert compat
    compat, reason = style_ref.check_compatibility("stylized_2d_narrative", active_style_version=1)
    assert not compat
    assert "designed for 'stickman_tactical_explainer'" in reason

    # 5. Explicit style_specific metadata requires matching style AND version
    spec_ref = ReferenceMetadata(
        entity_id="hero_v1",
        role="style_specific",
        asset_path="refs/hero_stick_v1.png",
        visual_style_id="stickman_tactical_explainer",
        style_version=1,
    )
    compat, _ = spec_ref.check_compatibility("stickman_tactical_explainer", active_style_version=1)
    assert compat
    compat_v2, reason_v2 = spec_ref.check_compatibility("stickman_tactical_explainer", active_style_version=2)
    assert not compat_v2
    assert "does not match active preset version" in reason_v2


def test_semantic_reference_safeguards_prompt_enforcement():
    """Verify semantic references communicate entity structure while target visual preset enforces rendering rules."""
    char = CharacterEntity(
        id="c1",
        name="Alex Rivera",
        role="Flight Specialist",
        visual_anchor="Slender pilot with braided hair, high cheekbones, pilot goggles on forehead",
        apparel_anchor="High-altitude olive flight suit with pressure straps",
        forbidden_mutations=["goggles", "hair braid"],
    )

    adapted = adapt_character(char, STICKMAN_TACTICAL_EXPLAINER_PRESET)

    # 1. Semantic invariants are preserved
    assert "Alex Rivera (Flight Specialist)" in adapted.adapted_description
    assert "Slender pilot with braided hair" in adapted.adapted_description
    assert "High-altitude olive flight suit" in adapted.adapted_description

    # 2. Visual preset rules are injected into prompt
    assert "2D stickman illustration style" in adapted.adapted_description
    assert "clean uniform black outlines" in adapted.adapted_description

    # 3. Negative anchors from preset prevent style bleed (e.g. realistic 3D, photorealism)
    neg_anchors = STICKMAN_TACTICAL_EXPLAINER_PRESET.negative_prompt_anchors
    assert any("photorealistic" in n.lower() or "3d" in n.lower() for n in neg_anchors)


# ==============================================================================
# 4. OpenAI Provider Reference Edits & Model Compatibility
# ==============================================================================

def test_openai_provider_text_only_generation_when_no_references():
    """When reference_images is None or empty, provider calls client.images.generate."""
    provider = OpenAIImageProvider(
        model="gpt-image-2",
        api_key="sk-test-key-fake",
    )

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_item = MagicMock()
    mock_item.b64_json = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    mock_item.url = None
    mock_resp.data = [mock_item]
    mock_resp.usage = MagicMock(input_tokens=10, output_tokens=20, total_tokens=30)
    mock_client.images.generate.return_value = mock_resp
    provider._client = mock_client

    result = provider.generate(prompt="A clean test prompt", size="1024x1536", quality="low")

    # Assert client.images.generate was called, and client.images.edit was NOT called
    assert mock_client.images.generate.called
    assert not mock_client.images.edit.called
    kwargs = mock_client.images.generate.call_args.kwargs
    assert kwargs["prompt"] == "A clean test prompt"
    assert kwargs["model"] == "gpt-image-2"
    assert kwargs["size"] == "1024x1536"
    assert kwargs["quality"] == "low"
    assert result.image_bytes is not None


def test_openai_provider_edit_request_omits_input_fidelity_for_gpt_image_2():
    """Reference-conditioned edit call uses client.images.edit and omits input_fidelity."""
    provider = OpenAIImageProvider(
        model="gpt-image-2",
        api_key="sk-test-key-fake",
    )

    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_item = MagicMock()
    mock_item.b64_json = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    mock_item.url = None
    mock_resp.data = [mock_item]
    mock_resp.usage = MagicMock(input_tokens=15, output_tokens=25, total_tokens=40)
    mock_client.images.edit.return_value = mock_resp
    provider._client = mock_client

    ref_1 = b"fake-png-bytes-1"
    ref_2 = b"fake-png-bytes-2"

    result = provider.generate(
        prompt="Stickman in cockpit",
        size="1024x1536",
        quality="low",
        reference_images=[ref_1, ref_2],
    )

    # Assert client.images.edit was called
    assert mock_client.images.edit.called
    assert not mock_client.images.generate.called

    kwargs = mock_client.images.edit.call_args.kwargs
    assert kwargs["prompt"] == "Stickman in cockpit"
    assert kwargs["model"] == "gpt-image-2"
    assert kwargs["size"] == "1024x1536"
    assert kwargs["quality"] == "low"

    # CRITICAL: Verify input_fidelity was NOT passed to gpt-image-2
    assert "input_fidelity" not in kwargs

    # Verify multiple images were passed as tuple sequence
    images_arg = kwargs["image"]
    assert isinstance(images_arg, list)
    assert len(images_arg) == 2
    assert images_arg[0][1] == ref_1
    assert images_arg[1][1] == ref_2


def test_openai_provider_fallback_disabled_by_default_raises_error():
    """When reference editing fails, fallback to text-only is disabled by default and raises error."""
    import openai

    provider = OpenAIImageProvider(
        model="gpt-image-2",
        api_key="sk-test-key-fake",
        allow_reference_fallback=False,  # default
    )

    mock_client = MagicMock()
    mock_client.images.edit.side_effect = openai.BadRequestError(
        message="Model does not support image edits",
        response=MagicMock(status_code=400, headers={}),
        body={"error": {"code": "model_not_supported", "message": "Model does not support image edits"}},
    )
    provider._client = mock_client

    with pytest.raises(ImageProviderError) as exc_info:
        provider.generate(
            prompt="Prompt that fails edit",
            size="1024x1536",
            quality="low",
            reference_images=[b"some-image-bytes"],
        )

    # Must NOT call text-only generate
    assert not mock_client.images.generate.called
    assert "OpenAI invalid image request" in str(exc_info.value)


def test_openai_provider_fallback_when_explicitly_configured_for_compatibility_error():
    """When explicitly configured, specifically recognized compatibility error falls back to generate."""
    import openai

    provider = OpenAIImageProvider(
        model="gpt-image-2",
        api_key="sk-test-key-fake",
        allow_reference_fallback=True,  # explicitly enabled
    )

    mock_client = MagicMock()
    # Edit fails with recognized code 'model_not_supported'
    mock_client.images.edit.side_effect = openai.BadRequestError(
        message="Model does not support image edits",
        response=MagicMock(status_code=400, headers={}),
        body={"error": {"code": "model_not_supported", "message": "Model does not support image edits"}},
    )

    # Generate succeeds
    mock_resp = MagicMock()
    mock_item = MagicMock()
    mock_item.b64_json = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
    mock_item.url = None
    mock_resp.data = [mock_item]
    mock_resp.usage = None
    mock_client.images.generate.return_value = mock_resp
    provider._client = mock_client

    result = provider.generate(
        prompt="Fallback prompt",
        size="1024x1536",
        quality="low",
        reference_images=[b"some-bytes"],
    )

    # Both were called
    assert mock_client.images.edit.called
    assert mock_client.images.generate.called
    assert result.image_bytes is not None


def test_openai_provider_fallback_does_not_mask_content_policy_or_other_errors():
    """Content policy violation or generic bad request does NOT trigger fallback even if fallback=True."""
    import openai

    provider = OpenAIImageProvider(
        model="gpt-image-2",
        api_key="sk-test-key-fake",
        allow_reference_fallback=True,
    )

    mock_client = MagicMock()
    mock_client.images.edit.side_effect = openai.BadRequestError(
        message="Your request was rejected due to content policy violation",
        response=MagicMock(status_code=400, headers={}),
        body={"error": {"code": "content_policy_violation", "message": "Safety system triggered"}},
    )
    provider._client = mock_client

    with pytest.raises(ImageContentBlockedError):
        provider.generate(
            prompt="Unsafe prompt",
            size="1024x1536",
            quality="low",
            reference_images=[b"some-bytes"],
        )

    # Generate MUST NOT be called
    assert not mock_client.images.generate.called


def test_openai_provider_fallback_does_not_trigger_on_rate_limit_or_timeout(monkeypatch):
    """Rate limit (429) or server error (500) never triggers fallback even when allow_reference_fallback=True."""
    import openai
    import time
    monkeypatch.setattr(time, "sleep", lambda s: None)

    provider = OpenAIImageProvider(
        model="gpt-image-2",
        api_key="sk-test-key-fake",
        allow_reference_fallback=True,
    )

    # 1. Rate limit error (429)
    mock_client = MagicMock()
    mock_client.images.edit.side_effect = openai.RateLimitError(
        message="Rate limit reached",
        response=MagicMock(status_code=429, headers={}),
        body={"error": {"message": "Rate limit reached"}},
    )
    provider._client = mock_client

    with pytest.raises(ImageRateLimitError) as exc_info:
        provider.generate(
            prompt="Rate limited prompt",
            size="1024x1536",
            quality="low",
            reference_images=[b"ref-bytes"],
        )
    assert not mock_client.images.generate.called
    assert "rate limit" in str(exc_info.value).lower()

    # 2. Server timeout / Internal error (500)
    mock_client.images.edit.side_effect = openai.InternalServerError(
        message="Internal server error",
        response=MagicMock(status_code=500, headers={}),
        body={"error": {"message": "Internal error"}},
    )
    with pytest.raises(ImageProviderError) as exc_info:
        provider.generate(
            prompt="Timeout prompt",
            size="1024x1536",
            quality="low",
            reference_images=[b"ref-bytes"],
        )
    assert not mock_client.images.generate.called


def test_base_image_provider_legacy_three_arg_subclass_success():
    """1. Legacy 3-argument BaseImageProvider subclass functions without reference_images."""
    class LegacyThreeArgProvider(BaseImageProvider):
        def __init__(self):
            super().__init__(model="legacy-model")
            self.call_count = 0
            self.called_prompt = None

        @property
        def provider_name(self) -> str:
            return "legacy_provider"

        def _execute_generate(
            self, prompt: str, size: str, quality: str
        ) -> tuple[bytes, int | None, int | None, int | None]:
            self.call_count += 1
            self.called_prompt = prompt
            return (b"legacy-render-output", 10, 20, 30)

    legacy_provider = LegacyThreeArgProvider()
    assert not legacy_provider.accepts_reference_images

    res = legacy_provider.generate(prompt="Legacy 3D prompt", size="1024x1536", quality="low")
    assert res.image_bytes == b"legacy-render-output"
    assert legacy_provider.called_prompt == "Legacy 3D prompt"
    assert legacy_provider.call_count == 1


def test_base_image_provider_new_subclass_with_reference_images_success():
    """2. New BaseImageProvider subclass accepting reference_images handles edits cleanly."""
    class ModernRefProvider(BaseImageProvider):
        def __init__(self):
            super().__init__(model="modern-ref-model")
            self.call_count = 0
            self.called_refs = None

        @property
        def provider_name(self) -> str:
            return "modern_ref_provider"

        def _execute_generate(
            self,
            prompt: str,
            size: str,
            quality: str,
            reference_images: list[bytes] | None = None,
        ) -> tuple[bytes, int | None, int | None, int | None]:
            self.call_count += 1
            self.called_refs = reference_images
            return (b"conditioned-output", 15, 25, 40)

    modern_provider = ModernRefProvider()
    assert modern_provider.accepts_reference_images

    ref_payload = [b"ref-image-1", b"ref-image-2"]
    res = modern_provider.generate(
        prompt="Stickman in hangar",
        size="1024x1536",
        quality="low",
        reference_images=ref_payload,
    )
    assert res.image_bytes == b"conditioned-output"
    assert modern_provider.called_refs == ref_payload
    assert modern_provider.call_count == 1


def test_base_image_provider_internal_type_error_not_retried():
    """3. New subclass raising an internal TypeError is NOT mistaken for legacy signature and NOT retried."""
    class FailingTypeErrorProvider(BaseImageProvider):
        def __init__(self):
            super().__init__(model="buggy-provider-model")
            self.call_count = 0

        @property
        def provider_name(self) -> str:
            return "buggy_provider"

        def _execute_generate(
            self,
            prompt: str,
            size: str,
            quality: str,
            reference_images: list[bytes] | None = None,
        ) -> tuple[bytes, int | None, int | None, int | None]:
            self.call_count += 1
            # Simulate internal bug inside provider implementation
            raise TypeError("Internal dict type mismatch inside provider SDK payload formatting")

    failing_provider = FailingTypeErrorProvider()
    assert failing_provider.accepts_reference_images

    # Must raise the exact internal TypeError, not swallow or retry
    with pytest.raises(TypeError, match="Internal dict type mismatch inside provider SDK"):
        failing_provider.generate(
            prompt="Prompt triggering bug",
            size="1024x1536",
            quality="low",
        )

    # CRITICAL: Verify provider was called exactly ONCE and never retried
    assert failing_provider.call_count == 1


def test_base_image_provider_unsupported_legacy_provider_rejects_reference_images():
    """4. Reference images passed to an unsupported legacy provider raise clear ImageProviderError without executing."""
    class LegacyProviderNoRefs(BaseImageProvider):
        def __init__(self):
            super().__init__(model="strictly-text-model")
            self.call_count = 0

        @property
        def provider_name(self) -> str:
            return "legacy_text_only"

        def _execute_generate(
            self, prompt: str, size: str, quality: str
        ) -> tuple[bytes, int | None, int | None, int | None]:
            self.call_count += 1
            return (b"text-only-output", 10, 20, 30)

    legacy_provider = LegacyProviderNoRefs()
    assert not legacy_provider.accepts_reference_images

    with pytest.raises(ImageProviderError) as exc_info:
        legacy_provider.generate(
            prompt="Legacy prompt with refs",
            size="1024x1536",
            quality="low",
            reference_images=[b"fake-ref-bytes"],
        )

    assert "does not support reference conditioning" in str(exc_info.value)
    # The unsupported provider was rejected immediately; _execute_generate was never called
    assert legacy_provider.call_count == 0


# ==============================================================================
# 5. Pipeline Integration & Telemetry
# ==============================================================================

def test_pipeline_images_step_records_reference_diagnostics(tmp_path, monkeypatch):
    """Pipeline images step records reference paths and structured selection diagnostics."""
    from app.config import settings
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    # Create dummy reference file on disk
    ref_file = tmp_path / "hero_silhouette.png"
    ref_file.write_bytes(b"dummy-png-silhouette")

    fake_video = Video(
        id=uuid.uuid4(),
        topic="Aviation Aerodynamics",
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )

    prev_data = {
        "storyboard": {
            "storyboard": {
                "visual_style": STICKMAN_TACTICAL_EXPLAINER_PRESET.display_name,
                "scenes": [
                    {
                        "index": 1,
                        "narration": "Air flowing over an aircraft wing.",
                        "final_prompt": "Clean 2D stickman pilot inspecting the wing.",
                        "entity_bindings": {"character_id": "c1"},
                    }
                ],
            }
        },
        "video_bible": {
            "reference_assets": [
                {
                    "entity_id": "c1",
                    "role": "semantic",
                    "asset_path": str(ref_file),
                }
            ]
        },
    }

    mock_provider = MockImageProvider(model="mock-gpt-image-2")
    step = VideoStep(name="images", output=None, position=3)

    output = generate_images_step(
        video=fake_video,
        prev=prev_data,
        step=step,
        image_provider=mock_provider,
    )

    items = output["images"]["items"]
    assert len(items) == 1
    item = items[0]

    assert item["status"] == "ok"
    assert "reference_images" in item
    assert str(ref_file) in item["reference_images"]
    assert "reference_selection" in item
    diag = item["reference_selection"]
    assert diag is not None
    assert diag["budget"] == 2
    assert diag["active_style_id"] == "stickman_tactical_explainer"
    assert len(diag["selected_references"]) == 1
    assert diag["selected_references"][0]["entity_id"] == "c1"

    # Confirm mock provider received the reference bytes
    assert mock_provider.last_call_type == "edit"
    assert mock_provider.last_reference_images is not None
    assert len(mock_provider.last_reference_images) == 1
    assert mock_provider.last_reference_images[0] == b"dummy-png-silhouette"
