"""Deterministic style-aware entity adaptation service."""

from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field

from .schemas import VisualStylePreset


class AdaptedEntity(BaseModel):
    """An entity whose prompt description has been compiled according to visual style drawing rules."""

    entity_id: str
    entity_type: str = Field(description="'character', 'object', 'environment', or 'concept'")
    original_data: dict[str, Any] = Field(description="Original untouched entity fields")
    adapted_description: str = Field(description="Style-aware compiled prompt segment")
    applied_style_id: str = Field(description="Preset ID used for adaptation")
    applied_rules: list[str] = Field(default_factory=list, description="Drawing rule categories applied")


def _extract_field(obj: Any, field_name: str, default: Any = "") -> Any:
    """Helper to read an attribute or dict key safely."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(field_name, default)
    return getattr(obj, field_name, default)


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert an object or Pydantic model to a dictionary safely."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    if isinstance(obj, dict):
        return dict(obj)
    return {}


def adapt_character(character: Any, preset: VisualStylePreset) -> AdaptedEntity:
    """Compile a style-aware character description while preserving source identity.

    Applies the preset's character_anatomy, linework_and_outlines, and color_and_shading rules.
    """
    orig = _to_dict(character)
    char_id = str(_extract_field(character, "id", "character"))
    name = str(_extract_field(character, "name", "")).strip()
    role = str(_extract_field(character, "role", "")).strip()
    visual_anchor = str(_extract_field(character, "visual_anchor", "")).strip()
    apparel_anchor = str(_extract_field(character, "apparel_anchor", "")).strip()

    parts: list[str] = []
    subject_label = f"{name} ({role})" if role and role.lower() not in name.lower() else name
    if not subject_label:
        subject_label = "Character"

    # Style-specific grammar based on preset id and drawing rules
    rules = preset.rules
    if preset.id == "stickman_tactical_explainer":
        # 2D stickman rules
        desc = (
            f"Rendered in 2D stickman illustration style: {subject_label}, {visual_anchor}. "
            f"Wearing {apparel_anchor}. Drawn with clean uniform black outlines, simplified stick figure anatomy, "
            f"and flat color fills without realistic facial features."
        )
    elif preset.id == "stylized_2d_narrative":
        # Stylized 2D cartoon / anime-proportioned
        desc = (
            f"Rendered in stylized 2D graphic illustration style: {subject_label}, {visual_anchor}. "
            f"Wearing {apparel_anchor}. Drawn with expressive ink linework, 2-tone cel-shading, and clean graphic novel contours."
        )
    elif preset.id == "legacy_default":
        # Smooth 3D matte render (baseline)
        desc = (
            f"Clean 3D stylized matte render: {subject_label}, {visual_anchor}. "
            f"Wearing {apparel_anchor}. Smooth rounded contours, matte materials, soft studio lighting."
        )
    else:
        # Dynamic fallback for generic presets using their drawing rules
        desc = (
            f"{preset.display_name} style: {subject_label}, {visual_anchor}. Wearing {apparel_anchor}. "
            f"Anatomy: {rules.character_anatomy}. Linework: {rules.linework_and_outlines}. Shading: {rules.color_and_shading}."
        )

    return AdaptedEntity(
        entity_id=char_id,
        entity_type="character",
        original_data=orig,
        adapted_description=desc.strip(),
        applied_style_id=preset.id,
        applied_rules=["character_anatomy", "linework_and_outlines", "color_and_shading"],
    )


def adapt_object(obj: Any, preset: VisualStylePreset) -> AdaptedEntity:
    """Compile a style-aware object description while preserving functional and visual properties."""
    orig = _to_dict(obj)
    obj_id = str(_extract_field(obj, "id", "object"))
    name = str(_extract_field(obj, "name", "")).strip()
    visual_desc = str(_extract_field(obj, "visual_description", "")).strip()
    material = str(_extract_field(obj, "material_and_finish", "")).strip()

    prop_label = name or "Object"
    rules = preset.rules

    if preset.id == "stickman_tactical_explainer":
        desc = (
            f"Rendered as a tactical 2D line-art prop: {prop_label} ({visual_desc}). "
            f"Clean technical outlines, flat color fills, and minimalist schematic finish."
        )
    elif preset.id == "stylized_2d_narrative":
        desc = (
            f"Rendered in stylized 2D graphic novel illustration: {prop_label} ({visual_desc}). "
            f"Finished with crisp line art, vibrant cel-shaded color accents, and subtle ink weight."
        )
    elif preset.id == "legacy_default":
        material_str = f", {material}" if material else ""
        desc = f"Clean 3D matte object: {prop_label} ({visual_desc}){material_str}."
    else:
        desc = (
            f"{preset.display_name} object rendering: {prop_label} ({visual_desc}). "
            f"Prop rendering: {rules.objects_and_machines}."
        )

    return AdaptedEntity(
        entity_id=obj_id,
        entity_type="object",
        original_data=orig,
        adapted_description=desc.strip(),
        applied_style_id=preset.id,
        applied_rules=["objects_and_machines", "linework_and_outlines"],
    )


def adapt_environment(
    env: Any,
    preset: VisualStylePreset,
    relevant_props: list[str] | None = None,
    include_spatial_layout: bool = True,
    local_setting: str | None = None,
    lighting_override: str | None = None,
) -> AdaptedEntity:
    """Compile a style-aware environment description while preserving spatial boundaries and props.

    Allows beat-aware environment slicing by optionally filtering props, overriding
    the local setting, or suppressing wide-angle spatial layouts for close-ups and interiors.
    Prevents conflicting environment names, outdoor lighting, and landscape layouts from contaminating
    interior or localized beat compositions.
    """
    orig = _to_dict(env)
    env_id = str(_extract_field(env, "id", "environment"))
    name = str(_extract_field(env, "name", "")).strip()
    spatial_layout = str(_extract_field(env, "spatial_layout", "")).strip()
    lighting_setup = str(_extract_field(env, "lighting_setup", "")).strip()
    time_of_day = str(_extract_field(env, "time_of_day", "")).strip()

    props = relevant_props if relevant_props is not None else (_extract_field(env, "persistent_props", []) or [])
    props_str = f", persistent props: {', '.join(str(p) for p in props)}" if props else ""
    time_str = f", {time_of_day}" if time_of_day and time_of_day != "unspecified" else ""

    # Determine effective name and layout content
    effective_name = name
    if local_setting and local_setting.strip():
        effective_name = local_setting.strip()

    effective_layout = spatial_layout if (include_spatial_layout and spatial_layout) else ""
    if effective_layout:
        layout_str = f" ({effective_layout}{props_str})"
    elif props_str:
        layout_str = f" ({props_str.lstrip(', ')})"
    else:
        layout_str = ""

    # Determine effective lighting setup
    if lighting_override and lighting_override.strip():
        effective_lighting = lighting_override.strip()
    elif local_setting and any(term in local_setting.lower() for term in ("interior", "inside", "room", "storeroom", "cellar", "cabin", "hall", "granary", "pantry", "workshop")):
        # Prevent outdoor daylight from contaminating interior settings
        if any(term in lighting_setup.lower() for term in ("daylight", "sunlight", "hazy sun", "open sky", "sunlit")):
            effective_lighting = "Dim interior ambient lighting with soft atmospheric shadows"
        else:
            effective_lighting = lighting_setup
    else:
        effective_lighting = lighting_setup

    rules = preset.rules

    if preset.id == "stickman_tactical_explainer":
        desc = (
            f"Environment in 1-point perspective architectural line art: {effective_name}{layout_str}. "
            f"Clean minimalist background linework with high readability, neutral ambient lighting{time_str}."
        )
    elif preset.id == "stylized_2d_narrative":
        desc = (
            f"Environment in stylized 2D narrative background art: {effective_name}{layout_str}. "
            f"Moody atmospheric lighting ({effective_lighting}{time_str}), warm/cool color grading with expressive linework."
        )
    elif preset.id == "legacy_default":
        lighting_str = f", {effective_lighting}" if effective_lighting else ""
        desc = f"Environment: {effective_name}{layout_str}{lighting_str}{time_str}."
    else:
        desc = (
            f"{preset.display_name} environment: {effective_name}{layout_str}. "
            f"Perspective: {rules.environment_and_perspective}. Lighting: {effective_lighting or rules.lighting_style}{time_str}."
        )

    return AdaptedEntity(
        entity_id=env_id,
        entity_type="environment",
        original_data=orig,
        adapted_description=desc.strip(),
        applied_style_id=preset.id,
        applied_rules=["environment_and_perspective", "lighting_style"],
    )


def adapt_concept(
    concept: Any,
    preset: VisualStylePreset,
    relevant_elements: list[str] | None = None,
    focused_explanation: str | None = None,
    focused_analogy: str | None = None,
) -> AdaptedEntity:
    """Compile a style-aware abstract concept or diagram description.

    Allows beat-specific concept slicing by optionally passing `relevant_elements`,
    `focused_explanation`, or `focused_analogy` to prevent multi-stage infographics
    from cluttering individual beats.
    """
    orig = _to_dict(concept)
    concept_id = str(_extract_field(concept, "id", "concept"))
    name = str(_extract_field(concept, "name", "")).strip()

    if focused_explanation is not None and focused_explanation.strip():
        explained = focused_explanation.strip()
    else:
        explained = str(_extract_field(concept, "concept_explained", "")).strip()

    if focused_analogy is not None and focused_analogy.strip():
        analogy = focused_analogy.strip()
    else:
        analogy = str(_extract_field(concept, "visual_analogy", "")).strip()

    if relevant_elements is not None:
        elements = relevant_elements
    else:
        elements = _extract_field(concept, "graphical_elements", []) or []

    elem_str = f", graphical elements: {', '.join(str(e) for e in elements)}" if elements else ""

    rules = preset.rules

    if preset.id == "stickman_tactical_explainer":
        desc = (
            f"Tactical schematic visual diagram: {name} illustrating {explained}. "
            f"Visual analogy: {analogy}{elem_str}. Simplified cross-section with clean vector arrows and technical callouts."
        )
    elif preset.id == "stylized_2d_narrative":
        desc = (
            f"Illustrated conceptual diagram: {name} illustrating {explained}. "
            f"Visual analogy: {analogy}{elem_str}. Narrative visual metaphors with clean cel-shaded graphical flows."
        )
    elif preset.id == "legacy_default":
        desc = f"Concept diagram: {name} illustrating {explained}. Visual analogy: {analogy}{elem_str}."
    else:
        desc = (
            f"{preset.display_name} concept visualization: {name} illustrating {explained}. "
            f"Analogy: {analogy}. Abstract rendering rules: {rules.abstract_concepts}."
        )

    return AdaptedEntity(
        entity_id=concept_id,
        entity_type="concept",
        original_data=orig,
        adapted_description=desc.strip(),
        applied_style_id=preset.id,
        applied_rules=["abstract_concepts"],
    )
