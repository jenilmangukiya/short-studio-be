"""Pydantic schemas for the Visual Style System."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

from app.schemas import ArtDirection

ReferenceRole = Literal["style", "semantic", "style_specific"]


class ReferenceMetadata(BaseModel):
    """Metadata tracking asset role, entity association, and style compatibility."""

    entity_id: str = Field(description="Unique entity identifier or 'global_style'")
    role: ReferenceRole | None = Field(
        default=None,
        description=(
            "'style' (defines medium/palette), 'semantic' (structure/identity without medium), "
            "'style_specific' (asset rendered in a specific visual preset), or None for unannotated legacy assets"
        ),
    )
    asset_path: str = Field(description="Relative or absolute path to reference asset on disk")
    visual_style_id: str | None = Field(
        default=None,
        description="Preset ID this asset was designed for (required if role is 'style_specific' or 'style')",
    )
    style_version: int | None = Field(
        default=None,
        description="Version of the preset this asset was created with",
    )
    description: str | None = Field(
        default=None,
        description="Optional human-readable description of the reference asset",
    )
    source_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Original untouched metadata from source import or entity catalog",
    )

    def is_compatible_with(self, active_style_id: str, active_style_version: int = 1) -> bool:
        """Evaluate whether this asset is safe to supply as an image reference."""
        compat, _ = self.check_compatibility(active_style_id, active_style_version)
        return compat

    def check_compatibility(self, active_style_id: str, active_style_version: int = 1) -> tuple[bool, str]:
        """Check compatibility and return (is_compatible, diagnostic_reason)."""
        # 1. Unannotated legacy reference (no role metadata)
        if self.role is None:
            # Under legacy_default (the exact original Clean 3D studio), unannotated legacy assets are accepted
            if active_style_id == "legacy_default":
                return True, "Legacy unannotated reference accepted under legacy_default style."
            # Under specialized presets, do not assume photos or CAD images are style-neutral
            return (
                False,
                f"Legacy reference '{self.asset_path}' has no role metadata. Cannot safely determine "
                f"compatibility with specialized preset '{active_style_id}'. Explicit 'semantic' or "
                f"'style_specific' metadata is required."
            )

        # 2. Semantic references communicate physical structure; compatible across all visual styles
        if self.role == "semantic":
            return True, "Semantic reference is compatible across all visual styles."

        # 3. Style references define the illustration medium; only valid for their own preset
        if self.role == "style":
            if self.visual_style_id == active_style_id:
                return True, f"Style benchmark matches active preset '{active_style_id}'."
            return (
                False,
                f"Style reference designed for '{self.visual_style_id}' is incompatible with active preset '{active_style_id}'."
            )

        # 4. Style-specific references (e.g. 2D turnaround, 3D model) must match preset ID and version
        if self.role == "style_specific":
            if not self.visual_style_id:
                return False, "Style-specific reference is missing required visual_style_id."
            if self.visual_style_id != active_style_id:
                return (
                    False,
                    f"Style-specific reference designed for '{self.visual_style_id}' is incompatible with active preset '{active_style_id}'."
                )
            if self.style_version is not None and self.style_version != active_style_version:
                return (
                    False,
                    f"Style-specific reference version {self.style_version} does not match active preset version {active_style_version}."
                )
            return True, f"Style-specific reference matches active preset '{active_style_id}' (v{active_style_version})."

        return False, f"Unknown reference role '{self.role}'."

    @classmethod
    def normalize(cls, raw: Any, default_entity_id: str = "unknown") -> "ReferenceMetadata":
        """Normalize legacy strings or unannotated dictionaries into ReferenceMetadata.

        Preserves existing reference behavior without inventing ungrounded style IDs or assuming
        arbitrary images are style-neutral.
        """
        if isinstance(raw, cls):
            return raw
        if isinstance(raw, str):
            return cls(
                entity_id=default_entity_id,
                role=None,
                asset_path=raw,
                visual_style_id=None,
                style_version=None,
                source_metadata={"raw_path": raw},
            )
        if isinstance(raw, dict):
            if "role" in raw and raw["role"] in ("style", "semantic", "style_specific"):
                return cls(**raw)
            # Legacy dictionary without role: do not invent role, style ID, or version
            path = raw.get("asset_path") or raw.get("path") or raw.get("file") or ""
            return cls(
                entity_id=raw.get("entity_id") or default_entity_id,
                role=None,
                asset_path=path,
                visual_style_id=raw.get("visual_style_id"),
                style_version=raw.get("style_version"),
                description=raw.get("description"),
                source_metadata=dict(raw),
            )
        raise ValueError(f"Cannot normalize reference of type {type(raw).__name__}")


class SelectedReference(BaseModel):
    """A reference asset selected for generation with priority and reasoning."""

    entity_id: str
    role: ReferenceRole | str = "semantic"
    asset_path: str
    priority_tier: int = Field(description="1: primary entity, 2: secondary entity, 3: style benchmark")
    selection_reason: str


class RejectedReference(BaseModel):
    """A reference asset evaluated and excluded by the selection engine."""

    entity_id: str
    asset_path: str
    role: str
    reason: Literal[
        "incompatible_style",
        "budget_exceeded",
        "duplicate_path",
        "unsupported_by_preset",
        "file_not_found",
    ]
    details: str


class ReferenceSelectionResult(BaseModel):
    """Structured diagnostic result returned by the Reference Selection Engine."""

    selected_references: list[SelectedReference] = Field(default_factory=list)
    rejected_references: list[RejectedReference] = Field(default_factory=list)
    budget: int = 2
    active_style_id: str
    summary: str


class StyleDrawingRules(BaseModel):
    """Illustration rules defining the visual medium and rendering grammar."""

    character_anatomy: str = Field(
        description="Head-to-body proportions, silhouette rules, facial feature conventions."
    )
    linework_and_outlines: str = Field(
        description="Stroke thickness, ink texture (e.g. 3px clean vector, rough charcoal, no outlines)."
    )
    color_and_shading: str = Field(
        description="Shading technique (e.g. flat fills, 2-tone cel-shading, soft ambient gradients)."
    )
    environment_and_perspective: str = Field(
        description="Background drawing rules (e.g. 1-point perspective, 2D architectural ink, flat backdrops)."
    )
    objects_and_machines: str = Field(
        description="How mechanical, technical, and everyday props are rendered in this illustration style."
    )
    abstract_concepts: str = Field(
        description="How diagrams, flows, and scientific concepts are simplified visually."
    )
    lighting_style: str = Field(
        description="Lighting philosophy (e.g. directional ambient cel-light, soft studio ambient)."
    )


class StyleAssetBindings(BaseModel):
    """Application-managed reference assets illustrating this style's visual rules."""

    style_benchmark: str | None = Field(
        default=None,
        description="Relative path to master benchmark demonstrating linework, materials, and lighting.",
    )
    anatomy_reference: str | None = Field(
        default=None,
        description="Relative path to anatomy turnaround (front/side/proportions) for characters.",
    )
    environment_reference: str | None = Field(
        default=None,
        description="Relative path to benchmark environment showing perspective and line art.",
    )


class StyleCapabilities(BaseModel):
    """Provider capabilities and reference constraints for this preset."""

    supports_reference_images: bool = Field(
        default=True,
        description="Whether this style benefits from reference-conditioned image generation.",
    )
    max_reference_budget: int = Field(
        default=2,
        description="Initial application-level reference image limit for this style.",
    )
    preferred_fidelity: Literal["high", "low"] | None = Field(
        default=None,
        description="Optional fidelity preference; kept None for models like gpt-image-2.",
    )


class StyleRenderProfile(BaseModel):
    """Optional renderer camera-motion profile."""

    motion_scale: float = Field(
        default=1.0,
        description="Multiplier for Ken Burns zoom/pan delta (e.g. 0.6 for subtle stickman motion).",
    )
    default_transition: str = Field(
        default="cut",
        description="Default cut or transition between scenes in this visual style.",
    )


class VisualStylePreset(BaseModel):
    """The canonical developer-managed visual style preset."""

    id: str = Field(description="Stable unique ID, e.g. 'stickman_tactical_explainer'.")
    display_name: str = Field(description="UI label displayed to users.")
    category: Literal["illustrated_2d", "3d_render", "photorealistic", "hybrid"] = "illustrated_2d"
    version: int = Field(default=1, description="Monotonic version number.")
    description: str = Field(description="Explanatory summary for frontend tooltips/cards.")
    preview_image_path: str = Field(description="Path to static preview thumbnail for UI.")

    rules: StyleDrawingRules
    permanent_prompt_anchors: list[str] = Field(
        description="Tokens injected into prompt composition to establish medium and finish."
    )
    negative_prompt_anchors: list[str] = Field(
        default_factory=list,
        description="Negative tokens forbidding incompatible rendering techniques.",
    )
    default_palette_hex: list[str] = Field(
        default_factory=list,
        description="Representative color palette (used for UI swatches and art direction).",
    )
    art_direction: ArtDirection = Field(
        description="Pre-compiled ArtDirection object populated into VideoBible."
    )
    assets: StyleAssetBindings = Field(default_factory=StyleAssetBindings)
    capabilities: StyleCapabilities = Field(default_factory=StyleCapabilities)
    render_profile: StyleRenderProfile = Field(default_factory=StyleRenderProfile)
