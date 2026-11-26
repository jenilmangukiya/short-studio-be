"""Hardcoded visual style registry for Shorts Studio."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from app.schemas import ArtDirection
from .schemas import (
    StyleAssetBindings,
    StyleCapabilities,
    StyleDrawingRules,
    StyleRenderProfile,
    VisualStylePreset,
)

DEFAULT_STYLE_ID = "legacy_default"

# ==============================================================================
# Preset 1: Legacy Default (Clean 3D Studio - Exact Codebase Baseline)
# ==============================================================================
LEGACY_DEFAULT_PRESET = VisualStylePreset(
    id="legacy_default",
    display_name="Clean 3D Studio (Default)",
    category="3d_render",
    version=1,
    description="Smooth stylized 3D matte characters and objects with soft studio lighting and clean pastel environments.",
    preview_image_path="previews/legacy_default.jpg",
    rules=StyleDrawingRules(
        character_anatomy="Smooth stylized 3D human figures with rounded features and matte skin textures.",
        linework_and_outlines="No line art or ink strokes; clean geometric edges with soft bevels.",
        color_and_shading="Subtle ambient occlusion, soft radiosity, smooth matte materials with diffuse reflection.",
        environment_and_perspective="Clean studio sets with soft pastel backdrops or minimalist 3D isometric rooms.",
        objects_and_machines="Smooth matte toy-like or industrial design 3D models with clean silhouettes.",
        abstract_concepts="Volumetric 3D glowing shapes, floating semi-translucent geometric diagrams.",
        lighting_style="Three-point soft studio lighting with warm key and cool pastel fill.",
    ),
    permanent_prompt_anchors=[
        "Clean 3D render",
        "soft studio lighting",
        "smooth matte materials",
        "simple pastel gradient background",
        "shallow depth of field",
    ],
    negative_prompt_anchors=[
        "harsh shadows",
        "sketchy line art",
        "photorealistic human skin pores",
        "2D flat vector",
    ],
    default_palette_hex=[
        "#FFFFFF",
        "#E0E0E0",
        "#B0BEC5",
        "#90CAF9",
        "#FFE082",
    ],
    art_direction=ArtDirection(
        style_family="cinematic",
        visual_genre="styled_3d",
        rendering_texture="smooth matte materials",
        lighting_philosophy="soft studio lighting",
        negative_anchors=[
            "no harsh shadows",
            "no sketchy line art",
            "no photorealistic human skin pores",
            "no 2D flat vector",
        ],
        primary_palette_hex=[
            "#FFFFFF",
            "#E0E0E0",
            "#B0BEC5",
            "#90CAF9",
            "#FFE082",
        ],
    ),
    assets=StyleAssetBindings(
        style_benchmark=None,
        anatomy_reference=None,
        environment_reference=None,
    ),
    capabilities=StyleCapabilities(
        supports_reference_images=False,
        max_reference_budget=0,
        preferred_fidelity=None,
    ),
    render_profile=StyleRenderProfile(
        motion_scale=1.0,
        default_transition="cut",
    ),
)


# ==============================================================================
# Preset 2: Stickman Tactical Explainer
# ==============================================================================
STICKMAN_TACTICAL_EXPLAINER_PRESET = VisualStylePreset(
    id="stickman_tactical_explainer",
    display_name="Stickman Tactical Explainer",
    category="illustrated_2d",
    version=1,
    description="Minimalist white circular-head stick figures inside detailed 2D architectural line-art environments with tactical survival diagrams.",
    preview_image_path="previews/stickman_tactical.jpg",
    rules=StyleDrawingRules(
        character_anatomy=(
            "Characters have pure white circular heads with uniform 3px black outlines. 1:3.5 head-to-body ratio. "
            "Bodies and limbs are slender single-stroke solid black stick lines. Hands are simplified white circular mittens. "
            "Eyes are minimalist solid black dots or ovals; eyebrows and mouths are clean black expressive strokes. "
            "No realistic noses, lips, or ears."
        ),
        linework_and_outlines=(
            "Uniform solid black ink contours (equivalent to 3px vector stroke) maintaining equal weight across "
            "characters, props, and architectural elements."
        ),
        color_and_shading=(
            "Flat solid color fills for clothing, furniture, and surfaces, accented by subtle single-tone cel-shading "
            "and ambient gradients to establish depth."
        ),
        environment_and_perspective=(
            "Detailed architectural environments drawn in matching 2D ink line art. Strict 1-point perspective for "
            "corridors, hallways, and shafts; 2-point perspective for rooms; top-down for maps and stairwells. "
            "No photorealistic backdrops."
        ),
        objects_and_machines=(
            "Technical equipment, furniture, and tools are drawn as clean line-and-fill diagrams with clear "
            "functional silhouettes and schematic details."
        ),
        abstract_concepts=(
            "Diagrammatic overlays, cross-sections, and flowcharts with minimalist clean iconography, bold sans-serif "
            "labels, and high-contrast color highlights."
        ),
        lighting_style=(
            "Directional ambient lighting from identifiable light sources casting soft flat shadow zones across "
            "adjacent surfaces and architectural planes."
        ),
    ),
    permanent_prompt_anchors=[
        "clean 2D animated stickman illustration style",
        "uniform solid black ink outlines",
        "flat color fills with subtle ambient cel-shading",
        "minimalist modern graphic novel aesthetic",
        "crisp vector finish",
    ],
    negative_prompt_anchors=[
        "photorealistic",
        "3D CGI",
        "realistic human face with skin texture",
        "detailed hands with fingernails",
        "anime hair spikes",
        "watercolor bleed",
        "rough sketchy pencil lines",
    ],
    default_palette_hex=[
        "#FFFFFF",
        "#1A1A1A",
        "#F4A92E",
        "#9EA6AC",
        "#D9D2C9",
        "#151C28",
        "#38A3A5",
    ],
    art_direction=ArtDirection(
        style_family="illustrated",
        visual_genre="stickman_illustrated",
        rendering_texture="2D vector ink contours with flat color fills and ambient cel-shading",
        lighting_philosophy="directional ambient lighting with soft flat shadow zones",
        negative_anchors=[
            "no photorealistic",
            "no 3D CGI",
            "no realistic human face with skin texture",
            "no detailed hands with fingernails",
            "no anime hair spikes",
            "no watercolor bleed",
            "no rough sketchy pencil lines",
        ],
        primary_palette_hex=[
            "#FFFFFF",
            "#1A1A1A",
            "#F4A92E",
            "#9EA6AC",
            "#D9D2C9",
            "#151C28",
            "#38A3A5",
        ],
    ),
    assets=StyleAssetBindings(
        style_benchmark="references/stickman_tactical/style_benchmark.png",
        anatomy_reference="references/stickman_tactical/anatomy_reference.png",
        environment_reference="references/stickman_tactical/environment_reference.png",
    ),
    capabilities=StyleCapabilities(
        supports_reference_images=True,
        max_reference_budget=2,
        preferred_fidelity=None,
    ),
    render_profile=StyleRenderProfile(
        motion_scale=0.6,
        default_transition="cut",
    ),
)


# ==============================================================================
# Preset 3: Stylized 2D Narrative Drama
# ==============================================================================
STYLIZED_2D_NARRATIVE_PRESET = VisualStylePreset(
    id="stylized_2d_narrative",
    display_name="Stylized 2D Narrative Drama",
    category="illustrated_2d",
    version=1,
    description="Expressive chibi-stick cartoon figures with stylized clothing, hair volume, and anime-inspired emotion in cinematic mood-lit environments.",
    preview_image_path="previews/stylized_2d.jpg",
    rules=StyleDrawingRules(
        character_anatomy=(
            "Expressive 2D stylized cartoon figures with white rounded heads and sculpted volume hair. Slender limbs "
            "with distinct colored clothing silhouettes matching story context. Large expressive vertical cartoon "
            "eyes with white specular catchlights; dynamic mouths and blush indicators."
        ),
        linework_and_outlines=(
            "Clean tapered black contour lines with primary silhouettes in medium-weight strokes and interior details "
            "in finer lines."
        ),
        color_and_shading=(
            "Two-tone digital cel-shading with clear highlight and core shadow zones; rich ambient bounce lighting."
        ),
        environment_and_perspective=(
            "Mood-lit interior and exterior backgrounds with cinematic framing and dynamic perspective angles. "
            "Atmospheric fog, smoke particles, and subtle localized light accents."
        ),
        objects_and_machines=(
            "Props and equipment rendered with bold colored silhouettes, subtle edge bevel highlights, and "
            "emissive glow accents where appropriate."
        ),
        abstract_concepts=(
            "Cinematic visual metaphors and illustrative graphic overlays integrated cohesively into the illustrated scene."
        ),
        lighting_style=(
            "Atmospheric dramatic lighting: strong warm and cool color temperature contrasts with motivated directional "
            "light sources and deep tonal shadows."
        ),
    ),
    permanent_prompt_anchors=[
        "stylized 2D digital cartoon animation art",
        "clean bold outlines",
        "two-tone cel-shading",
        "expressive graphic novel character aesthetic",
    ],
    negative_prompt_anchors=[
        "photorealism",
        "3D volumetric render",
        "blurry watercolor",
        "inconsistent line weights",
        "unshaded flat vector clip art",
    ],
    default_palette_hex=[
        "#FFFFFF",
        "#212529",
        "#5C3A21",
        "#E91E63",
        "#FF8DA1",
        "#FFE082",
        "#FF5722",
    ],
    art_direction=ArtDirection(
        style_family="illustrated",
        visual_genre="2d_cartoon",
        rendering_texture="clean tapered black contour lines with two-tone digital cel-shading",
        lighting_philosophy="atmospheric dramatic lighting with strong warm and cool temperature contrasts",
        negative_anchors=[
            "no photorealism",
            "no 3D volumetric render",
            "no blurry watercolor",
            "no inconsistent line weights",
            "no unshaded flat vector clip art",
        ],
        primary_palette_hex=[
            "#FFFFFF",
            "#212529",
            "#5C3A21",
            "#E91E63",
            "#FF8DA1",
            "#FFE082",
            "#FF5722",
        ],
    ),
    assets=StyleAssetBindings(
        style_benchmark="references/stylized_2d/style_benchmark.png",
        anatomy_reference="references/stylized_2d/anatomy_reference.png",
        environment_reference="references/stylized_2d/environment_reference.png",
    ),
    capabilities=StyleCapabilities(
        supports_reference_images=True,
        max_reference_budget=2,
        preferred_fidelity=None,
    ),
    render_profile=StyleRenderProfile(
        motion_scale=0.8,
        default_transition="cut",
    ),
)


# ==============================================================================
# Central Registry Mapping
# ==============================================================================
STYLES_REGISTRY: dict[str, VisualStylePreset] = {
    LEGACY_DEFAULT_PRESET.id: LEGACY_DEFAULT_PRESET,
    STICKMAN_TACTICAL_EXPLAINER_PRESET.id: STICKMAN_TACTICAL_EXPLAINER_PRESET,
    STYLIZED_2D_NARRATIVE_PRESET.id: STYLIZED_2D_NARRATIVE_PRESET,
}
_PRESETS_BY_ID = STYLES_REGISTRY


def is_valid_style_id(style_id: str | None) -> bool:
    """Return True if style_id corresponds to a registered preset."""
    if not style_id or not style_id.strip():
        return False
    return style_id.strip().lower() in STYLES_REGISTRY


def get_style_by_id(style_id: str) -> VisualStylePreset:
    """Retrieve a visual style preset strictly by ID.

    Raises KeyError if style_id is None, empty, or unknown.
    """
    if not style_id or not style_id.strip():
        raise KeyError(f"Style ID cannot be empty (got {style_id!r})")
    norm = style_id.strip().lower()
    if norm not in STYLES_REGISTRY:
        raise KeyError(f"Unknown visual style '{style_id}'. Valid styles: {sorted(STYLES_REGISTRY.keys())}")
    return STYLES_REGISTRY[norm]


def get_style(style_id: str | None) -> VisualStylePreset:
    """Retrieve a visual style preset by ID.

    Falls back to LEGACY_DEFAULT_PRESET if style_id is None, empty, or unknown.
    """
    if not style_id or not style_id.strip():
        return LEGACY_DEFAULT_PRESET
    resolved = _PRESETS_BY_ID.get(style_id.strip().lower())
    if resolved is None:
        return LEGACY_DEFAULT_PRESET
    return resolved


def list_styles() -> list[VisualStylePreset]:
    """Return all developer-managed presets in canonical order."""
    return [
        LEGACY_DEFAULT_PRESET,
        STICKMAN_TACTICAL_EXPLAINER_PRESET,
        STYLIZED_2D_NARRATIVE_PRESET,
    ]


class VisualStyleResolutionError(ValueError):
    """Raised when an explicit visual style snapshot or ID cannot be safely resolved."""
    pass


def canonical_snapshot_hash(snapshot: Any = None) -> str:
    """Return a deterministic 64-char hex SHA-256 hash of a visual style snapshot.

    Produces '' if snapshot is None or empty.
    Handles dict, VisualStylePreset, or JSON string with sorted keys and compact separators.
    """
    if snapshot is None:
        return ""
    if isinstance(snapshot, VisualStylePreset):
        data = snapshot.model_dump()
    elif isinstance(snapshot, dict):
        data = snapshot
    elif isinstance(snapshot, str):
        try:
            parsed = json.loads(snapshot)
            if isinstance(parsed, dict):
                data = parsed
            else:
                return hashlib.sha256(snapshot.strip().encode("utf-8")).hexdigest()
        except Exception:
            return hashlib.sha256(snapshot.strip().encode("utf-8")).hexdigest()
    else:
        return ""

    if not data:
        return ""
    try:
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    except Exception:
        return ""


def resolve_visual_style_preset(
    snapshot: Any = None,
    style_id: str | None = None,
    style_version: int | None = None,
) -> VisualStylePreset | None:
    """Resolve a VisualStylePreset with fail-closed validation.

    Rules:
    1. Valid visual_style_snapshot exists:
       - Use it as authoritative source. Do not substitute registry.
    2. Snapshot exists but is malformed, invalid, or cannot be interpreted:
       - Raise a clear VisualStyleResolutionError identifying the invalid snapshot.
    3. Snapshot is missing (None), but visual_style_id exists:
       - Resolve registry preset only if ID is valid and version is compatible.
       - If version is missing (None), follow existing compatibility policy (accept candidate).
       - If ID is unknown or version is incompatible, raise VisualStyleResolutionError.
    4. Both snapshot and visual_style_id are NULL (None or blank):
       - Preserve historical Clean 3D behavior (return None).
       - Never dynamically inspect or inherit channel default.
    """
    # 1. Snapshot handling (fail-closed on malformed snapshot)
    if snapshot is not None:
        if isinstance(snapshot, VisualStylePreset):
            return snapshot
        if isinstance(snapshot, dict):
            try:
                return VisualStylePreset.model_validate(snapshot)
            except Exception as exc:
                raise VisualStyleResolutionError(
                    f"Malformed or invalid visual style snapshot: {exc}"
                ) from exc
        if isinstance(snapshot, str):
            try:
                import json
                parsed = json.loads(snapshot)
                if isinstance(parsed, dict):
                    return VisualStylePreset.model_validate(parsed)
                raise ValueError("Parsed JSON is not a dictionary")
            except Exception as exc:
                raise VisualStyleResolutionError(
                    f"Malformed visual style snapshot JSON string: {exc}"
                ) from exc
        raise VisualStyleResolutionError(
            f"Malformed visual style snapshot of unexpected type '{type(snapshot).__name__}'"
        )

    # 2. Explicit style_id handling (fail-closed on unknown ID or incompatible version)
    clean_style_id = str(style_id).strip() if style_id is not None else ""
    if clean_style_id:
        if not is_valid_style_id(clean_style_id):
            raise VisualStyleResolutionError(
                f"Unknown visual style ID '{clean_style_id}'. Valid styles: {sorted(STYLES_REGISTRY.keys())}"
            )
        candidate = get_style_by_id(clean_style_id)
        if style_version is not None and candidate.version != style_version:
            raise VisualStyleResolutionError(
                f"Incompatible visual style version {style_version} for preset '{clean_style_id}' "
                f"(current registry version is {candidate.version})"
            )
        return candidate

    # 3. Both NULL: return None (historical Clean 3D legacy path)
    return None
