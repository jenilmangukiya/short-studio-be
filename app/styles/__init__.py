"""Visual Style System module for Shorts Studio."""

from .adapter import (
    AdaptedEntity,
    adapt_character,
    adapt_concept,
    adapt_environment,
    adapt_object,
)
from .references import (
    DEFAULT_REFERENCE_BUDGET,
    ReferenceSelectionEngine,
    select_scene_references,
)
from .registry import (
    DEFAULT_STYLE_ID,
    LEGACY_DEFAULT_PRESET,
    STICKMAN_TACTICAL_EXPLAINER_PRESET,
    STYLES_REGISTRY,
    STYLIZED_2D_NARRATIVE_PRESET,
    get_style,
    get_style_by_id,
    is_valid_style_id,
    list_styles,
)
from .schemas import (
    ReferenceMetadata,
    ReferenceRole,
    ReferenceSelectionResult,
    RejectedReference,
    SelectedReference,
    StyleAssetBindings,
    StyleCapabilities,
    StyleDrawingRules,
    StyleRenderProfile,
    VisualStylePreset,
)

__all__ = [
    "AdaptedEntity",
    "DEFAULT_REFERENCE_BUDGET",
    "DEFAULT_STYLE_ID",
    "LEGACY_DEFAULT_PRESET",
    "ReferenceMetadata",
    "ReferenceRole",
    "ReferenceSelectionEngine",
    "ReferenceSelectionResult",
    "RejectedReference",
    "STICKMAN_TACTICAL_EXPLAINER_PRESET",
    "STYLES_REGISTRY",
    "STYLIZED_2D_NARRATIVE_PRESET",
    "SelectedReference",
    "StyleAssetBindings",
    "StyleCapabilities",
    "StyleDrawingRules",
    "StyleRenderProfile",
    "VisualStylePreset",
    "adapt_character",
    "adapt_concept",
    "adapt_environment",
    "adapt_object",
    "get_style",
    "get_style_by_id",
    "is_valid_style_id",
    "list_styles",
    "select_scene_references",
]
