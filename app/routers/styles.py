"""Visual styles API router."""

from fastapi import APIRouter, HTTPException

from ..schemas import VisualStyleSummaryOut, VisualStylesListOut
from ..styles import DEFAULT_STYLE_ID, VisualStylePreset, get_style_by_id, list_styles

router = APIRouter(prefix="/visual-styles", tags=["visual-styles"])


@router.get("", response_model=VisualStylesListOut)
def list_visual_styles():
    """Retrieve all registered visual style presets in canonical order."""
    styles = list_styles()
    summaries = [
        VisualStyleSummaryOut(
            id=s.id,
            name=s.display_name,
            version=s.version,
            category=s.category,
            description=s.description,
            preview_image_path=s.preview_image_path,
            is_default=(s.id == DEFAULT_STYLE_ID),
        )
        for s in styles
    ]
    return VisualStylesListOut(default_style_id=DEFAULT_STYLE_ID, styles=summaries)


@router.get("/{style_id}", response_model=VisualStylePreset)
def get_visual_style(style_id: str):
    """Retrieve full details of a specific visual style preset by ID."""
    try:
        return get_style_by_id(style_id)
    except KeyError:
        raise HTTPException(404, f"Visual style '{style_id}' not found")
