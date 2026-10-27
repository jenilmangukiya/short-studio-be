"""AI-directed camera animation package for Shorts Studio."""

from .analyzer import analyze_image_composition
from .planner import plan_scene_animation
from .renderer import (
    build_fallback_ken_burns_filter,
    build_ffmpeg_zoompan_filter,
)
from .types import (
    ALLOWED_ANIMATION_TYPES,
    DEFAULT_SUBJECT_Y,
    SUBTITLE_SAFETY_MAX_Y,
    AnimationPlan,
    AnimationType,
)

__all__ = [
    "ALLOWED_ANIMATION_TYPES",
    "DEFAULT_SUBJECT_Y",
    "SUBTITLE_SAFETY_MAX_Y",
    "AnimationPlan",
    "AnimationType",
    "analyze_image_composition",
    "build_fallback_ken_burns_filter",
    "build_ffmpeg_zoompan_filter",
    "plan_scene_animation",
]
