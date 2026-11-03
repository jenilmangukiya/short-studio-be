"""AI-directed camera animation package for Shorts Studio."""

from .analyzer import analyze_image_composition
from .crop import (
    QUESTIONABLE_REUSE_MAX_SCALE,
    SAFE_REUSE_MAX_SCALE,
    TARGET_ASPECT_RATIO,
    BeatCameraPlan,
    BeatPlanningResult,
    CropFeasibilityResult,
    CropFeasibilityStatus,
    CropRegion,
    calculate_portrait_crop,
    evaluate_beat_crop_feasibility,
    evaluate_scene_beats_crop_plan,
    plan_beat_camera_motion,
)
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
    "BeatCameraPlan",
    "BeatPlanningResult",
    "CropFeasibilityResult",
    "CropFeasibilityStatus",
    "CropRegion",
    "DEFAULT_SUBJECT_Y",
    "QUESTIONABLE_REUSE_MAX_SCALE",
    "SAFE_REUSE_MAX_SCALE",
    "SUBTITLE_SAFETY_MAX_Y",
    "TARGET_ASPECT_RATIO",
    "AnimationPlan",
    "AnimationType",
    "analyze_image_composition",
    "build_fallback_ken_burns_filter",
    "build_ffmpeg_zoompan_filter",
    "calculate_portrait_crop",
    "evaluate_beat_crop_feasibility",
    "evaluate_scene_beats_crop_plan",
    "plan_beat_camera_motion",
    "plan_scene_animation",
]
