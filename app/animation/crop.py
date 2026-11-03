"""Crop engine, virtual camera reframing, and motion planning for VisualBeats.

Evaluates whether a source scene image can support individual visual beats via deterministic
portrait (9:16) cropping, classifies image-reuse feasibility, and plans smooth virtual-camera
motion without FFmpeg rendering dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Literal

from .analyzer import analyze_image_composition
from .planner import _infer_semantic_focal_point
from .types import (
    DEFAULT_SUBJECT_Y,
    SUBTITLE_SAFETY_MAX_Y,
    AnimationPlan,
    AnimationType,
)

TARGET_ASPECT_RATIO: float = 9.0 / 16.0  # 0.5625

# Feasibility status outcomes
CropFeasibilityStatus = Literal["safe_reuse", "questionable_reuse", "new_image_needed"]

# Absolute framing baseline scales relative to full scene view
SHOT_TYPE_ABSOLUTE_SCALE: dict[str, float] = {
    "extreme-wide": 1.00,
    "aerial": 1.00,
    "wide": 1.00,
    "medium-wide": 1.10,
    "in-context": 1.10,
    "diagram": 1.05,
    "cutaway": 1.05,
    "point-of-view": 1.15,
    "medium": 1.20,
    "medium-close-up": 1.35,
    "close-up": 1.50,
    "extreme-close-up": 1.80,
    "macro": 1.90,
}

# Scale thresholds for feasibility evaluation
SAFE_REUSE_MAX_SCALE: float = 1.45
QUESTIONABLE_REUSE_MAX_SCALE: float = 1.65

# Incompatible camera perspective shifts that cannot be produced by 2D reframing
INCOMPATIBLE_ANGLES: set[tuple[str, str]] = {
    ("eye-level", "top-down"),
    ("eye-level", "aerial"),
    ("eye-level", "overhead"),
    ("low-angle", "top-down"),
    ("low-angle", "aerial"),
    ("low-angle", "overhead"),
    ("high-angle", "ground-level"),
    ("top-down", "eye-level"),
    ("aerial", "eye-level"),
    ("overhead", "eye-level"),
}


@dataclass
class CropRegion:
    """Represents a bounded rectangular crop window inside a source image.

    Coordinates are normalized to [0.0, 1.0] relative to source image dimensions,
    along with exact integer pixel coordinates.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    pixel_x: int
    pixel_y: int
    pixel_width: int
    pixel_height: int
    scale_factor: float = 1.0

    @property
    def normalized_width(self) -> float:
        """Width of crop window in normalized coordinates."""
        return round(self.x_max - self.x_min, 4)

    @property
    def normalized_height(self) -> float:
        """Height of crop window in normalized coordinates."""
        return round(self.y_max - self.y_min, 4)

    @property
    def center_x(self) -> float:
        """Horizontal center of crop window in normalized coordinates."""
        return round((self.x_min + self.x_max) / 2.0, 4)

    @property
    def center_y(self) -> float:
        """Vertical center of crop window in normalized coordinates."""
        return round((self.y_min + self.y_max) / 2.0, 4)

    @property
    def aspect_ratio(self) -> float:
        """Aspect ratio (width / height) of the pixel crop region."""
        if self.pixel_height <= 0:
            return 0.0
        return round(self.pixel_width / self.pixel_height, 4)

    def to_dict(self) -> dict[str, Any]:
        """Convert crop region to JSON-serializable dictionary."""
        return {
            "x_min": round(self.x_min, 4),
            "y_min": round(self.y_min, 4),
            "x_max": round(self.x_max, 4),
            "y_max": round(self.y_max, 4),
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "pixel_width": self.pixel_width,
            "pixel_height": self.pixel_height,
            "scale_factor": round(self.scale_factor, 3),
            "center_x": self.center_x,
            "center_y": self.center_y,
            "aspect_ratio": self.aspect_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CropRegion:
        """Construct a CropRegion from a dictionary."""
        return cls(
            x_min=float(data["x_min"]),
            y_min=float(data["y_min"]),
            x_max=float(data["x_max"]),
            y_max=float(data["y_max"]),
            pixel_x=int(data["pixel_x"]),
            pixel_y=int(data["pixel_y"]),
            pixel_width=int(data["pixel_width"]),
            pixel_height=int(data["pixel_height"]),
            scale_factor=float(data.get("scale_factor", 1.0)),
        )


@dataclass
class CropFeasibilityResult:
    """Evaluation result communicating whether a source image can support a beat framing."""

    status: CropFeasibilityStatus
    reason: str
    source_dimensions: tuple[int, int]
    target_aspect_ratio: float = TARGET_ASPECT_RATIO
    requested_shot_type: str = "medium"
    scale_factor: float | None = None
    proposed_crop: CropRegion | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert feasibility result to JSON-serializable dictionary."""
        return {
            "status": self.status,
            "reason": self.reason,
            "source_dimensions": list(self.source_dimensions),
            "target_aspect_ratio": round(self.target_aspect_ratio, 4),
            "requested_shot_type": self.requested_shot_type,
            "scale_factor": round(self.scale_factor, 3) if self.scale_factor is not None else None,
            "proposed_crop": self.proposed_crop.to_dict() if self.proposed_crop else None,
            "warnings": list(self.warnings),
        }


@dataclass
class BeatCameraPlan:
    """Smooth virtual camera motion plan for an individual visual beat."""

    animation_type: AnimationType
    start_crop: CropRegion
    end_crop: CropRegion
    start_scale: float
    end_scale: float
    focal_x: float
    focal_y: float
    intensity: str = "subtle"
    duration_seconds: float = 2.5
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert camera plan to JSON-serializable dictionary."""
        return {
            "animation_type": self.animation_type,
            "start_crop": self.start_crop.to_dict(),
            "end_crop": self.end_crop.to_dict(),
            "start_scale": round(self.start_scale, 3),
            "end_scale": round(self.end_scale, 3),
            "focal_coords": [round(self.focal_x, 3), round(self.focal_y, 3)],
            "intensity": self.intensity,
            "duration_seconds": round(self.duration_seconds, 3),
            "reasoning": self.reasoning,
        }

    def to_animation_plan(
        self,
        focal_desc: str = "",
        max_scale: float | None = None,
    ) -> AnimationPlan:
        """Convert BeatCameraPlan into a standard AnimationPlan for downstream rendering."""
        if max_scale is not None:
            effective_max = max_scale
        elif self.start_scale > 1.15:
            effective_max = max(1.65, self.end_scale)
        else:
            effective_max = 1.15

        return AnimationPlan(
            animation_type=self.animation_type,
            focal_point=focal_desc,
            focal_x=round(self.focal_x, 3),
            focal_y=round(self.focal_y, 3),
            start_scale=round(self.start_scale, 3),
            end_scale=round(self.end_scale, 3),
            start_x=round(self.start_crop.center_x, 3),
            start_y=round(self.start_crop.center_y, 3),
            end_x=round(self.end_crop.center_x, 3),
            end_y=round(self.end_crop.center_y, 3),
            intensity=self.intensity,
            duration_seconds=round(self.duration_seconds, 3),
            reasoning=self.reasoning,
        ).validate_and_clamp(max_scale=effective_max, allow_static_scale=True)


@dataclass
class BeatPlanningResult:
    """Complete planning output for an individual visual beat."""

    beat_id: str
    scene_id: str
    order: int
    feasibility: CropFeasibilityResult
    camera_plan: BeatCameraPlan | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert beat planning result to JSON-serializable dictionary."""
        return {
            "beat_id": self.beat_id,
            "scene_id": self.scene_id,
            "order": self.order,
            "feasibility": self.feasibility.to_dict(),
            "camera_plan": self.camera_plan.to_dict() if self.camera_plan else None,
        }


def calculate_portrait_crop(
    source_width: int,
    source_height: int,
    focal_x: float = 0.5,
    focal_y: float = DEFAULT_SUBJECT_Y,
    scale_factor: float = 1.0,
    target_aspect: float = TARGET_ASPECT_RATIO,
) -> CropRegion:
    """Calculate a deterministic, strictly bounded 9:16 portrait crop window inside a source image.

    Guarantees:
    1. Resulting crop region aspect ratio strictly matches target_aspect (9:16 = 0.5625).
    2. Crop coordinates never exceed source image boundaries (0 <= x_min < x_max <= 1.0).
    3. Pixel dimensions are even integers to ensure video encoder (yuv420p) compatibility.
    4. Focal point centering is clamped gracefully without warping or cropping outside bounds.

    Args:
        source_width: Source image pixel width (> 0).
        source_height: Source image pixel height (> 0).
        focal_x: Normalized horizontal focal point (0.0 to 1.0).
        focal_y: Normalized vertical focal point (0.0 to 1.0).
        scale_factor: Zoom/crop magnification factor (>= 1.0, where 1.0 is full 9:16 view).
        target_aspect: Desired aspect ratio (default 9/16 = 0.5625).

    Returns:
        CropRegion with exact pixel and normalized coordinates.
    """
    if source_width <= 0 or source_height <= 0:
        raise ValueError(f"Invalid image dimensions: {source_width}x{source_height}")

    scale = max(1.0, float(scale_factor))

    # 1. Base maximum 9:16 rectangle that fits completely inside source dimensions
    src_ratio = source_width / source_height
    if src_ratio >= target_aspect:
        # Source is wider than target aspect (e.g. 2:3, 1:1, 16:9)
        base_h = source_height
        base_w = int(round(base_h * target_aspect))
        if base_w % 2 != 0:
            base_w -= 1
    else:
        # Source is narrower than target aspect
        base_w = source_width
        base_h = int(round(base_w / target_aspect))
        if base_h % 2 != 0:
            base_h -= 1

    # 2. Scale down the crop window according to zoom/framing factor
    crop_h = max(2, int(round(base_h / scale)))
    if crop_h % 2 != 0:
        crop_h -= 1
    crop_w = max(2, int(round(crop_h * target_aspect)))
    if crop_w % 2 != 0:
        crop_w -= 1

    # Ensure crop window does not exceed source dimensions
    crop_w = min(crop_w, source_width)
    crop_h = min(crop_h, source_height)

    # 3. Position crop window centered around focal point with boundary clamping
    fx = max(0.0, min(1.0, float(focal_x)))
    fy = max(0.0, min(1.0, float(focal_y)))

    center_px_x = fx * source_width
    center_px_y = fy * source_height

    # Clamping range for center coordinate:
    min_cx = crop_w / 2.0
    max_cx = source_width - (crop_w / 2.0)
    clamped_cx = max(min_cx, min(max_cx, center_px_x))

    min_cy = crop_h / 2.0
    max_cy = source_height - (crop_h / 2.0)
    clamped_cy = max(min_cy, min(max_cy, center_px_y))

    # Compute top-left pixel coordinates
    px_x = int(round(clamped_cx - crop_w / 2.0))
    px_y = int(round(clamped_cy - crop_h / 2.0))

    # Strict clamping to source image boundaries
    px_x = max(0, min(source_width - crop_w, px_x))
    px_y = max(0, min(source_height - crop_h, px_y))

    # Normalized bounds
    x_min = px_x / float(source_width)
    y_min = px_y / float(source_height)
    x_max = (px_x + crop_w) / float(source_width)
    y_max = (px_y + crop_h) / float(source_height)

    return CropRegion(
        x_min=round(x_min, 6),
        y_min=round(y_min, 6),
        x_max=round(x_max, 6),
        y_max=round(y_max, 6),
        pixel_x=px_x,
        pixel_y=px_y,
        pixel_width=crop_w,
        pixel_height=crop_h,
        scale_factor=round(scale, 3),
    )


def evaluate_beat_crop_feasibility(
    beat: Any,
    parent_scene: Any = None,
    source_width: int = 1024,
    source_height: int = 1536,
    image_path: Path | str | None = None,
) -> CropFeasibilityResult:
    """Evaluate whether a source image can support the requested beat framing.

    Outcomes:
    - `safe_reuse`: The source image cleanly supports the framing (scale <= 1.45x) with
      acceptable composition and visual clarity.
    - `questionable_reuse`: The requested framing pushes composition or zoom boundaries
      (1.45x < scale <= 1.65x) with risk of slight softening, but is technically croppable.
    - `new_image_needed`: The beat cannot be cleanly produced by cropping (e.g. requires
      unseen wide context, extreme zoom > 1.65x, conflicting camera angle, or explicit new image).
    """
    warnings: list[str] = []

    # 1. Extract beat attributes (supporting both Pydantic model and dict)
    if isinstance(beat, dict):
        beat_shot_type = str(beat.get("shot_type") or "medium").lower().strip()
        beat_angle = str(beat.get("camera_angle") or "eye-level").lower().strip()
        beat_tech = str(beat.get("visual_technique") or "cinematic-photo").lower().strip()
        beat_strategy = str(beat.get("image_strategy") or "reuse_scene_image").strip()
        beat_desc = str(beat.get("visual_description") or "").strip()
        beat_focal = str(beat.get("focal_point") or "").strip()
    else:
        beat_shot_type = str(getattr(beat, "shot_type", "medium") or "medium").lower().strip()
        beat_angle = str(getattr(beat, "camera_angle", "eye-level") or "eye-level").lower().strip()
        beat_tech = str(getattr(beat, "visual_technique", "cinematic-photo") or "cinematic-photo").lower().strip()
        beat_strategy = str(getattr(beat, "image_strategy", "reuse_scene_image") or "reuse_scene_image").strip()
        beat_desc = str(getattr(beat, "visual_description", "") or "").strip()
        beat_focal = str(getattr(beat, "focal_point", "") or "").strip()

    # 2. Extract parent scene attributes
    parent_shot_type = "medium"
    parent_angle = "eye-level"
    parent_tech = "cinematic-photo"
    parent_focal = ""
    if parent_scene is not None:
        if isinstance(parent_scene, dict):
            parent_shot_type = str(parent_scene.get("shot_type") or "medium").lower().strip()
            parent_angle = str(parent_scene.get("camera_angle") or "eye-level").lower().strip()
            parent_tech = str(parent_scene.get("visual_technique") or "cinematic-photo").lower().strip()
            parent_focal = str(parent_scene.get("focal_point") or "").strip()
        else:
            parent_shot_type = str(getattr(parent_scene, "shot_type", "medium") or "medium").lower().strip()
            parent_angle = str(getattr(parent_scene, "camera_angle", "eye-level") or "eye-level").lower().strip()
            parent_tech = str(getattr(parent_scene, "visual_technique", "cinematic-photo") or "cinematic-photo").lower().strip()
            parent_focal = str(getattr(parent_scene, "focal_point", "") or "").strip()

    # 3. Check explicit image strategy
    if beat_strategy == "new_image":
        return CropFeasibilityResult(
            status="new_image_needed",
            reason="Visual beat explicitly specifies image strategy 'new_image'.",
            source_dimensions=(source_width, source_height),
            requested_shot_type=beat_shot_type,
            warnings=warnings,
        )

    # 4. Check visual technique compatibility (e.g. photo vs diagram/schematic)
    photo_techs = {"cinematic-photo", "macro-photo", "photo", "candid-photo"}
    diagram_techs = {"cutaway-diagram", "technical-illustration", "blueprint", "infographic", "schematic"}
    if parent_tech in photo_techs and beat_tech in diagram_techs:
        return CropFeasibilityResult(
            status="new_image_needed",
            reason=f"Visual technique mismatch: cannot crop photograph ('{parent_tech}') into diagram ('{beat_tech}').",
            source_dimensions=(source_width, source_height),
            requested_shot_type=beat_shot_type,
            warnings=warnings,
        )

    # 5. Check camera angle compatibility
    if (parent_angle, beat_angle) in INCOMPATIBLE_ANGLES:
        return CropFeasibilityResult(
            status="new_image_needed",
            reason=f"Camera angle shift from '{parent_angle}' to '{beat_angle}' cannot be achieved by 2D reframing.",
            source_dimensions=(source_width, source_height),
            requested_shot_type=beat_shot_type,
            warnings=warnings,
        )

    # 6. Calculate relative crop scale factor between beat and parent scene
    base_parent_scale = SHOT_TYPE_ABSOLUTE_SCALE.get(parent_shot_type, 1.20)
    target_beat_scale = SHOT_TYPE_ABSOLUTE_SCALE.get(beat_shot_type, 1.20)

    # Relative magnification required:
    relative_scale = round(target_beat_scale / base_parent_scale, 3)

    # If beat requests framing wider than parent image (e.g. wide from close-up)
    if relative_scale < 0.95:
        return CropFeasibilityResult(
            status="new_image_needed",
            reason=(
                f"Requested framing ('{beat_shot_type}') is wider than parent scene image "
                f"('{parent_shot_type}') and cannot be obtained by cropping."
            ),
            source_dimensions=(source_width, source_height),
            requested_shot_type=beat_shot_type,
            scale_factor=relative_scale,
            warnings=warnings,
        )

    # If beat requests extreme zoom beyond clarity threshold
    if relative_scale > QUESTIONABLE_REUSE_MAX_SCALE:
        return CropFeasibilityResult(
            status="new_image_needed",
            reason=(
                f"Requested framing ('{beat_shot_type}') requires digital crop factor of {relative_scale:.2f}x "
                f"from parent scene ('{parent_shot_type}'), exceeding the {QUESTIONABLE_REUSE_MAX_SCALE:.2f}x clarity limit."
            ),
            source_dimensions=(source_width, source_height),
            requested_shot_type=beat_shot_type,
            scale_factor=relative_scale,
            warnings=warnings,
        )

    # 7. Determine focal anchor coordinates
    fx, fy = 0.5, DEFAULT_SUBJECT_Y
    if image_path and Path(image_path).is_file():
        img_analysis = analyze_image_composition(image_path)
        cx = float(img_analysis.get("centroid_x", 0.5))
        cy = float(img_analysis.get("centroid_y", DEFAULT_SUBJECT_Y))
        fx = 0.50 if abs(cx - 0.50) <= 0.04 else round(max(0.44, min(0.56, cx)), 2)
        fy = DEFAULT_SUBJECT_Y if abs(cy - DEFAULT_SUBJECT_Y) <= 0.03 else round(max(0.36, min(0.48, cy)), 2)
    elif beat_focal or parent_focal:
        focal_ctx = {"focal_point": beat_focal or parent_focal, "shot_type": beat_shot_type}
        sx, sy = _infer_semantic_focal_point(focal_ctx)
        fx = max(0.45, min(0.55, sx))
        fy = max(0.38, min(0.46, sy))
    else:
        warnings.append("Focal coordinates not explicitly provided; defaulted to center composition (0.50, 0.42).")

    # Check and enforce subtitle safety zone on vertical focal anchor
    if fy > SUBTITLE_SAFETY_MAX_Y:
        fy = SUBTITLE_SAFETY_MAX_Y
        warnings.append(f"Focal vertical anchor adjusted to {SUBTITLE_SAFETY_MAX_Y:.2f} to stay above subtitle safe zone.")

    # 8. Compute proposed 9:16 crop window
    crop_scale = max(1.0, relative_scale)
    proposed_crop = calculate_portrait_crop(
        source_width=source_width,
        source_height=source_height,
        focal_x=fx,
        focal_y=fy,
        scale_factor=crop_scale,
        target_aspect=TARGET_ASPECT_RATIO,
    )

    # 9. Determine status: safe_reuse vs questionable_reuse
    if crop_scale > SAFE_REUSE_MAX_SCALE:
        status: CropFeasibilityStatus = "questionable_reuse"
        warnings.append(
            f"Crop scale factor ({crop_scale:.2f}x) exceeds safe threshold ({SAFE_REUSE_MAX_SCALE:.2f}x); "
            "minor digital crop softening may occur."
        )
        reason = (
            f"Reframing from '{parent_shot_type}' to '{beat_shot_type}' is feasible ({crop_scale:.2f}x crop), "
            "but approaches maximum recommended digital zoom."
        )
    else:
        status = "safe_reuse"
        reason = f"Reframing from '{parent_shot_type}' to '{beat_shot_type}' is clean and feasible ({crop_scale:.2f}x crop)."

    return CropFeasibilityResult(
        status=status,
        reason=reason,
        source_dimensions=(source_width, source_height),
        target_aspect_ratio=TARGET_ASPECT_RATIO,
        requested_shot_type=beat_shot_type,
        scale_factor=crop_scale,
        proposed_crop=proposed_crop,
        warnings=warnings,
    )


def plan_beat_camera_motion(
    beat: Any,
    crop_region: CropRegion,
    source_width: int,
    source_height: int,
    focal_x: float = 0.5,
    focal_y: float = DEFAULT_SUBJECT_Y,
    duration_seconds: float = 2.5,
    preferred_animation: AnimationType | None = None,
    prev_anim_type: str | None = None,
) -> BeatCameraPlan:
    """Plan smooth virtual camera movement within an evaluated beat crop region.

    Ensures start_crop and end_crop remain 100% within source image boundaries without jitter.
    """
    # 1. Resolve duration
    dur = duration_seconds
    if isinstance(beat, dict):
        beat_dur = beat.get("duration_seconds")
        beat_shot_type = str(beat.get("shot_type") or "medium").lower()
    else:
        beat_dur = getattr(beat, "duration_seconds", None)
        beat_shot_type = str(getattr(beat, "shot_type", "medium") or "medium").lower()

    if beat_dur is not None:
        try:
            val = float(beat_dur)
            if val > 0.1:
                dur = val
        except (ValueError, TypeError):
            pass

    # 2. Select conservative animation type
    if preferred_animation:
        anim_type: AnimationType = preferred_animation
    elif dur < 1.6:
        # Avoid rapid motion on short beats; static hold gives viewers time to focus
        anim_type = "static"
    elif beat_shot_type in ("diagram", "cutaway", "cross-section", "blueprint", "schematic"):
        anim_type = "static"
    elif prev_anim_type == "zoom-in":
        # Avoid repeated consecutive push-ins across beats to prevent visual fatigue
        if beat_shot_type in ("close-up", "extreme-close-up", "macro"):
            anim_type = "zoom-out"
        else:
            anim_type = "static"
    elif beat_shot_type in ("close-up", "extreme-close-up", "macro"):
        anim_type = "zoom-out"
    else:
        anim_type = "zoom-in"

    base_scale = max(1.0, crop_region.scale_factor)
    fx = max(0.05, min(0.95, focal_x))
    fy = max(0.10, min(SUBTITLE_SAFETY_MAX_Y, focal_y))

    # 3. Calculate start_crop and end_crop based on animation motion
    if anim_type == "static":
        start_scale = base_scale
        end_scale = base_scale
        start_crop = crop_region
        end_crop = crop_region
        reasoning = "Static hold maintains crisp legibility of details."

    elif anim_type == "zoom-in":
        start_scale = base_scale
        end_scale = round(base_scale * 1.05, 3)
        start_crop = crop_region
        end_crop = calculate_portrait_crop(
            source_width=source_width,
            source_height=source_height,
            focal_x=fx,
            focal_y=fy,
            scale_factor=end_scale,
        )
        reasoning = f"Gentle push-in from {start_scale:.2f}x to {end_scale:.2f}x draws viewer focus."

    elif anim_type == "zoom-out":
        start_scale = round(base_scale * 1.05, 3)
        end_scale = base_scale
        start_crop = calculate_portrait_crop(
            source_width=source_width,
            source_height=source_height,
            focal_x=fx,
            focal_y=fy,
            scale_factor=start_scale,
        )
        end_crop = crop_region
        reasoning = f"Gentle pull-out from {start_scale:.2f}x to {end_scale:.2f}x reveals surrounding detail."

    elif anim_type == "pan-left":
        start_scale = max(1.02, base_scale)
        end_scale = start_scale
        start_crop = calculate_portrait_crop(
            source_width=source_width,
            source_height=source_height,
            focal_x=min(0.92, fx + 0.03),
            focal_y=fy,
            scale_factor=start_scale,
        )
        end_crop = calculate_portrait_crop(
            source_width=source_width,
            source_height=source_height,
            focal_x=max(0.08, fx - 0.03),
            focal_y=fy,
            scale_factor=end_scale,
        )
        reasoning = "Restrained lateral pan to the left tracks scene visual direction."

    elif anim_type == "pan-right":
        start_scale = max(1.02, base_scale)
        end_scale = start_scale
        start_crop = calculate_portrait_crop(
            source_width=source_width,
            source_height=source_height,
            focal_x=max(0.08, fx - 0.03),
            focal_y=fy,
            scale_factor=start_scale,
        )
        end_crop = calculate_portrait_crop(
            source_width=source_width,
            source_height=source_height,
            focal_x=min(0.92, fx + 0.03),
            focal_y=fy,
            scale_factor=end_scale,
        )
        reasoning = "Restrained lateral pan to the right tracks scene visual direction."

    elif anim_type == "pan-up":
        start_scale = max(1.02, base_scale)
        end_scale = start_scale
        start_crop = calculate_portrait_crop(
            source_width=source_width,
            source_height=source_height,
            focal_x=fx,
            focal_y=min(SUBTITLE_SAFETY_MAX_Y, fy + 0.03),
            scale_factor=start_scale,
        )
        end_crop = calculate_portrait_crop(
            source_width=source_width,
            source_height=source_height,
            focal_x=fx,
            focal_y=max(0.15, fy - 0.03),
            scale_factor=end_scale,
        )
        reasoning = "Restrained vertical pan upwards reveals subject structure."

    elif anim_type == "pan-down":
        start_scale = max(1.02, base_scale)
        end_scale = start_scale
        start_crop = calculate_portrait_crop(
            source_width=source_width,
            source_height=source_height,
            focal_x=fx,
            focal_y=max(0.15, fy - 0.03),
            scale_factor=start_scale,
        )
        end_crop = calculate_portrait_crop(
            source_width=source_width,
            source_height=source_height,
            focal_x=fx,
            focal_y=min(SUBTITLE_SAFETY_MAX_Y, fy + 0.03),
            scale_factor=end_scale,
        )
        reasoning = "Restrained vertical pan downwards, respecting subtitle safety."

    else:  # subtle-drift
        start_scale = max(1.02, base_scale)
        end_scale = round(base_scale * 1.03, 3)
        start_crop = calculate_portrait_crop(
            source_width=source_width,
            source_height=source_height,
            focal_x=max(0.08, fx - 0.015),
            focal_y=fy,
            scale_factor=start_scale,
        )
        end_crop = calculate_portrait_crop(
            source_width=source_width,
            source_height=source_height,
            focal_x=min(0.92, fx + 0.015),
            focal_y=fy,
            scale_factor=end_scale,
        )
        reasoning = "Subtle cinematic drift creates organic visual interest."

    return BeatCameraPlan(
        animation_type=anim_type,
        start_crop=start_crop,
        end_crop=end_crop,
        start_scale=round(start_scale, 3),
        end_scale=round(end_scale, 3),
        focal_x=round(fx, 3),
        focal_y=round(fy, 3),
        intensity="subtle",
        duration_seconds=round(dur, 3),
        reasoning=reasoning,
    )


def evaluate_scene_beats_crop_plan(
    scene_data: Any,
    source_width: int = 1024,
    source_height: int = 1536,
    image_path: Path | str | None = None,
) -> list[BeatPlanningResult]:
    """Evaluate image-reuse feasibility and calculate camera plans for all beats in a scene.

    Returns an empty list for legacy scenes lacking visual beats.
    """
    # Extract visual beats
    beats: list[Any] = []
    if isinstance(scene_data, dict):
        beats = scene_data.get("visual_beats") or []
        scene_id = str(scene_data.get("index") or scene_data.get("id") or "1")
    else:
        beats = getattr(scene_data, "visual_beats", None) or []
        scene_id = str(getattr(scene_data, "index", None) or getattr(scene_data, "id", None) or "1")

    if not beats:
        return []

    results: list[BeatPlanningResult] = []
    prev_anim_type: str | None = None
    for idx, beat in enumerate(beats, start=1):
        if isinstance(beat, dict):
            b_id = str(beat.get("id") or f"beat_{scene_id}_{idx}")
            b_order = int(beat.get("order") or idx)
            b_dur = float(beat.get("duration_seconds") or 2.5)
        else:
            b_id = str(getattr(beat, "id", "") or f"beat_{scene_id}_{idx}")
            b_order = int(getattr(beat, "order", idx) or idx)
            b_dur = float(getattr(beat, "duration_seconds", 2.5) or 2.5)

        feasibility = evaluate_beat_crop_feasibility(
            beat=beat,
            parent_scene=scene_data,
            source_width=source_width,
            source_height=source_height,
            image_path=image_path,
        )

        camera_plan: BeatCameraPlan | None = None
        if feasibility.status in ("safe_reuse", "questionable_reuse") and feasibility.proposed_crop:
            fx = feasibility.proposed_crop.center_x
            fy = feasibility.proposed_crop.center_y
            camera_plan = plan_beat_camera_motion(
                beat=beat,
                crop_region=feasibility.proposed_crop,
                source_width=source_width,
                source_height=source_height,
                focal_x=fx,
                focal_y=fy,
                duration_seconds=b_dur,
                prev_anim_type=prev_anim_type,
            )
            if camera_plan:
                prev_anim_type = camera_plan.animation_type

        results.append(
            BeatPlanningResult(
                beat_id=b_id,
                scene_id=scene_id,
                order=b_order,
                feasibility=feasibility,
                camera_plan=camera_plan,
            )
        )

    return results

