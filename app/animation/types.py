"""Types and data structures for AI-directed camera animation in Shorts Studio."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, get_args

AnimationType = Literal[
    "zoom-in",
    "zoom-out",
    "pan-left",
    "pan-right",
    "pan-up",
    "pan-down",
    "subtle-drift",
    "static",
]

ALLOWED_ANIMATION_TYPES: tuple[str, ...] = (
    "zoom-in",
    "zoom-out",
    "pan-left",
    "pan-right",
    "pan-up",
    "pan-down",
    "subtle-drift",
    "static",
)

# Subtitle safety boundary: focal point center should not drift below 68% frame height
SUBTITLE_SAFETY_MAX_Y: float = 0.68
DEFAULT_SUBJECT_Y: float = 0.42


@dataclass
class AnimationPlan:
    """Structured camera animation instructions for a single scene in a video.

    Attributes:
        animation_type: Camera movement style ('zoom-in', 'zoom-out', 'pan-left', etc.).
        focal_point: Description of the subject or focal element the camera tracks.
        focal_x: Normalized horizontal coordinate of focal subject (0.0 to 1.0).
        focal_y: Normalized vertical coordinate of focal subject (0.0 to 1.0).
        start_scale: Beginning zoom scale factor (>= 1.00, <= 1.15).
        end_scale: Ending zoom scale factor (>= 1.00, <= 1.15).
        start_x: Normalized starting camera viewport center X (0.0 to 1.0).
        start_y: Normalized starting camera viewport center Y (0.0 to 1.0).
        end_x: Normalized ending camera viewport center X (0.0 to 1.0).
        end_y: Normalized ending camera viewport center Y (0.0 to 1.0).
        intensity: Movement intensity ('subtle', 'moderate', 'restrained').
        duration_seconds: Scene duration in seconds.
        reasoning: Brief explanation of why this motion suits the scene composition and narration.
    """

    animation_type: AnimationType = "zoom-in"
    focal_point: str = ""
    focal_x: float = 0.5
    focal_y: float = DEFAULT_SUBJECT_Y
    start_scale: float = 1.00
    end_scale: float = 1.05
    start_x: float = 0.5
    start_y: float = DEFAULT_SUBJECT_Y
    end_x: float = 0.5
    end_y: float = DEFAULT_SUBJECT_Y
    intensity: str = "subtle"
    duration_seconds: float = 2.5
    reasoning: str = ""

    def validate_and_clamp(self) -> AnimationPlan:
        """Sanitize and clamp all parameters to safe operational bounds.

        Prevents extreme zoom, out-of-bounds crops, subtitle collisions, or invalid types.
        """
        # Validate animation type
        anim_type = str(self.animation_type).lower().strip()
        if anim_type not in ALLOWED_ANIMATION_TYPES:
            anim_type = "zoom-in"

        # Clamp zoom scales to restrained [1.00, 1.15] range to preserve image clarity
        s_start = max(1.00, min(1.15, float(self.start_scale)))
        s_end = max(1.00, min(1.15, float(self.end_scale)))

        if anim_type == "static":
            s_start = 1.00
            s_end = 1.00

        # Clamp normalized coordinates to [0.0, 1.0]
        fx = max(0.05, min(0.95, float(self.focal_x)))
        # Respect subtitle safety zone on vertical axis
        fy = max(0.10, min(SUBTITLE_SAFETY_MAX_Y, float(self.focal_y)))

        sx = max(0.05, min(0.95, float(self.start_x)))
        sy = max(0.10, min(SUBTITLE_SAFETY_MAX_Y, float(self.start_y)))

        ex = max(0.05, min(0.95, float(self.end_x)))
        ey = max(0.10, min(SUBTITLE_SAFETY_MAX_Y, float(self.end_y)))

        # Clean reasoning
        reasoning_clean = str(self.reasoning or "").strip()

        return AnimationPlan(
            animation_type=anim_type,  # type: ignore[arg-type]
            focal_point=str(self.focal_point or "").strip(),
            focal_x=round(fx, 3),
            focal_y=round(fy, 3),
            start_scale=round(s_start, 3),
            end_scale=round(s_end, 3),
            start_x=round(sx, 3),
            start_y=round(sy, 3),
            end_x=round(ex, 3),
            end_y=round(ey, 3),
            intensity=str(self.intensity or "subtle").lower(),
            duration_seconds=max(0.1, round(float(self.duration_seconds), 3)),
            reasoning=reasoning_clean,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "animation_type": self.animation_type,
            "focal_point": self.focal_point,
            "focal_coords": [self.focal_x, self.focal_y],
            "start_coords": [self.start_x, self.start_y],
            "end_coords": [self.end_x, self.end_y],
            "start_scale": self.start_scale,
            "end_scale": self.end_scale,
            "intensity": self.intensity,
            "duration_seconds": self.duration_seconds,
            "reasoning": self.reasoning,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], duration_seconds: float = 2.5) -> AnimationPlan:
        """Safely construct an AnimationPlan from a dictionary with fallbacks."""
        if not isinstance(data, dict):
            return AnimationPlan(duration_seconds=duration_seconds).validate_and_clamp()

        anim_type = data.get("animation_type") or "zoom-in"

        # Coordinates may be provided as lists [x, y] or separate keys focal_x, focal_y
        f_coords = data.get("focal_coords")
        if isinstance(f_coords, (list, tuple)) and len(f_coords) >= 2:
            fx, fy = float(f_coords[0]), float(f_coords[1])
        else:
            fx = float(data.get("focal_x", 0.5))
            fy = float(data.get("focal_y", DEFAULT_SUBJECT_Y))

        s_coords = data.get("start_coords")
        if isinstance(s_coords, (list, tuple)) and len(s_coords) >= 2:
            sx, sy = float(s_coords[0]), float(s_coords[1])
        else:
            sx = float(data.get("start_x", fx))
            sy = float(data.get("start_y", fy))

        e_coords = data.get("end_coords")
        if isinstance(e_coords, (list, tuple)) and len(e_coords) >= 2:
            ex, ey = float(e_coords[0]), float(e_coords[1])
        else:
            ex = float(data.get("end_x", fx))
            ey = float(data.get("end_y", fy))

        plan = cls(
            animation_type=anim_type,  # type: ignore[arg-type]
            focal_point=str(data.get("focal_point", "")),
            focal_x=fx,
            focal_y=fy,
            start_scale=float(data.get("start_scale", 1.00)),
            end_scale=float(data.get("end_scale", 1.05)),
            start_x=sx,
            start_y=sy,
            end_x=ex,
            end_y=ey,
            intensity=str(data.get("intensity", "subtle")),
            duration_seconds=float(data.get("duration_seconds", duration_seconds)),
            reasoning=str(data.get("reasoning", "")),
        )
        return plan.validate_and_clamp()
