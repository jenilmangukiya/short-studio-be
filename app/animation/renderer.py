"""FFmpeg zoompan filter compilation and rendering integration for AI-directed animation."""

from __future__ import annotations

from typing import Any
from .types import DEFAULT_SUBJECT_Y, AnimationPlan


def build_fallback_ken_burns_filter(
    shot_type: str,
    scene_idx: int,
    scene_dur: float,
    target_fps: int = 30,
    target_width: int = 1080,
    target_height: int = 1920,
) -> tuple[str, str, str, str]:
    """Safe fallback producing the restrained Phase 1 Ken Burns effect with smooth easing.

    Used when animation metadata is missing, invalid, or encounters an unexpected error.
    """
    total_frames = max(1, int(round(scene_dur * target_fps)))
    zoom_delta = 0.05
    st = str(shot_type or "medium").lower()

    # Smoothstep Hermite easing: 3p^2 - 2p^3 eliminates mechanical velocity jump at cut boundaries
    ease_ramp = f"(on/{total_frames})*(on/{total_frames})*(3-2*(on/{total_frames}))"

    if st in ("extreme-close-up", "close-up", "macro"):
        z_smooth = f"({1.0 + zoom_delta:.3f}-{zoom_delta:.3f}*{ease_ramp})"
    elif st in ("diagram", "cutaway") and scene_idx % 2 == 0:
        z_smooth = f"({1.0 + zoom_delta:.3f}-{zoom_delta:.3f}*{ease_ramp})"
    else:
        z_smooth = f"(1.0+{zoom_delta:.3f}*{ease_ramp})"

    z_expr = f"(iw/(2*round(iw/(2*{z_smooth}))+0.01))"
    x_expr = "iw/2-(iw/zoom/2)+0.5"
    y_expr = f"clip((ih-trunc(ih/zoom))*{DEFAULT_SUBJECT_Y:.2f}+0.5,0,ih-ih/zoom)"

    intermediate_w = int(target_width * 2)
    intermediate_h = int(target_height * 2)

    filter_chain = (
        f"scale={intermediate_w}:{intermediate_h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={intermediate_w}:{intermediate_h},"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':"
        f"d={total_frames + 2}:s={target_width}x{target_height}:fps={target_fps},"
        f"format=yuv444p"
    )

    return z_expr, x_expr, y_expr, filter_chain


def build_ffmpeg_zoompan_filter(
    plan: AnimationPlan | dict[str, Any] | None,
    scene_dur: float,
    target_fps: int = 30,
    target_width: int = 1080,
    target_height: int = 1920,
    fallback_shot_type: str = "medium",
    fallback_scene_idx: int = 1,
) -> tuple[str, str, str, str, dict[str, Any]]:
    """Compile an AnimationPlan into exact, smooth-eased FFmpeg zoompan filter expressions.

    Guarantees:
    1. Smooth Hermite (smoothstep) easing on zoom and motion so movement does not feel mechanically linear.
    2. Focal anchor preservation: zooms expand directly away from (fx, fy), keeping the subject stable on screen.
    3. Subtitle safe zone preservation: vertical anchor fy stays safely in upper-middle of 9:16 frame.
    4. Zero sub-pixel jitter via 1.5x pre-scaling before zoompan.
    5. Safe fallback to Phase 1 Ken Burns behavior if plan is None or invalid.

    Returns:
        tuple of (z_expr, x_expr, y_expr, filter_chain, resolved_plan_dict).
    """
    # Fallback if plan is None
    if plan is None:
        z, x, y, fc = build_fallback_ken_burns_filter(
            shot_type=fallback_shot_type,
            scene_idx=fallback_scene_idx,
            scene_dur=scene_dur,
            target_fps=target_fps,
            target_width=target_width,
            target_height=target_height,
        )
        fallback_dict = {
            "animation_type": "fallback-ken-burns",
            "focal_coords": [0.5, DEFAULT_SUBJECT_Y],
            "start_scale": 1.0,
            "end_scale": 1.05,
            "reasoning": "Fallback to standard Ken Burns motion.",
        }
        return z, x, y, fc, fallback_dict

    # Coerce to validated AnimationPlan
    if isinstance(plan, dict):
        try:
            valid_plan = AnimationPlan.from_dict(plan, duration_seconds=scene_dur)
        except Exception:
            z, x, y, fc = build_fallback_ken_burns_filter(
                shot_type=fallback_shot_type,
                scene_idx=fallback_scene_idx,
                scene_dur=scene_dur,
                target_fps=target_fps,
                target_width=target_width,
                target_height=target_height,
            )
            return z, x, y, fc, {"animation_type": "fallback-ken-burns"}
    elif isinstance(plan, AnimationPlan):
        valid_plan = plan.validate_and_clamp()
    else:
        z, x, y, fc = build_fallback_ken_burns_filter(
            shot_type=fallback_shot_type,
            scene_idx=fallback_scene_idx,
            scene_dur=scene_dur,
            target_fps=target_fps,
            target_width=target_width,
            target_height=target_height,
        )
        return z, x, y, fc, {"animation_type": "fallback-ken-burns"}

    total_frames = max(1, int(round(scene_dur * target_fps)))
    intermediate_w = int(target_width * 2)
    intermediate_h = int(target_height * 2)

    # Smoothstep easing curve: 3p^2 - 2p^3
    ease_ramp = f"(on/{total_frames})*(on/{total_frames})*(3-2*(on/{total_frames}))"

    # 1. Zoom scale expression z(p)
    z_start = valid_plan.start_scale
    z_end = valid_plan.end_scale
    z_diff = z_end - z_start

    if abs(z_diff) < 0.001 or valid_plan.animation_type == "static":
        if abs(z_start - 1.0) < 0.001:
            z_expr = f"{z_start:.3f}"
        else:
            z_expr = f"(iw/(2*round(iw/(2*{z_start:.3f}))+0.01))"
    else:
        # Smooth Hermite acceleration/deceleration quantized to even crop widths to prevent parity wobble
        z_smooth = f"({z_start:.3f}+({z_diff:+.3f})*{ease_ramp})"
        z_expr = f"(iw/(2*round(iw/(2*{z_smooth}))+0.01))"

    # 2. X and Y coordinate positioning
    fx = valid_plan.focal_x
    fy = valid_plan.focal_y

    if valid_plan.animation_type in ("pan-left", "pan-right", "pan-up", "pan-down", "subtle-drift"):
        # Explicit pan or drift with smooth interpolation inside valid crop bounds [0, iw - trunc(iw/zoom)]
        x_start = valid_plan.start_x
        x_end = valid_plan.end_x
        x_diff = x_end - x_start

        if abs(x_diff) < 0.001:
            if abs(fx - 0.50) < 0.001:
                x_expr = "iw/2-(iw/zoom/2)+0.5"
            else:
                x_expr = f"clip((iw-trunc(iw/zoom))*{fx:.3f}+0.5,0,iw-iw/zoom)"
        else:
            x_pos = f"({x_start:.3f}+({x_diff:+.3f})*{ease_ramp})"
            x_expr = f"clip((iw-trunc(iw/zoom))*{x_pos}+0.5,0,iw-iw/zoom)"

        y_start = valid_plan.start_y
        y_end = valid_plan.end_y
        y_diff = y_end - y_start

        if abs(y_diff) < 0.001:
            y_expr = f"clip((ih-trunc(ih/zoom))*{fy:.3f}+0.5,0,ih-ih/zoom)"
        else:
            y_pos = f"({y_start:.3f}+({y_diff:+.3f})*{ease_ramp})"
            y_expr = f"clip((ih-trunc(ih/zoom))*{y_pos}+0.5,0,ih-ih/zoom)"
    else:
        # Standard Ken Burns (smooth zoom-in, zoom-out, static)
        # Pinned directly on focal anchor (fx, fy) with truncation-matched rounding
        if abs(fx - 0.50) < 0.001:
            x_expr = "iw/2-(iw/zoom/2)+0.5"
        else:
            x_expr = f"clip((iw-trunc(iw/zoom))*{fx:.3f}+0.5,0,iw-iw/zoom)"

        y_expr = f"clip((ih-trunc(ih/zoom))*{fy:.3f}+0.5,0,ih-ih/zoom)"

    filter_chain = (
        f"scale={intermediate_w}:{intermediate_h}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={intermediate_w}:{intermediate_h},"
        f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':"
        f"d={total_frames + 2}:s={target_width}x{target_height}:fps={target_fps},"
        f"format=yuv444p"
    )

    return z_expr, x_expr, y_expr, filter_chain, valid_plan.to_dict()
