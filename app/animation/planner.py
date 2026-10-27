"""Intelligent, conservative animation planning engine based on scene composition and semantics.

Defaults to smooth, composition-anchored Ken Burns zoom, reserving static holds for diagrams
and pull-outs for close-ups, while avoiding unnatural sideways panning.
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .analyzer import analyze_image_composition
from .types import (
    DEFAULT_SUBJECT_Y,
    SUBTITLE_SAFETY_MAX_Y,
    AnimationPlan,
    AnimationType,
)


def _infer_semantic_focal_point(scene_data: dict[str, Any]) -> tuple[float, float]:
    """Infer subtle focal offset from textual descriptors if image is unavailable."""
    fp_text = str(scene_data.get("focal_point") or "").lower()
    prompt_text = str(scene_data.get("image_prompt") or "").lower()
    angle_text = str(scene_data.get("camera_angle") or "").lower()
    combined = f"{fp_text} {prompt_text} {angle_text}"

    # Default horizontal center
    x = 0.50
    # Default vertical anchor in upper-middle region
    y = DEFAULT_SUBJECT_Y

    # Subtle horizontal positioning cues (kept restrained within [0.46, 0.54])
    if re.search(r"\b(left|left-side|left-hand)\b", combined):
        x = 0.47
    elif re.search(r"\b(right|right-side|right-hand)\b", combined):
        x = 0.53

    # Subtle vertical positioning cues (kept within [0.38, 0.46])
    if re.search(r"\b(upper|top|high|above|ceiling)\b", combined):
        y = 0.38
    elif re.search(r"\b(bottom|lower|ground|floor)\b", combined):
        y = 0.46

    return x, y


def plan_scene_animation(
    scene_data: dict[str, Any],
    image_path: Path | str | None = None,
    duration_seconds: float = 2.5,
    prev_anim_type: str | None = None,
) -> AnimationPlan:
    """Create a restrained, AI-directed animation plan customized for a scene's composition.

    Rules & Priorities:
    1. Smooth zoom is the default (Phase 1 Ken Burns restored & refined).
    2. Close-ups / macros use gentle pull-out (zoom-out) so edge details are not cropped.
    3. Diagrams, cutaways, and schematics use static hold (or subtle zoom) for maximum readability.
    4. Wide, medium, and atmospheric shots use gentle slow push-in (zoom-in) anchored on the subject.
    5. Lateral or vertical pans are NEVER selected automatically by heuristic; they are only
       permitted if explicitly requested with clear rationale in upstream storyboard metadata.
    6. Natural repetition across adjacent scenes is preserved (no jarring forced variety).

    Args:
        scene_data: Dictionary containing scene metadata (narration, shot_type, focal_point, etc.).
        image_path: Optional path to the rendered scene image on disk.
        duration_seconds: Duration of the audio/scene in seconds.
        prev_anim_type: Animation type used in the previous scene (informational).

    Returns:
        Validated, coordinate-clamped AnimationPlan instance.
    """
    # 1. If explicit animation metadata is already present in scene_data, validate and calibrate it
    raw_anim = scene_data.get("animation")
    if isinstance(raw_anim, dict) and raw_anim.get("animation_type"):
        existing_plan = AnimationPlan.from_dict(raw_anim, duration_seconds=duration_seconds)
        # If image file exists, verify/refine focal coordinates with real image centroid
        if image_path and Path(image_path).is_file():
            img_metrics = analyze_image_composition(image_path)
            # Refine anchor subtly toward visual centroid if close to center
            cx = img_metrics.get("centroid_x", 0.5)
            cy = img_metrics.get("centroid_y", DEFAULT_SUBJECT_Y)
            if existing_plan.focal_x == 0.5 and existing_plan.focal_y == DEFAULT_SUBJECT_Y:
                existing_plan.focal_x = max(0.44, min(0.56, float(cx)))
                existing_plan.focal_y = max(0.36, min(0.48, float(cy)))
                existing_plan.start_x = existing_plan.focal_x
                existing_plan.start_y = existing_plan.focal_y
                existing_plan.end_x = existing_plan.focal_x
                existing_plan.end_y = existing_plan.focal_y
        return existing_plan.validate_and_clamp()

    # 2. Extract semantic cues
    shot_type = str(scene_data.get("shot_type") or "medium").lower().strip()
    focal_desc = str(scene_data.get("focal_point") or "").strip()
    prompt = str(scene_data.get("image_prompt") or "").lower()
    visual_purpose = str(scene_data.get("visual_purpose") or "").lower()
    combined_desc = f"{shot_type} {focal_desc} {prompt} {visual_purpose}"

    # 3. Analyze real image composition if present
    img_analysis = analyze_image_composition(image_path)
    centroid_x = img_analysis.get("centroid_x", 0.5)
    centroid_y = img_analysis.get("centroid_y", DEFAULT_SUBJECT_Y)
    comp_type = img_analysis.get("composition_type", "balanced_scene")

    # Confident focal point determination:
    # Anchor must be restrained near center of vertical 9:16 frame to prevent awkward framing
    if image_path and Path(image_path).is_file() and comp_type != "diffuse_low_contrast":
        fx = max(0.45, min(0.55, float(centroid_x)))
        fy = max(0.38, min(0.46, float(centroid_y)))
    else:
        # Inferred from semantic text or default Phase 1 anchor
        sx, sy = _infer_semantic_focal_point(scene_data)
        fx = max(0.45, min(0.55, sx))
        fy = max(0.38, min(0.46, sy))

    # 4. Conservative Animation Selection
    anim_type: AnimationType = "zoom-in"
    start_scale = 1.00
    end_scale = 1.05
    start_x, start_y = fx, fy
    end_x, end_y = fx, fy
    intensity = "subtle"
    reasoning = ""

    # Rule A: Extreme close-ups & macro photography
    # Gentle pull-out to reveal surrounding context without cropping edges
    if shot_type in ("extreme-close-up", "close-up", "macro") or re.search(r"\b(macro|extreme close|close up|close-up)\b", combined_desc):
        anim_type = "zoom-out"
        start_scale = 1.05
        end_scale = 1.00
        start_x, start_y = fx, fy
        end_x, end_y = fx, fy
        intensity = "moderate"
        reasoning = f"Gentle pull-out from close-up '{focal_desc or 'subject'}' reveals surrounding detail without cropping fine edges."

    # Rule B: Technical schematics, cutaways, diagrams, and blueprints
    # Static hold ensures total legibility of lines, labels, and text
    elif (
        shot_type in ("diagram", "cutaway", "cross-section")
        or re.search(r"\b(diagram|schematic|cutaway|cross-section|chart|infographic|blueprint|flowchart)\b", combined_desc)
    ):
        anim_type = "static"
        start_scale = 1.00
        end_scale = 1.00
        start_x, end_x = fx, fx
        start_y, end_y = fy, fy
        intensity = "restrained"
        reasoning = "Static hold maintains crisp legibility of technical details and diagram components."

    # Rule C: All other shots (wide, medium, establishing, aerial, in-context, POV, top-down)
    # The default: Smooth, gentle push-in anchored on the focal subject
    else:
        anim_type = "zoom-in"
        start_scale = 1.00
        end_scale = 1.05
        start_x, start_y = fx, fy
        end_x, end_y = fx, fy
        intensity = "subtle"
        reasoning = f"Gentle slow push-in toward '{focal_desc or 'primary subject'}' draws viewer focus naturally while preserving composition."

    plan = AnimationPlan(
        animation_type=anim_type,
        focal_point=focal_desc,
        focal_x=round(fx, 3),
        focal_y=round(fy, 3),
        start_scale=start_scale,
        end_scale=end_scale,
        start_x=round(start_x, 3),
        start_y=round(start_y, 3),
        end_x=round(end_x, 3),
        end_y=round(end_y, 3),
        intensity=intensity,
        duration_seconds=duration_seconds,
        reasoning=reasoning,
    )

    return plan.validate_and_clamp()
