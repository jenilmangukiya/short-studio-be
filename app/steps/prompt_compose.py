"""Prompt composition utilities for Shorts Studio.

Combines scene-specific image prompts, recurring subject anchors, visual styles,
environment settings, camera directions, focal points, and fixed vertical framing rules
into a unified final prompt for image generation.
"""

from __future__ import annotations

import re
from typing import Any

# Fixed composition and framing rules appended to every final prompt
COMPOSITION_RULES: str = (
    "Vertical 9:16, main subject in the upper-middle of the frame, "
    "calm uncluttered lower third, no text, no letters, no logos, no watermark."
)

# Maximum sensible length for a code-assembled final prompt
MAX_FINAL_PROMPT_LENGTH: int = 1500


def _clean_segment(text: str) -> str:
    """Normalize whitespace and ensure clean sentence-ending punctuation."""
    t = text.strip()
    if not t:
        return ""
    # Strip any trailing commas, semicolons, colons, or whitespace
    t = re.sub(r"[,;:\s]+$", "", t)
    # Deduplicate multiple consecutive trailing periods (e.g. '..' -> '.')
    t = re.sub(r"\.+$", ".", t)
    # If not ending in terminal punctuation (. ! ?), add a period
    if not re.search(r"[.!?]$", t):
        t += "."
    return t


def adapt_visual_style(
    visual_style: str | None,
    environment: str = "",
    shot_type: str = "",
    visual_technique: str = "",
) -> str:
    """Adapt the overall visual style to the specific scene context.

    Prevents generic studio pastel backdrops or shallow blur from overriding
    specific environments (e.g. data centers, space, aircraft cabins, landscapes)
    or technical diagram / wide framing shots.
    """
    if not visual_style:
        return ""

    style_clean = visual_style.strip()
    if not style_clean:
        return ""

    has_specific_env = bool(environment.strip()) or any(
        w in environment.lower()
        for w in (
            "server",
            "data center",
            "cloud",
            "sky",
            "space",
            "cabin",
            "interior",
            "exterior",
            "landscape",
            "ocean",
            "room",
            "hall",
            "facility",
            "factory",
            "chassis",
            "rack",
        )
    )
    is_wide_or_technical = shot_type.lower() in (
        "wide",
        "aerial",
        "exterior",
        "cutaway",
        "cross-section",
        "diagram",
        "top-down",
        "in-context",
    ) or any(
        t in visual_technique.lower()
        for t in ("diagram", "cutaway", "schematic", "technical", "cross-section", "macro-photo")
    )

    # Split by comma to evaluate individual style descriptors
    parts = [p.strip() for p in style_clean.split(",") if p.strip()]
    kept_parts: list[str] = []

    for part in parts:
        low = part.lower()
        # Remove generic pastel/gradient backdrop when a real environment or technical shot is present
        if (has_specific_env or is_wide_or_technical) and (
            "pastel gradient" in low or "gradient background" in low or "simple pastel" in low
        ):
            continue
        # Remove shallow depth of field on wide vistas, architectural halls, or exploded diagrams
        if is_wide_or_technical and "shallow depth of field" in low:
            continue
        kept_parts.append(part)

    if not kept_parts:
        return style_clean

    return ", ".join(kept_parts)


def compose_final_prompt(
    scene: Any,
    subject_anchor: str | None = None,
    visual_style: str | None = None,
    *,
    environment_anchor: str | None = None,
    video_topic: str | None = None,
    video_title: str | None = None,
    prev_scene: Any | None = None,
    next_scene: Any | None = None,
) -> str:
    """Compose the final image prompt for a scene with rich visual context.

    Assembly structure:
    1. Scene subject & visual description (from scene.image_prompt)
    2. Recurring subject anchor (when scene.uses_anchor is True)
    3. Scene-specific environment and background (from scene.environment or environment_anchor)
    4. Narration and visual concept context (from scene.visual_purpose or scene.narration)
    5. Camera framing and perspective direction (from scene.shot_type and scene.camera_angle)
    6. Primary visual focal point (from scene.focal_point)
    7. Visual technique (from scene.visual_technique)
    8. Scene-adapted visual style (ensuring background responsiveness)
    9. Fixed vertical framing rules and negative constraints (COMPOSITION_RULES)

    The code writes final_prompt; the model never does.
    """
    if isinstance(scene, dict):
        img_prompt = scene.get("image_prompt", "")
        uses_anchor = bool(scene.get("uses_anchor", False))
        shot_type = str(scene.get("shot_type", "") or "").strip()
        camera_angle = str(scene.get("camera_angle", "") or "").strip()
        focal_point = str(scene.get("focal_point", "") or "").strip()
        visual_technique = str(scene.get("visual_technique", "") or "").strip()
        visual_purpose = str(scene.get("visual_purpose", "") or "").strip()
        narration = str(scene.get("narration", "") or "").strip()
        environment = str(scene.get("environment", "") or "").strip()
    else:
        img_prompt = getattr(scene, "image_prompt", "")
        uses_anchor = bool(getattr(scene, "uses_anchor", False))
        shot_type = str(getattr(scene, "shot_type", "") or "").strip()
        camera_angle = str(getattr(scene, "camera_angle", "") or "").strip()
        focal_point = str(getattr(scene, "focal_point", "") or "").strip()
        visual_technique = str(getattr(scene, "visual_technique", "") or "").strip()
        visual_purpose = str(getattr(scene, "visual_purpose", "") or "").strip()
        narration = str(getattr(scene, "narration", "") or "").strip()
        environment = str(getattr(scene, "environment", "") or "").strip()

    segments: list[str] = []

    # 1. Scene image prompt (Subject & primary action)
    cleaned_img = _clean_segment(str(img_prompt or ""))
    if cleaned_img:
        segments.append(cleaned_img)

    # 2. Subject anchor (only if uses_anchor is True and anchor is provided)
    if uses_anchor and subject_anchor:
        cleaned_anchor = _clean_segment(str(subject_anchor).strip())
        if cleaned_anchor:
            segments.append(cleaned_anchor)

    # 3. Environment & Background Setting
    effective_env = environment or (environment_anchor or "").strip()
    if effective_env:
        # Avoid redundant phrasing if the environment is already fully in cleaned_img
        if effective_env.lower() not in cleaned_img.lower():
            prefix = "Environment" if environment else "Environment setting"
            segments.append(_clean_segment(f"{prefix}: {effective_env}"))

    # 4. Narration and Visual Concept Context
    if visual_purpose:
        if visual_purpose.lower() not in cleaned_img.lower():
            segments.append(_clean_segment(f"Visual concept: {visual_purpose}"))
    elif narration and len(narration.split()) >= 3:
        if narration.lower() not in cleaned_img.lower():
            segments.append(_clean_segment(f"Narration context: {narration}"))

    # 5. Camera framing and perspective direction (shot_type, camera_angle)
    camera_parts: list[str] = []
    if shot_type and shot_type.lower() not in ("default", ""):
        camera_parts.append(f"{shot_type} shot")
    if camera_angle and camera_angle.lower() not in ("eye-level", "default", ""):
        angle_clean = camera_angle.strip()
        if not angle_clean.lower().endswith(("angle", "perspective", "view")):
            angle_clean += " angle"
        camera_parts.append(angle_clean)
    if camera_parts:
        segments.append(_clean_segment(f"Camera direction: {', '.join(camera_parts)}"))

    # 6. Primary visual focal point
    if focal_point and focal_point.lower() not in ("default", ""):
        if focal_point.lower() not in cleaned_img.lower():
            segments.append(_clean_segment(f"Primary focal point: {focal_point}"))

    # 7. Visual technique
    if visual_technique and visual_technique.lower() not in ("cinematic-photo", "default", ""):
        if visual_technique.lower() not in cleaned_img.lower():
            segments.append(_clean_segment(f"Visual technique: {visual_technique}"))

    # 8. Adaptive Visual Style
    if visual_style:
        adapted_style = adapt_visual_style(
            visual_style=visual_style,
            environment=effective_env,
            shot_type=shot_type,
            visual_technique=visual_technique,
        )
        cleaned_style = _clean_segment(adapted_style)
        if cleaned_style:
            segments.append(cleaned_style)

    # 9. Fixed Composition Rules & Negative Constraints
    cleaned_rules = _clean_segment(COMPOSITION_RULES)
    if cleaned_rules:
        segments.append(cleaned_rules)

    joined = " ".join(segments)
    return re.sub(r"\s+", " ", joined).strip()


def recompute_final_prompts(
    storyboard: dict[str, Any],
    visual_style: str | None = None,
) -> dict[str, Any]:
    """Rebuild every scene's final_prompt from its component parts.

    Can be invoked on the root step output (e.g. step.output) or directly
    on the inner storyboard dict. Updates scenes in-place and returns the dict.
    """
    target = (
        storyboard.get("storyboard")
        if isinstance(storyboard.get("storyboard"), dict)
        else storyboard
    )

    style = visual_style or target.get("visual_style") or ""
    anchor = target.get("subject_anchor") or ""
    env_anchor = target.get("environment_anchor") or ""

    scenes = target.get("scenes") or []
    for idx, s in enumerate(scenes):
        prev_s = scenes[idx - 1] if idx > 0 else None
        next_s = scenes[idx + 1] if idx + 1 < len(scenes) else None
        s["final_prompt"] = compose_final_prompt(
            scene=s,
            subject_anchor=anchor,
            visual_style=style,
            environment_anchor=env_anchor,
            prev_scene=prev_s,
            next_scene=next_s,
        )

    return storyboard
