"""Prompt composition utilities for Shorts Studio.

Combines scene-specific image prompts, recurring subject anchors, visual styles,
and fixed vertical framing rules into a unified final prompt for image generation.
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
MAX_FINAL_PROMPT_LENGTH: int = 1000


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


def compose_final_prompt(
    scene: Any,
    subject_anchor: str | None = None,
    visual_style: str | None = None,
) -> str:
    """Compose the final image prompt for a scene.

    Assembly order:
    1. Scene image_prompt
    2. Recurring subject_anchor (only if scene.uses_anchor is True and anchor is non-empty)
    3. Visual style description (if provided)
    4. Fixed COMPOSITION_RULES

    Joins segments cleanly with single spaces and no duplicate punctuation.
    The code writes final_prompt; the model never does.
    """
    if isinstance(scene, dict):
        img_prompt = scene.get("image_prompt", "")
        uses_anchor = bool(scene.get("uses_anchor", False))
    else:
        img_prompt = getattr(scene, "image_prompt", "")
        uses_anchor = bool(getattr(scene, "uses_anchor", False))

    segments: list[str] = []

    # 1. Scene image prompt
    cleaned_img = _clean_segment(str(img_prompt or ""))
    if cleaned_img:
        segments.append(cleaned_img)

    # 2. Subject anchor (only if uses_anchor is True)
    if uses_anchor and subject_anchor:
        cleaned_anchor = _clean_segment(str(subject_anchor).strip())
        if cleaned_anchor:
            segments.append(cleaned_anchor)

    # 3. Visual style
    if visual_style:
        cleaned_style = _clean_segment(str(visual_style).strip())
        if cleaned_style:
            segments.append(cleaned_style)

    # 4. Fixed composition rules
    cleaned_rules = _clean_segment(COMPOSITION_RULES)
    if cleaned_rules:
        segments.append(cleaned_rules)

    joined = " ".join(segments)
    # Collapse multiple whitespaces
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

    scenes = target.get("scenes") or []
    for s in scenes:
        s["final_prompt"] = compose_final_prompt(
            scene=s,
            subject_anchor=anchor,
            visual_style=style,
        )

    return storyboard
