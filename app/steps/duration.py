"""Canonical duration resolution and dynamic word budget calculations for Shorts Studio.

Maintains strict conceptual separation between:
1. Requested duration configuration (video.duration_mode, video.target_duration).
2. Resolved target duration (persisted in Creative Direction output as canonical source).
3. Estimated narration spoken duration (computed from script word count and syllable density).
4. Actual rendered video duration (from continuous audio WAV and rendered MP4 stream).
"""

from __future__ import annotations

import math
from typing import Any

from app.config import settings

# Canonical duration limits for short-form video storytelling
MIN_ALLOWED_DURATION_SECONDS: int = 30
MAX_ALLOWED_DURATION_SECONDS: int = 180
DEFAULT_TARGET_DURATION_SECONDS: int = 60


def clamp_duration_seconds(duration: float | int) -> int:
    """Clamp a duration value to the valid [30, 180] second bounds."""
    try:
        val = int(round(float(duration)))
        return max(MIN_ALLOWED_DURATION_SECONDS, min(MAX_ALLOWED_DURATION_SECONDS, val))
    except (TypeError, ValueError):
        return DEFAULT_TARGET_DURATION_SECONDS


def resolve_video_target_duration(
    video: Any,
    creative_direction_data: dict[str, Any] | None = None,
    default_duration: int | None = None,
    require_resolved: bool = False,
) -> int:
    """Resolve the canonical target duration in seconds for a video.

    Canonical Precedence:
    1. Manual Mode: If video.duration_mode == "manual" and video.target_duration is set,
       the user's explicit duration ALWAYS takes precedence over AI recommendations.
    2. Creative Direction: If Creative Direction has run (passed via creative_direction_data
       from the current pipeline state), extract 'resolved_target_duration' or
       'recommended_duration_seconds'.
    3. Failure/Fallback:
       - If require_resolved=True: Raises PipelineError rather than silently falling back.
       - If require_resolved=False: Uses default_duration (or DEFAULT_TARGET_DURATION_SECONDS).

    Important Architectural Guarantees:
    - No Stale Step Fallback: Never inspects historical video.steps, preventing rerun,
      revision, rollback, or cache invalidations from consuming an outdated duration.
    - No Silent Downstream Fallback: Downstream steps require an explicit or CD-resolved duration.
    - No Legacy Heuristics: Removed legacy target_duration without manual mode and regex parsing.
    """
    duration_mode = getattr(video, "duration_mode", "auto") if video else "auto"
    manual_target = getattr(video, "target_duration", None) if video else None

    # 1. Manual user configuration takes absolute precedence
    if duration_mode == "manual":
        if manual_target is not None:
            return clamp_duration_seconds(manual_target)
        if require_resolved:
            from app.pipeline import PipelineError
            raise PipelineError("Video duration_mode is 'manual' but target_duration is not set.")

    # 2. Current passed creative_direction_data (from current pipeline state)
    if creative_direction_data and isinstance(creative_direction_data, dict):
        cd_inner = creative_direction_data.get("creative_direction")
        if isinstance(cd_inner, dict):
            cd_dict = cd_inner
        else:
            cd_dict = creative_direction_data

        if "resolved_target_duration" in cd_dict and cd_dict["resolved_target_duration"] is not None:
            return clamp_duration_seconds(cd_dict["resolved_target_duration"])
        if "recommended_duration_seconds" in cd_dict and cd_dict["recommended_duration_seconds"] is not None:
            return clamp_duration_seconds(cd_dict["recommended_duration_seconds"])

    # 3. If require_resolved is True, downstream generation CANNOT silently fall back to 60s
    if require_resolved:
        has_cd_pipeline = True
        if video and hasattr(video, "steps") and video.steps:
            has_cd_pipeline = any(getattr(s, "name", None) == "creative_direction" for s in video.steps)

        if has_cd_pipeline:
            from app.pipeline import PipelineError
            raise PipelineError(
                "Downstream generation requires a valid current Creative Direction resolved duration "
                "or manual duration configuration."
            )

    if default_duration is not None:
        return clamp_duration_seconds(default_duration)

    return DEFAULT_TARGET_DURATION_SECONDS


def calculate_word_budget_for_duration(
    target_duration: float | int | tuple[float, float],
    words_per_second: float | None = None,
    speaking_speed: float | None = None,
) -> tuple[int, int]:
    """Calculate dynamic min and max word count targets for a duration budget.

    Accounts for realistic spoken narration rate (~2.3-2.4 words/second),
    configured TTS playback speed, and inter-sentence natural breath pauses
    (~0.35s per sentence boundary).

    Returns:
        tuple of (min_words, max_words)
    """
    if isinstance(target_duration, (list, tuple)) and len(target_duration) >= 2:
        d_mid = (float(target_duration[0]) + float(target_duration[1])) / 2.0
        d_min = float(target_duration[0])
        d_max = float(target_duration[1])
    else:
        d_mid = float(target_duration)
        d_min = max(20.0, d_mid * 0.9)
        d_max = d_mid * 1.1

    base_wps = words_per_second or getattr(settings, "script_speaking_rate_wps", 2.3)
    speed = float(speaking_speed) if speaking_speed and float(speaking_speed) > 0 else 1.0
    wps = base_wps * speed

    # Allow natural breath pauses: roughly 1 pause every ~6 seconds
    breaks_min = max(0.0, (d_min / 6.0) - 1.0)
    breaks_max = max(0.0, (d_max / 6.0) - 1.0)
    net_speech_min = max(10.0, d_min - (breaks_min * 0.35))
    net_speech_max = max(15.0, d_max - (breaks_max * 0.35))

    raw_min = net_speech_min * wps
    raw_max = net_speech_max * wps

    min_words = max(35, int(math.floor(raw_min * 0.90)))
    max_words = max(min_words + 20, int(math.ceil(raw_max * 1.10)))
    return (min_words, max_words)
