"""Subtitles package for Shorts Studio."""

from .base import BaseSubtitleProvider, SubtitleConfigError, SubtitleError
from .factory import get_subtitle_provider
from .styles import (
    DEFAULT_SUBTITLE_STYLE_ID,
    SUBTITLE_STYLES,
    SubtitleStyle,
    get_subtitle_style,
    list_subtitle_styles,
)
from .types import SubtitleResult, SubtitleSegment, WordTimestamp
from .alignment import (
    align_scene_visual_beats,
    alignment_cache_rel_path,
    apply_aligned_beat_timings_to_storyboard,
    apply_aligned_timings_to_items,
    compute_aligned_scene_timings,
    compute_alignment_hash,
    load_cached_alignment,
    load_or_compute_alignment,
)

__all__ = [
    "BaseSubtitleProvider",
    "DEFAULT_SUBTITLE_STYLE_ID",
    "SUBTITLE_STYLES",
    "SubtitleConfigError",
    "SubtitleError",
    "SubtitleResult",
    "SubtitleSegment",
    "SubtitleStyle",
    "WordTimestamp",
    "align_scene_visual_beats",
    "alignment_cache_rel_path",
    "apply_aligned_beat_timings_to_storyboard",
    "apply_aligned_timings_to_items",
    "compute_aligned_scene_timings",
    "compute_alignment_hash",
    "get_subtitle_provider",
    "get_subtitle_style",
    "list_subtitle_styles",
    "load_cached_alignment",
    "load_or_compute_alignment",
]
