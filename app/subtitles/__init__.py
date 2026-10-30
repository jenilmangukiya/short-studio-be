"""Subtitles package for Shorts Studio."""

from .base import BaseSubtitleProvider, SubtitleConfigError, SubtitleError
from .factory import get_subtitle_provider
from .types import SubtitleResult, SubtitleSegment, WordTimestamp
from .alignment import (
    alignment_cache_rel_path,
    apply_aligned_timings_to_items,
    compute_aligned_scene_timings,
    compute_alignment_hash,
    load_cached_alignment,
    load_or_compute_alignment,
)

__all__ = [
    "BaseSubtitleProvider",
    "SubtitleConfigError",
    "SubtitleError",
    "SubtitleResult",
    "SubtitleSegment",
    "WordTimestamp",
    "alignment_cache_rel_path",
    "apply_aligned_timings_to_items",
    "compute_aligned_scene_timings",
    "compute_alignment_hash",
    "get_subtitle_provider",
    "load_cached_alignment",
    "load_or_compute_alignment",
]
