"""Shared narration alignment cache and scene-boundary mapping.

Word-level forced alignment (Vosk) is computed once against the continuous
narration WAV, persisted to ``storage/<video_id>/audio/alignment_<hash12>.json``
and reused by the voiceover, render, and subtitles steps.

``compute_aligned_scene_timings`` converts word timestamps into gapless,
non-overlapping visual scene intervals whose cuts land inside the natural
pause between the last word of one scene and the first word of the next.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .base import BaseSubtitleProvider, SubtitleError
from .types import SubtitleResult

# Bump only when the cached alignment payload format changes. Must stay in sync with the
# subtitles step so the cache key written by voiceover/render is a cache hit for subtitles.
ALIGNMENT_CONFIG_VERSION = "v2_caption_grouping"

_SENTENCE_END_CHARS = (".", "!", "?", "…")


def compute_alignment_hash(
    sorted_scenes: list[dict[str, Any]],
    narration_script: str | None,
    continuous_audio_abs: Path | None,
    audio_hash: str | None,
    provider: BaseSubtitleProvider,
) -> str:
    """Compute a stable SHA-256 hash of narration text, audio identity, and aligner configuration."""
    scene_texts = "|".join(
        f"{s.get('index', i)}:{str(s.get('narration') or '').strip()}"
        for i, s in enumerate(sorted_scenes, 1)
    )
    script_part = (narration_script or scene_texts).strip()

    audio_identity = audio_hash or ""
    if continuous_audio_abs and continuous_audio_abs.is_file():
        stat = continuous_audio_abs.stat()
        audio_identity = f"{audio_identity}:{continuous_audio_abs.name}:{stat.st_size}"

    config_part = f"{provider.provider_name}:{provider.model}:{ALIGNMENT_CONFIG_VERSION}"
    payload = f"{script_part}||{audio_identity}||{config_part}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def alignment_cache_rel_path(video_id: Any, alignment_hash: str) -> str:
    """Relative storage path of the persisted alignment for a given hash."""
    return f"{video_id}/audio/alignment_{alignment_hash[:12]}.json"


def load_cached_alignment(cache_abs_path: Path, alignment_hash: str) -> SubtitleResult | None:
    """Load a persisted alignment if it exists and matches the expected hash."""
    if not cache_abs_path.is_file():
        return None
    try:
        data = json.loads(cache_abs_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("alignment_hash") != alignment_hash or not data.get("segments"):
        return None
    try:
        return SubtitleResult.from_dict(data, reused=True)
    except Exception:
        return None


def load_or_compute_alignment(
    *,
    video_id: Any,
    storage_root: Path,
    sorted_scenes: list[dict[str, Any]],
    narration_script: str | None,
    continuous_audio_abs: Path,
    audio_hash: str | None,
    provider: BaseSubtitleProvider,
) -> tuple[SubtitleResult, str, bool]:
    """Return (alignment, alignment_hash, reused) using the on-disk cache when possible.

    Runs the provider (Vosk forced alignment) only on a cache miss and persists the result.
    Raises SubtitleError if alignment cannot be produced.
    """
    alignment_hash = compute_alignment_hash(
        sorted_scenes=sorted_scenes,
        narration_script=narration_script,
        continuous_audio_abs=continuous_audio_abs,
        audio_hash=audio_hash,
        provider=provider,
    )
    cache_abs_path = storage_root / alignment_cache_rel_path(video_id, alignment_hash)

    cached = load_cached_alignment(cache_abs_path, alignment_hash)
    if cached is not None:
        return cached, alignment_hash, True

    try:
        result = provider.generate(
            scenes=sorted_scenes,
            storage_root=storage_root,
            continuous_audio_path=continuous_audio_abs,
            narration_script=narration_script,
        )
    except SubtitleError:
        raise
    except Exception as exc:
        raise SubtitleError(f"Narration alignment failed: {exc}") from exc

    result.alignment_hash = alignment_hash
    result.reused = False

    try:
        cache_abs_path.parent.mkdir(parents=True, exist_ok=True)
        cache_abs_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    except Exception:
        pass

    return result, alignment_hash, False


def _cut_in_pause(prev_speech_end: float, next_speech_start: float, sentence_boundary: bool) -> float:
    """Place a visual cut inside the silence between two scenes' narration.

    The cut lands slightly before the next scene's first word so the new image is on screen
    when the narrator starts describing it, without clipping the previous scene's last word.
    """
    gap = next_speech_start - prev_speech_end
    if gap <= 0.0:
        # Words touch acoustically (aligned words are monotonic, so no true overlap):
        # cut exactly where the previous word ends.
        return prev_speech_end
    if sentence_boundary:
        offset = min(gap * 0.65, max(0.0, gap - 0.08))
    else:
        offset = min(gap * 0.55, max(0.0, gap - 0.05))
    return prev_speech_end + offset


def compute_aligned_scene_timings(
    scenes: list[dict[str, Any]],
    alignment: SubtitleResult,
    total_duration: float,
    min_scene_seconds: float = 0.5,
) -> list[dict[str, Any]]:
    """Map scenes to visual intervals using actual narration word timestamps.

    Guarantees: first scene starts at 0.0, every ``end_time`` equals the next ``start_time``
    (no gaps/overlaps), the last scene ends at ``total_duration``, and every cut falls in the
    pause between scenes (never inside a spoken word).

    Raises SubtitleError if the alignment does not cover every scene with words.
    """
    if not scenes:
        return []

    total = round(float(total_duration if total_duration > 0 else alignment.total_duration_seconds), 3)
    if total <= 0.0:
        raise SubtitleError("Cannot compute aligned scene timings for zero-length narration")

    segments = list(alignment.segments)
    if len(segments) != len(scenes):
        raise SubtitleError(
            f"Alignment has {len(segments)} segments but storyboard has {len(scenes)} scenes"
        )

    speech: list[tuple[float, float]] = []
    for seg in segments:
        if not seg.words:
            raise SubtitleError(f"Alignment segment {seg.index} has no word timestamps")
        speech.append((float(seg.words[0].start), float(seg.words[-1].end)))

    # Visual cut points between consecutive scenes.
    cuts: list[float] = [0.0]
    for i in range(len(scenes) - 1):
        narration = str(scenes[i].get("narration") or segments[i].text or "").strip()
        sentence_boundary = narration.rstrip("\"'”’)").endswith(_SENTENCE_END_CHARS)
        cut = _cut_in_pause(speech[i][1], speech[i + 1][0], sentence_boundary)
        # Keep cuts monotonic and leave room for a minimum on-screen duration.
        cut = max(cut, cuts[-1] + min_scene_seconds)
        cut = min(cut, total - min_scene_seconds * (len(scenes) - 1 - i))
        cuts.append(round(max(cuts[-1], cut), 3))
    cuts.append(total)

    timings: list[dict[str, Any]] = []
    for i, sc in enumerate(scenes):
        start_t = cuts[i]
        end_t = cuts[i + 1]
        scene_idx = int(sc.get("index", i + 1))
        timings.append(
            {
                "index": scene_idx,
                "scene_id": sc.get("id") or sc.get("scene_id") or scene_idx,
                "narration": str(sc.get("narration") or "").strip(),
                "start_time": round(start_t, 3),
                "end_time": round(end_t, 3),
                "duration_seconds": round(end_t - start_t, 3),
                "speech_start_time": round(speech[i][0], 3),
                "speech_end_time": round(speech[i][1], 3),
                "timing_type": "aligned",
            }
        )
    return timings


def apply_aligned_timings_to_items(
    items: list[dict[str, Any]], timings: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Return copies of voiceover items updated with aligned timing fields (matched by index)."""
    by_idx = {t["index"]: t for t in timings}
    updated: list[dict[str, Any]] = []
    for it in items:
        new_it = dict(it)
        t = by_idx.get(it.get("index"))
        if t is not None:
            for key in (
                "start_time",
                "end_time",
                "duration_seconds",
                "speech_start_time",
                "speech_end_time",
                "timing_type",
            ):
                new_it[key] = t[key]
        updated.append(new_it)
    return updated
