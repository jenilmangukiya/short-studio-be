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
import re
from pathlib import Path
from typing import Any

from .base import BaseSubtitleProvider, SubtitleError
from .types import SubtitleResult, WordTimestamp

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


def _normalize_word_token(text: str) -> str:
    """Normalize a word token by stripping punctuation, quotes, brackets and lowercasing."""
    cleaned = re.sub(r"[^\w\s]", "", str(text or "").strip()).lower()
    return cleaned


def _score_token_match(candidate_tokens: list[str], beat_tokens: list[str]) -> float:
    """Calculate match score between candidate word tokens and target beat tokens.

    Uses longest common subsequence (LCS) ratio combined with length difference penalty.
    """
    if not candidate_tokens or not beat_tokens:
        return 0.0

    n_c = len(candidate_tokens)
    n_b = len(beat_tokens)

    dp = [0] * (n_b + 1)
    for c_tok in candidate_tokens:
        curr = [0] * (n_b + 1)
        for j, b_tok in enumerate(beat_tokens, 1):
            if c_tok == b_tok:
                curr[j] = dp[j - 1] + 1
            else:
                curr[j] = max(dp[j], curr[j - 1])
        dp = curr

    lcs_len = dp[n_b]
    len_diff = abs(n_c - n_b)
    score = (2.0 * lcs_len) - (0.5 * len_diff)
    return max(0.0, score)


def _find_optimal_beat_partition(
    norm_word_tokens: list[str],
    beat_token_lists: list[list[str]],
) -> list[tuple[int, int]]:
    """Partition N words among K beats monotonically to maximize sequential word match.

    Returns a list of K (start_word_index, end_word_index) tuples, inclusive, 0-based.
    Guarantees:
      - partitions[0][0] == 0
      - partitions[k][0] == partitions[k-1][1] + 1
      - partitions[-1][1] == len(norm_word_tokens) - 1
      - partitions[k][1] >= partitions[k][0] (when N >= K)
    """
    n = len(norm_word_tokens)
    k = len(beat_token_lists)
    if k <= 1 or n == 0:
        return [(0, max(0, n - 1))] * max(1, k)

    if n < k:
        # More beats than words: clamp so each beat gets an index
        partitions: list[tuple[int, int]] = []
        for b_idx in range(k):
            w_idx = min(b_idx, n - 1)
            partitions.append((w_idx, w_idx))
        return partitions

    has_tokens = any(len(bt) > 0 for bt in beat_token_lists)
    if not has_tokens:
        # Fallback: proportional partition by word count
        step = n / float(k)
        partitions = []
        curr_s = 0
        for b_idx in range(k):
            if b_idx == k - 1:
                curr_e = n - 1
            else:
                curr_e = min(n - (k - b_idx), max(curr_s, int(round((b_idx + 1) * step)) - 1))
            partitions.append((curr_s, curr_e))
            curr_s = curr_e + 1
        return partitions

    # DP optimization: memo[(b_idx, s_w)] -> (max_score, best_end_w)
    memo: dict[tuple[int, int], tuple[float, int]] = {}

    def solve(b_idx: int, s_w: int) -> tuple[float, int]:
        if (b_idx, s_w) in memo:
            return memo[(b_idx, s_w)]

        # Base case: last beat takes all remaining words up to n - 1
        if b_idx == k - 1:
            cand = norm_word_tokens[s_w:n]
            sc = _score_token_match(cand, beat_token_lists[b_idx])
            res = (sc, n - 1)
            memo[(b_idx, s_w)] = res
            return res

        best_score = -float("inf")
        best_end = s_w
        max_possible_end = n - (k - b_idx)
        for cand_end in range(s_w, max_possible_end + 1):
            cand = norm_word_tokens[s_w : cand_end + 1]
            sc_curr = _score_token_match(cand, beat_token_lists[b_idx])
            sc_next, _ = solve(b_idx + 1, cand_end + 1)
            total_sc = sc_curr + sc_next
            if total_sc > best_score:
                best_score = total_sc
                best_end = cand_end

        res = (best_score, best_end)
        memo[(b_idx, s_w)] = res
        return res

    partitions = []
    curr_s = 0
    for b_idx in range(k):
        _, curr_e = solve(b_idx, curr_s)
        partitions.append((curr_s, curr_e))
        curr_s = curr_e + 1

    return partitions


def align_scene_visual_beats(
    scene_beats: list[dict[str, Any]],
    words: list[WordTimestamp],
    scene_start_time: float,
    scene_end_time: float,
    scene_narration: str = "",
) -> list[dict[str, Any]]:
    """Assign precise, non-overlapping, ordered word boundaries and timestamps to a scene's VisualBeats.

    Derives beat cut timestamps from aligned word intervals while respecting natural acoustic pauses.
    Guarantees:
      - First beat starts at scene_start_time
      - Last beat ends at scene_end_time
      - Beat k end_time == Beat k+1 start_time (strictly gapless and non-overlapping)
      - Every beat has end_time > start_time
    """
    if not scene_beats:
        return []

    k = len(scene_beats)
    n = len(words)
    total_dur = max(0.1, scene_end_time - scene_start_time)

    # Convert beats to dictionary representation
    beats_out = [dict(b) if isinstance(b, dict) else b.model_dump() for b in scene_beats]

    if n == 0:
        # No acoustic word timestamps available: fall back to proportional division of scene duration
        step = total_dur / float(k)
        running = scene_start_time
        for i, b in enumerate(beats_out):
            s_t = running
            e_t = scene_end_time if i == k - 1 else round(s_t + step, 3)
            b["start_time"] = round(s_t, 3)
            b["end_time"] = round(e_t, 3)
            b["duration_seconds"] = round(e_t - s_t, 3)
            b["start_word_index"] = 0
            b["end_word_index"] = 0
            b["timing_type"] = "fallback_proportional"
            running = e_t
        return beats_out

    if k == 1:
        # Single beat covers the entire scene
        b = beats_out[0]
        b["start_word_index"] = 0
        b["end_word_index"] = n - 1
        b["start_time"] = round(scene_start_time, 3)
        b["end_time"] = round(scene_end_time, 3)
        b["duration_seconds"] = round(scene_end_time - scene_start_time, 3)
        b["timing_type"] = "aligned"
        if not b.get("narration_text"):
            b["narration_text"] = scene_narration
        return beats_out

    # Multiple beats: extract tokens and compute optimal sequential partition
    norm_words = [_normalize_word_token(w.text) for w in words]
    beat_tokens = [
        [_normalize_word_token(t) for t in str(b.get("narration_text") or "").split() if _normalize_word_token(t)]
        for b in beats_out
    ]

    partitions = _find_optimal_beat_partition(norm_words, beat_tokens)

    # Compute cut points between consecutive beats
    min_beat_dur = 0.08
    cuts = [round(scene_start_time, 3)]

    for i in range(k - 1):
        end_w_idx = partitions[i][1]
        next_w_idx = partitions[i + 1][0]
        w_curr = words[end_w_idx]
        w_next = words[next_w_idx]

        t_end = float(w_curr.end)
        t_start = float(w_next.start)
        gap = t_start - t_end

        is_sentence_end = w_curr.text.rstrip("\"'”’)").endswith(_SENTENCE_END_CHARS)
        if gap > 0.04:
            if is_sentence_end:
                offset = min(gap * 0.65, max(0.0, gap - 0.08))
            else:
                offset = min(gap * 0.55, max(0.0, gap - 0.04))
            raw_cut = t_end + offset
        else:
            # Continuous speech without audible pause: cut right at acoustic phonetic boundary
            raw_cut = t_end

        # Clamp cut to guarantee ordered intervals and minimum beat duration
        min_allowed = cuts[-1] + min_beat_dur
        max_allowed = scene_end_time - (min_beat_dur * (k - 1 - i))
        cut = max(min_allowed, min(raw_cut, max_allowed))
        cuts.append(round(cut, 3))

    cuts.append(round(scene_end_time, 3))

    for i, b in enumerate(beats_out):
        b["start_word_index"] = partitions[i][0]
        b["end_word_index"] = partitions[i][1]
        b["start_time"] = cuts[i]
        b["end_time"] = cuts[i + 1]
        b["duration_seconds"] = round(cuts[i + 1] - cuts[i], 3)
        b["timing_type"] = "aligned"
        if not b.get("narration_text") and n > 0:
            assigned_tokens = [words[w_i].text for w_i in range(partitions[i][0], partitions[i][1] + 1)]
            b["narration_text"] = " ".join(assigned_tokens)

    return beats_out


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
        t_entry: dict[str, Any] = {
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
        if sc.get("visual_beats"):
            t_entry["visual_beats"] = align_scene_visual_beats(
                scene_beats=sc["visual_beats"],
                words=segments[i].words,
                scene_start_time=start_t,
                scene_end_time=end_t,
                scene_narration=str(sc.get("narration") or segments[i].text or ""),
            )
        timings.append(t_entry)
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
                "visual_beats",
            ):
                if key in t:
                    new_it[key] = t[key]
        updated.append(new_it)
    return updated


def apply_aligned_beat_timings_to_storyboard(
    storyboard_dict: dict[str, Any],
    alignment: SubtitleResult,
    total_duration: float,
) -> dict[str, Any]:
    """Update a storyboard dictionary's scenes and visual beats with aligned timestamps.

    Returns the updated storyboard dictionary.
    """
    inner = (
        storyboard_dict.get("storyboard")
        if isinstance(storyboard_dict.get("storyboard"), dict)
        else storyboard_dict
    )
    scenes = inner.get("scenes") if isinstance(inner.get("scenes"), list) else []
    if not scenes or not alignment.segments:
        return storyboard_dict

    scene_timings = compute_aligned_scene_timings(scenes, alignment, total_duration)
    for sc, timing in zip(scenes, scene_timings):
        sc["start_time"] = timing["start_time"]
        sc["end_time"] = timing["end_time"]
        sc["duration_seconds"] = timing["duration_seconds"]
        if "visual_beats" in timing:
            sc["visual_beats"] = timing["visual_beats"]

    return storyboard_dict
