"""Continuous narration generation, text-aware chunking, and scene mapping for Shorts Studio."""

from __future__ import annotations

import io
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import soundfile as sf

from app.config import settings

from .base import BaseVoiceoverProvider
from .exceptions import VoiceoverError, VoiceoverProviderError
from .types import VoiceoverResult


def _compute_word_weights(tokens: list[str]) -> list[float]:
    """Estimate relative spoken duration weight for each word based on syllables and characters."""
    weights: list[float] = []
    for token in tokens:
        clean = re.sub(r"[^a-zA-Z0-9]", "", token.lower())
        if not clean:
            weights.append(1.0)
            continue
        # Count vowel groups as phonetic syllable proxies
        vowels = len(re.findall(r"[aeiouy]+", clean))
        syllables = max(1, vowels)
        weight = syllables * 1.6 + len(clean) * 0.25
        weights.append(weight)
    return weights


def collect_narration_script(
    scenes: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Collect narration text from all scenes in their intended order.

    Validates that each scene contains non-empty narration text and constructs
    a unified continuous narration script preserving exact wording and ordering.

    Args:
        scenes: List of scene dictionaries from the storyboard step.

    Returns:
        Tuple of (full_narration_script, validated_scenes_sorted)

    Raises:
        ValueError: If scenes list is empty or any scene has empty narration.
    """
    if not scenes or not isinstance(scenes, list):
        raise ValueError("Storyboard contains no scenes for voiceover generation")

    sorted_scenes = sorted(scenes, key=lambda s: int(s.get("index", 0)))
    validated_scenes: list[dict[str, Any]] = []

    for i, sc in enumerate(sorted_scenes):
        scene_idx = int(sc.get("index", i + 1))
        narration = str(sc.get("narration") or "").strip()
        if not narration:
            raise ValueError(f"Scene {scene_idx} narration text is empty or missing")

        sc_entry: dict[str, Any] = {
            "index": scene_idx,
            "id": sc.get("id") or sc.get("scene_id") or scene_idx,
            "narration": narration,
            "est_seconds": sc.get("est_seconds"),
        }
        if "visual_beats" in sc:
            sc_entry["visual_beats"] = sc["visual_beats"]
        validated_scenes.append(sc_entry)

    full_script = " ".join(s["narration"] for s in validated_scenes).strip()
    if not full_script:
        raise ValueError("Constructed narration script is empty")

    return full_script, validated_scenes


# Sentence split pattern that avoids splitting on common abbreviations (vs., Dr., Mr., etc.) or decimals (2.5)
_SENTENCE_SPLIT_RE = re.compile(
    r"(?<!\bvs)(?<!\bDr)(?<!\bMr)(?<!\bMs)(?<!\bSt)(?<!\bJr)(?<!\bSr)(?<=[.!?])\s+(?=[A-Z0-9\"'“‘])"
)


def _partition_units_balanced(
    units: list[str], max_words: int, max_chars: int
) -> list[str]:
    """Partition sentence/clause units into the minimum number of chunks with balanced lengths.

    Prevents tiny trailing orphan chunks (which shift Kokoro's length-indexed voice style
    embedding `voice[len(tokens)]` and alter vocal timbre) while strictly enforcing
    `words <= max_words` and `chars <= max_chars` for every multi-unit chunk.
    """
    n = len(units)
    if n <= 1:
        return list(units)

    unit_words = [len(u.split()) for u in units]
    unit_chars = [len(u) for u in units]
    total_words = sum(unit_words)

    # First check if everything fits in a single chunk
    total_chars = sum(unit_chars) + (n - 1)
    if total_words <= max_words and total_chars <= max_chars:
        return [" ".join(units)]

    # Dynamic programming over unit indices:
    # dp[i] = (num_chunks, sum_sq_deviation, prev_idx) for prefix units[:i]
    # To compute deviation from a target chunk size, estimate ideal words per chunk:
    min_possible_chunks = max(
        2,
        (total_words + max(1, max_words) - 1) // max(1, max_words),
        (total_chars + max(1, max_chars) - 1) // max(1, max_chars),
    )
    ideal_words = total_words / float(min_possible_chunks)

    inf_tuple = (10**9, float("inf"), -1)
    dp: list[tuple[int, float, int]] = [inf_tuple] * (n + 1)
    dp[0] = (0, 0.0, -1)

    for i in range(1, n + 1):
        w_acc = 0
        c_acc = 0
        for j in range(i - 1, -1, -1):
            w_acc += unit_words[j]
            c_acc += unit_chars[j] + (1 if j < i - 1 else 0)

            # A single unit (j == i - 1) is always allowed even if it slightly exceeds limits;
            # multi-unit groups (j < i - 1) must strictly satisfy max_words and max_chars.
            if j < i - 1 and (w_acc > max_words or c_acc > max_chars):
                break

            prev_chunks, prev_cost, _ = dp[j]
            if prev_chunks >= 10**9:
                continue

            dev = (w_acc - ideal_words) ** 2
            # Extra penalty for very short orphan chunks (< 14 words) when max_words >= 30
            if max_words >= 30 and w_acc < 14 and n > 2:
                dev += 250.0

            cand = (prev_chunks + 1, prev_cost + dev, j)
            if cand[0] < dp[i][0] or (cand[0] == dp[i][0] and cand[1] < dp[i][1]):
                dp[i] = cand

    # Reconstruct balanced chunks from dp table
    result: list[str] = []
    curr = n
    while curr > 0:
        _, _, prev = dp[curr]
        if prev < 0:
            prev = 0
        result.append(" ".join(units[prev:curr]))
        curr = prev
    result.reverse()
    return result


def split_script_into_chunks(
    text: str,
    max_words: int = 65,
    max_chars: int = 380,
    isolate_hook: bool = False,
    hook_text: str | None = None,
) -> list[str]:
    """Split narration script into balanced linguistic chunks at sentence or paragraph boundaries.

    When isolate_hook is True, isolates the opening hook sentence/clause as the very first
    chunk so it can receive an intentional dramatic breath pause, and preserves the balanced
    chunking strategy for all remaining narration.

    Ensures words are never split mid-word, sentence order and punctuation are preserved,
    no words are duplicated or omitted, and multi-chunk splits are balanced to keep
    each chunk well below Kokoro's 510-phoneme window without tiny orphan chunks.

    Args:
        text: Full narration script text.
        max_words: Maximum target spoken words per chunk (default: 65).
        max_chars: Maximum target characters per chunk (default: 380).
        isolate_hook: Whether to isolate the first sentence/hook as its own chunk.
        hook_text: Optional explicit opening hook text to isolate.

    Returns:
        List of cleanly chunked narration text strings.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

    if isolate_hook:
        candidate_hook: str | None = None
        if hook_text and isinstance(hook_text, str) and hook_text.strip():
            h_candidate = hook_text.strip()
            if cleaned_text.startswith(h_candidate):
                candidate_hook = h_candidate

        if not candidate_hook:
            sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(cleaned_text) if s.strip()]
            if len(sentences) <= 1:
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", cleaned_text) if s.strip()]
            if len(sentences) > 1:
                candidate_hook = sentences[0]

        if candidate_hook and len(candidate_hook) < len(cleaned_text):
            remainder = cleaned_text[len(candidate_hook) :].strip()
            if remainder:
                if len(candidate_hook.split()) > max_words or len(candidate_hook) > max_chars:
                    hook_chunks = split_script_into_chunks(
                        candidate_hook, max_words=max_words, max_chars=max_chars, isolate_hook=False
                    )
                else:
                    hook_chunks = [candidate_hook]
                remaining_chunks = split_script_into_chunks(
                    remainder, max_words=max_words, max_chars=max_chars, isolate_hook=False
                )
                return hook_chunks + remaining_chunks

    words = cleaned_text.split()
    if len(words) <= max_words and len(cleaned_text) <= max_chars and "\n\n" not in cleaned_text:
        return [cleaned_text]

    # Divide by paragraphs first
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n|\r\n\s*\r\n", cleaned_text) if p.strip()]

    chunks: list[str] = []

    for para in paragraphs:
        para_words = para.split()
        if len(para_words) <= max_words and len(para) <= max_chars:
            chunks.append(para)
            continue

        # Split paragraph into sentences preserving punctuation (. ! ?)
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(para) if s.strip()]
        if len(sentences) <= 1:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]

        # Expand any single sentence that exceeds max_words or max_chars on its own into clause units
        units: list[str] = []
        for sentence in sentences:
            s_words = sentence.split()
            if len(s_words) <= max_words and len(sentence) <= max_chars:
                units.append(sentence)
                continue

            clauses = [c.strip() for c in re.split(r"(?<=[;:,—–])\s+", sentence) if c.strip()]
            curr_clause_parts: list[str] = []
            curr_c_words = 0
            curr_c_chars = 0
            for clause in clauses:
                c_len = len(clause.split())
                c_chars = len(clause)
                if (
                    curr_clause_parts
                    and (curr_c_words + c_len > max_words or curr_c_chars + 1 + c_chars > max_chars)
                ):
                    units.append(" ".join(curr_clause_parts))
                    curr_clause_parts = [clause]
                    curr_c_words = c_len
                    curr_c_chars = c_chars
                else:
                    curr_clause_parts.append(clause)
                    curr_c_words += c_len
                    curr_c_chars += c_chars + (1 if len(curr_clause_parts) > 1 else 0)
            if curr_clause_parts:
                units.append(" ".join(curr_clause_parts))

        para_chunks = _partition_units_balanced(units, max_words=max_words, max_chars=max_chars)
        chunks.extend(para_chunks)

    return chunks


def compute_inter_chunk_gaps(
    chunks: list[str],
    sentence_gap_seconds: float = 0.24,
    clause_gap_seconds: float = 0.09,
    hook_pause_seconds: float | None = None,
) -> list[float]:
    """Compute natural spoken breath gaps between adjacent narration chunks.

    If hook_pause_seconds is set (e.g. 0.75s), it is applied to the very first boundary
    (immediately after the isolated opening hook) exactly once, while subsequent boundaries
    use standard sentence/clause gap durations.
    """
    if len(chunks) <= 1:
        return []

    gaps: list[float] = []
    for idx, chunk in enumerate(chunks[:-1]):
        if idx == 0 and hook_pause_seconds is not None and hook_pause_seconds > 0.0:
            gaps.append(float(hook_pause_seconds))
            continue
        stripped = chunk.strip()
        if re.search(r"[.!?][\"'”’)]*\s*$", stripped):
            gaps.append(sentence_gap_seconds)
        else:
            gaps.append(clause_gap_seconds)
    return gaps


def concatenate_audio_segments(
    segments: list[tuple[bytes, int]],
    gap_seconds: float | list[float] = 0.0,
    start_pad_seconds: float = 0.0,
    end_pad_seconds: float = 0.0,
) -> tuple[bytes, int, float]:
    """Concatenate multiple WAV byte streams into a single seamless continuous WAV asset.

    Validates that sample rates, channel formats, and data types match, applies subtle
    boundary de-click tapers, prepends start_pad_seconds of silence, appends
    end_pad_seconds of silence, and returns uncompressed 16-bit PCM WAV bytes with valid headers.

    Args:
        segments: List of tuples (wav_bytes, sample_rate).
        gap_seconds: Silence gap(s) between chunks in seconds—either a single float applied
            to all boundaries (default: 0.0) or a list of per-boundary floats.
        start_pad_seconds: Leading silence padding in seconds (default: 0.0).
        end_pad_seconds: Trailing silence padding in seconds (default: 0.0).

    Returns:
        Tuple of (combined_wav_bytes, sample_rate, total_duration_seconds).
    """
    if not segments:
        raise ValueError("No audio segments provided to concatenate")

    target_sr = segments[0][1]

    if len(segments) == 1 and start_pad_seconds <= 0.0 and end_pad_seconds <= 0.0:
        wav_bytes, sr = segments[0]
        data, file_sr = sf.read(io.BytesIO(wav_bytes))
        dur = round(len(data) / float(file_sr), 3)
        return wav_bytes, file_sr, dur

    all_data: list[np.ndarray] = []
    n_segs = len(segments)

    # 1. Prepend start silence padding
    if start_pad_seconds > 0.0:
        start_samples = int(round(start_pad_seconds * target_sr))
        if start_samples > 0:
            all_data.append(np.zeros(start_samples, dtype=np.float32))

    for idx, (seg_bytes, sr) in enumerate(segments):
        data, read_sr = sf.read(io.BytesIO(seg_bytes))
        if read_sr != target_sr:
            raise ValueError(
                f"Sample rate mismatch in audio segment {idx}: expected {target_sr} Hz, got {read_sr} Hz"
            )
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        seg_arr = data.astype(np.float32).copy()

        # Apply subtle 3ms boundary de-click taper at join edges without changing sample count
        fade_len = min(len(seg_arr) // 8, int(0.003 * target_sr))
        if fade_len > 0:
            if idx > 0 or start_pad_seconds > 0.0:
                seg_arr[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            if idx < n_segs - 1 or end_pad_seconds > 0.0:
                seg_arr[-fade_len:] *= np.linspace(1.0, 0.0, fade_len, dtype=np.float32)

        all_data.append(seg_arr)

        if idx < n_segs - 1:
            if isinstance(gap_seconds, (list, tuple)):
                boundary_gap = float(gap_seconds[idx]) if idx < len(gap_seconds) else 0.0
            else:
                boundary_gap = float(gap_seconds)
            if boundary_gap > 0.0:
                silence_samples = int(round(boundary_gap * target_sr))
                if silence_samples > 0:
                    all_data.append(np.zeros(silence_samples, dtype=np.float32))

    # 2. Append end silence padding
    if end_pad_seconds > 0.0:
        end_samples = int(round(end_pad_seconds * target_sr))
        if end_samples > 0:
            all_data.append(np.zeros(end_samples, dtype=np.float32))

    combined = np.concatenate(all_data) if all_data else np.array([], dtype=np.float32)
    total_dur = round(len(combined) / float(target_sr), 3)

    buf = io.BytesIO()
    sf.write(buf, combined, target_sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), target_sr, total_dur


def estimate_scene_timings(
    scenes: list[dict[str, Any]],
    total_duration: float,
    start_silence_seconds: float = 0.0,
    end_silence_seconds: float = 0.0,
    opening_hook_pause_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    """Estimate start and end timestamps for each scene within the continuous narration.

    Uses phonetic word weighting based on syllables and consonants to allocate the continuous
    duration proportionally across scenes. Respects start and end silence paddings.

    Args:
        scenes: List of scene dictionaries (with 'index' and 'narration').
        total_duration: Total duration of the continuous narration audio in seconds.
        start_silence_seconds: Leading silence padding in seconds.
        end_silence_seconds: Trailing silence padding in seconds.
        opening_hook_pause_seconds: Opening hook pause padding in seconds.

    Returns:
        List of dictionaries with scene timing metadata.
    """
    if not scenes:
        return []

    scene_weights: list[float] = []
    for sc in scenes:
        narration = str(sc.get("narration") or "").strip()
        tokens = re.findall(r"\b[\w'-]+\b", narration)
        weights = _compute_word_weights(tokens)
        total_w = sum(weights) if weights else 1.0
        scene_weights.append(max(0.1, total_w))

    total_weight = sum(scene_weights) or 1.0
    usable_duration = max(0.1, total_duration - start_silence_seconds - end_silence_seconds)

    timings: list[dict[str, Any]] = []
    current_time = 0.0
    n_scenes = len(scenes)

    for idx, sc in enumerate(scenes):
        scene_idx = int(sc.get("index", idx + 1))
        scene_id = sc.get("id") or sc.get("scene_id") or scene_idx
        narration = str(sc.get("narration") or "").strip()

        w = scene_weights[idx]
        allocated_dur = round(usable_duration * (w / total_weight), 3)

        if idx == 0:
            start_time = 0.0
            speech_start = round(start_silence_seconds, 3)
            if n_scenes == 1:
                end_time = round(total_duration, 3)
                speech_end = round(total_duration - end_silence_seconds, 3)
            else:
                extra_hook_gap = min(opening_hook_pause_seconds * 0.5, 0.4) if opening_hook_pause_seconds > 0 else 0.0
                end_time = round(start_silence_seconds + allocated_dur + extra_hook_gap, 3)
                speech_end = round(start_silence_seconds + allocated_dur, 3)
                current_time = end_time
        elif idx == n_scenes - 1:
            start_time = round(current_time, 3)
            end_time = round(total_duration, 3)
            speech_start = round(start_time, 3)
            speech_end = round(max(speech_start + 0.1, total_duration - end_silence_seconds), 3)
        else:
            start_time = round(current_time, 3)
            end_time = round(start_time + allocated_dur, 3)
            speech_start = round(start_time, 3)
            speech_end = round(end_time, 3)
            current_time = end_time

        scene_dur = round(max(0.1, end_time - start_time), 3)
        timings.append(
            {
                "index": scene_idx,
                "scene_id": scene_id,
                "narration": narration,
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": scene_dur,
                "speech_start_time": speech_start,
                "speech_end_time": speech_end,
                "timing_type": "estimated",
            }
        )

    return timings


def generate_continuous_audio(
    narration_script: str,
    provider: BaseVoiceoverProvider,
    voice: str = "am_adam",
    speed: float = 1.0,
    target_duration: float | None = None,
    max_chunk_words: int = 65,
    start_silence_seconds: float | None = None,
    opening_hook_pause_seconds: float | None = None,
    end_silence_seconds: float | None = None,
    hook_text: str | None = None,
    isolate_hook: bool = True,
) -> tuple[VoiceoverResult, int]:
    """Generate speech for the entire continuous narration script.

    Splits into balanced sentence/paragraph chunks when chunking is required,
    isolates the opening hook when requested, generates speech using the configured
    voiceover provider, combines segments with natural inter-chunk breath pauses
    and hook pause, and pads with start and end silence.

    Returns:
        Tuple of (VoiceoverResult, chunks_count).
    """
    clean_text = narration_script.strip()
    if not clean_text:
        raise ValueError("Narration script text is empty")

    start_silence = (
        float(start_silence_seconds)
        if start_silence_seconds is not None
        else float(getattr(settings, "voiceover_start_silence_seconds", 0.75))
    )
    hook_pause = (
        float(opening_hook_pause_seconds)
        if opening_hook_pause_seconds is not None
        else float(getattr(settings, "voiceover_opening_hook_pause_seconds", 0.75))
    )
    end_silence = (
        float(end_silence_seconds)
        if end_silence_seconds is not None
        else float(getattr(settings, "voiceover_end_silence_seconds", 1.0))
    )

    chunks = split_script_into_chunks(
        clean_text,
        max_words=max_chunk_words,
        isolate_hook=isolate_hook,
        hook_text=hook_text,
    )
    if not chunks:
        chunks = [clean_text]

    start_time = time.perf_counter()

    if len(chunks) == 1:
        # If target_duration was specified for mock, subtract paddings if positive
        adj_target = target_duration
        if adj_target is not None and float(adj_target) > (start_silence + end_silence):
            adj_target = round(float(adj_target) - start_silence - end_silence, 3)

        res = provider.generate(
            text=chunks[0], voice=voice, speed=speed, target_duration=adj_target
        )
        if start_silence > 0.0 or end_silence > 0.0:
            comb_bytes, sample_rate, total_duration = concatenate_audio_segments(
                [(res.audio_bytes, res.sample_rate)],
                start_pad_seconds=start_silence,
                end_pad_seconds=end_silence,
            )
            res.audio_bytes = comb_bytes
            res.duration_seconds = total_duration
            res.sample_rate = sample_rate
        return res, 1

    segments: list[tuple[bytes, int]] = []
    total_calls = 0

    for chunk in chunks:
        chunk_res = provider.generate(text=chunk, voice=voice, speed=speed)
        total_calls += getattr(chunk_res, "inference_calls", 1) or 1
        segments.append((chunk_res.audio_bytes, chunk_res.sample_rate))

    inter_chunk_gaps = compute_inter_chunk_gaps(
        chunks, hook_pause_seconds=hook_pause
    )
    comb_bytes, sample_rate, total_duration = concatenate_audio_segments(
        segments,
        gap_seconds=inter_chunk_gaps,
        start_pad_seconds=start_silence,
        end_pad_seconds=end_silence,
    )
    latency = time.perf_counter() - start_time

    combined_result = VoiceoverResult(
        audio_bytes=comb_bytes,
        sample_rate=sample_rate,
        duration_seconds=total_duration,
        latency_seconds=round(latency, 4),
        provider=provider.provider_name,
        model=provider.model,
        voice=voice,
        speed=speed,
        inference_calls=total_calls,
        chunks_count=len(chunks),
    )
    return combined_result, len(chunks)
