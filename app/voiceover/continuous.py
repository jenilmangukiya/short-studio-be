"""Continuous narration generation, text-aware chunking, and scene mapping for Shorts Studio."""

from __future__ import annotations

import io
from pathlib import Path
import re
import time
from typing import Any

import numpy as np
import soundfile as sf

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

        validated_scenes.append(
            {
                "index": scene_idx,
                "id": sc.get("id") or sc.get("scene_id") or scene_idx,
                "narration": narration,
                "est_seconds": sc.get("est_seconds"),
            }
        )

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
    text: str, max_words: int = 65, max_chars: int = 380
) -> list[str]:
    """Split narration script into balanced linguistic chunks at sentence or paragraph boundaries.

    Ensures words are never split mid-word, sentence order and punctuation are preserved,
    no words are duplicated or omitted, and multi-chunk splits are balanced to keep
    each chunk well below Kokoro's 510-phoneme window without tiny orphan chunks.

    Args:
        text: Full narration script text.
        max_words: Maximum target spoken words per chunk (default: 65).
        max_chars: Maximum target characters per chunk (default: 380).

    Returns:
        List of cleanly chunked narration text strings.
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return []

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
) -> list[float]:
    """Compute natural spoken breath gaps between adjacent narration chunks.

    When combined with `trim_silence_samples` (which preserves 110ms trailing pad on
    sentence-ending chunks and 35ms leading pad on the next chunk), a 240ms inter-chunk
    sentence gap yields a total ~385ms pause—matching Kokoro's natural internal
    sentence-to-sentence pause (360ms–480ms).
    """
    if len(chunks) <= 1:
        return []

    gaps: list[float] = []
    for chunk in chunks[:-1]:
        stripped = chunk.strip()
        if re.search(r"[.!?][\"'”’)]*\s*$", stripped):
            gaps.append(sentence_gap_seconds)
        else:
            gaps.append(clause_gap_seconds)
    return gaps


def concatenate_audio_segments(
    segments: list[tuple[bytes, int]], gap_seconds: float | list[float] = 0.0
) -> tuple[bytes, int, float]:
    """Concatenate multiple WAV byte streams into a single seamless continuous WAV asset.

    Validates that sample rates, channel formats, and data types match, applies a subtle
    3ms boundary taper to prevent waveform clicks, and returns uncompressed 16-bit PCM WAV
    bytes with valid headers.

    Args:
        segments: List of tuples (wav_bytes, sample_rate).
        gap_seconds: Silence gap(s) between chunks in seconds—either a single float applied
            to all boundaries (default: 0.0) or a list of per-boundary floats.

    Returns:
        Tuple of (combined_wav_bytes, sample_rate, total_duration_seconds).
    """
    if not segments:
        raise ValueError("No audio segments provided to concatenate")

    if len(segments) == 1:
        wav_bytes, sr = segments[0]
        data, file_sr = sf.read(io.BytesIO(wav_bytes))
        dur = round(len(data) / float(file_sr), 3)
        return wav_bytes, file_sr, dur

    target_sr = segments[0][1]
    all_data: list[np.ndarray] = []
    n_segs = len(segments)

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
            if idx > 0:
                seg_arr[:fade_len] *= np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            if idx < n_segs - 1:
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

    combined = np.concatenate(all_data)
    total_dur = round(len(combined) / float(target_sr), 3)

    buf = io.BytesIO()
    sf.write(buf, combined, target_sr, format="WAV", subtype="PCM_16")
    return buf.getvalue(), target_sr, total_dur


def estimate_scene_timings(
    scenes: list[dict[str, Any]], total_duration: float
) -> list[dict[str, Any]]:
    """Estimate start and end timestamps for each scene within the continuous narration.

    Uses phonetic word weighting based on syllables and consonants to allocate the continuous
    duration proportionally across scenes. Explicitly labels timing as 'estimated'.

    Args:
        scenes: List of scene dictionaries (with 'index' and 'narration').
        total_duration: Total duration of the continuous narration audio in seconds.

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

    timings: list[dict[str, Any]] = []
    current_time = 0.0

    for idx, sc in enumerate(scenes):
        scene_idx = int(sc.get("index", idx + 1))
        scene_id = sc.get("id") or sc.get("scene_id") or scene_idx
        narration = str(sc.get("narration") or "").strip()

        w = scene_weights[idx]
        allocated_dur = round(total_duration * (w / total_weight), 3)

        start_time = round(current_time, 3)
        if idx == len(scenes) - 1:
            end_time = round(total_duration, 3)
            scene_dur = round(max(0.1, end_time - start_time), 3)
        else:
            end_time = round(start_time + allocated_dur, 3)
            scene_dur = round(max(0.1, allocated_dur), 3)
            current_time = end_time

        timings.append(
            {
                "index": scene_idx,
                "scene_id": scene_id,
                "narration": narration,
                "start_time": start_time,
                "end_time": end_time,
                "duration_seconds": scene_dur,
                "timing_type": "estimated",
            }
        )

    return timings


def generate_continuous_audio(
    narration_script: str,
    provider: BaseVoiceoverProvider,
    voice: str = "af_heart",
    speed: float = 1.0,
    target_duration: float | None = None,
    max_chunk_words: int = 65,
) -> tuple[VoiceoverResult, int]:
    """Generate speech for the entire continuous narration script.

    Splits into balanced sentence/paragraph chunks when chunking is required,
    generates speech using the configured voiceover provider, and combines segments
    with natural inter-chunk breath pauses into a single continuous WAV asset.

    Returns:
        Tuple of (VoiceoverResult, chunks_count).
    """
    clean_text = narration_script.strip()
    if not clean_text:
        raise ValueError("Narration script text is empty")

    chunks = split_script_into_chunks(clean_text, max_words=max_chunk_words)
    if not chunks:
        chunks = [clean_text]

    start_time = time.perf_counter()

    if len(chunks) == 1:
        res = provider.generate(
            text=chunks[0], voice=voice, speed=speed, target_duration=target_duration
        )
        return res, 1

    segments: list[tuple[bytes, int]] = []
    total_calls = 0

    for chunk in chunks:
        chunk_res = provider.generate(text=chunk, voice=voice, speed=speed)
        total_calls += getattr(chunk_res, "inference_calls", 1) or 1
        segments.append((chunk_res.audio_bytes, chunk_res.sample_rate))

    inter_chunk_gaps = compute_inter_chunk_gaps(chunks)
    comb_bytes, sample_rate, total_duration = concatenate_audio_segments(
        segments, gap_seconds=inter_chunk_gaps
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
