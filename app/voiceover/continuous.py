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


def split_script_into_chunks(
    text: str, max_words: int = 60, max_chars: int = 350
) -> list[str]:
    """Split narration script into natural linguistic chunks at sentence or paragraph boundaries.

    Ensures words are never split mid-word, sentence order and punctuation are preserved,
    and no words are duplicated or omitted.

    Args:
        text: Full narration script text.
        max_words: Maximum target spoken words per chunk (default: 60).
        max_chars: Maximum target characters per chunk (default: 350).

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
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]

        current_chunk_sentences: list[str] = []
        current_word_count = 0
        current_char_count = 0

        for sentence in sentences:
            s_words = sentence.split()
            s_word_len = len(s_words)
            s_char_len = len(sentence)

            # If a single sentence exceeds limits on its own, split on clause boundaries
            if s_word_len > max_words or s_char_len > max_chars:
                if current_chunk_sentences:
                    chunks.append(" ".join(current_chunk_sentences))
                    current_chunk_sentences = []
                    current_word_count = 0
                    current_char_count = 0

                # Split clause on semicolon, colon, or comma
                clauses = [c.strip() for c in re.split(r"(?<=[;:,—–])\s+", sentence) if c.strip()]
                curr_clause_parts: list[str] = []
                curr_c_words = 0
                for clause in clauses:
                    c_len = len(clause.split())
                    if curr_c_words + c_len > max_words and curr_clause_parts:
                        chunks.append(" ".join(curr_clause_parts))
                        curr_clause_parts = [clause]
                        curr_c_words = c_len
                    else:
                        curr_clause_parts.append(clause)
                        curr_c_words += c_len
                if curr_clause_parts:
                    chunks.append(" ".join(curr_clause_parts))
                continue

            if (
                current_chunk_sentences
                and (current_word_count + s_word_len > max_words or current_char_count + s_char_len > max_chars)
            ):
                chunks.append(" ".join(current_chunk_sentences))
                current_chunk_sentences = [sentence]
                current_word_count = s_word_len
                current_char_count = s_char_len
            else:
                current_chunk_sentences.append(sentence)
                current_word_count += s_word_len
                current_char_count += s_char_len

        if current_chunk_sentences:
            chunks.append(" ".join(current_chunk_sentences))

    return chunks


def concatenate_audio_segments(
    segments: list[tuple[bytes, int]], gap_seconds: float = 0.0
) -> tuple[bytes, int, float]:
    """Concatenate multiple WAV byte streams into a single seamless continuous WAV asset.

    Validates that sample rates, channel formats, and data types match,
    and returns uncompressed 16-bit PCM WAV bytes with valid headers.

    Args:
        segments: List of tuples (wav_bytes, sample_rate).
        gap_seconds: Optional silence gap between chunks in seconds (default: 0.0).

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

    for idx, (seg_bytes, sr) in enumerate(segments):
        data, read_sr = sf.read(io.BytesIO(seg_bytes))
        if read_sr != target_sr:
            raise ValueError(
                f"Sample rate mismatch in audio segment {idx}: expected {target_sr} Hz, got {read_sr} Hz"
            )
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        all_data.append(data.astype(np.float32))

        if gap_seconds > 0 and idx < len(segments) - 1:
            silence_samples = int(gap_seconds * target_sr)
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


def slice_scene_audio(
    continuous_audio_bytes: bytes, scene_timings: list[dict[str, Any]]
) -> list[bytes]:
    """Slice continuous narration WAV into individual scene WAV files for pipeline compatibility.

    Applies subtle boundary micro-fades (3ms) to prevent speaker pops/clicks.

    Args:
        continuous_audio_bytes: WAV byte stream of the continuous narration audio.
        scene_timings: List of scene timing metadata from estimate_scene_timings.

    Returns:
        List of WAV byte streams corresponding to each scene.
    """
    if not continuous_audio_bytes:
        raise ValueError("Continuous audio bytes are empty")
    if not scene_timings:
        return []

    data, sr = sf.read(io.BytesIO(continuous_audio_bytes))
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    total_samples = len(data)
    sliced_bytes_list: list[bytes] = []

    for i, timing in enumerate(scene_timings):
        start_t = float(timing.get("start_time", 0.0))
        end_t = float(timing.get("end_time", 0.0))

        start_idx = max(0, min(total_samples, int(round(start_t * sr))))
        if i == len(scene_timings) - 1:
            end_idx = total_samples
        else:
            end_idx = max(start_idx + 1, min(total_samples, int(round(end_t * sr))))

        scene_samples = data[start_idx:end_idx].copy()
        if len(scene_samples) == 0:
            scene_samples = np.zeros(int(0.1 * sr), dtype=np.float32)

        # Apply subtle 3ms micro-fade at cut boundaries
        fade_len = min(len(scene_samples) // 4, int(0.003 * sr))
        if fade_len > 0:
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            scene_samples[:fade_len] *= fade_in
            scene_samples[-fade_len:] *= fade_out

        buf = io.BytesIO()
        sf.write(buf, scene_samples, sr, format="WAV", subtype="PCM_16")
        sliced_bytes_list.append(buf.getvalue())

    return sliced_bytes_list


def generate_continuous_audio(
    narration_script: str,
    provider: BaseVoiceoverProvider,
    voice: str = "af_heart",
    speed: float = 1.0,
    target_duration: float | None = None,
    max_chunk_words: int = 60,
) -> tuple[VoiceoverResult, int]:
    """Generate speech for the entire continuous narration script.

    Splits at natural sentence/paragraph boundaries if chunking is required,
    generates speech using the configured voiceover provider, and combines segments
    into a single continuous WAV asset.

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

    comb_bytes, sample_rate, total_duration = concatenate_audio_segments(segments, gap_seconds=0.0)
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
