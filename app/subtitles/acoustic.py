"""Acoustic energy and phonetic alignment subtitle provider.

Uses standard-library and already-installed scientific packages (soundfile, numpy)
to analyze scene WAV audio waveforms, detect exact speech activity onsets and offsets,
and align narration words to acoustic energy envelopes with micro-pause detection.
Zero paid APIs, zero heavy dependencies, 100% offline and deterministic.
"""

from __future__ import annotations

from pathlib import Path
import re
import time
from typing import Any

import numpy as np

from .base import BaseSubtitleProvider, SubtitleError
from .types import SubtitleResult, SubtitleSegment, WordTimestamp


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
        # Syllables account for core voiced duration; characters account for consonant clusters
        weight = syllables * 1.6 + len(clean) * 0.25
        weights.append(weight)
    return weights


def _align_scene_audio(
    audio_path: Path | None,
    narration: str,
    duration_fallback: float,
    scene_offset: float,
) -> tuple[float, list[WordTimestamp]]:
    """Analyze audio waveform (if available) and align narration words to acoustic timing."""
    tokens = re.findall(r"\b[\w'-]+\b", narration)
    if not tokens:
        return duration_fallback, []

    weights = _compute_word_weights(tokens)
    total_weight = sum(weights) or 1.0

    # Default boundaries without audio inspection
    onset_rel = 0.04
    offset_rel = max(0.2, duration_fallback - 0.08)
    actual_duration = duration_fallback
    smoothed_rms = None
    frame_ms = 10

    # Try acoustic analysis if audio file is present
    if audio_path and audio_path.is_file():
        try:
            import soundfile as sf

            data, sr = sf.read(str(audio_path))
            if len(data) > 0 and sr > 0:
                actual_duration = round(len(data) / float(sr), 3)

                # Convert stereo to mono if needed
                if data.ndim > 1:
                    data = np.mean(data, axis=1)

                frame_len = int(sr * (frame_ms / 1000.0))
                num_frames = len(data) // frame_len

                if num_frames > 4:
                    frames = data[: num_frames * frame_len].reshape(num_frames, frame_len)
                    rms = np.sqrt(np.mean(frames**2, axis=1))

                    # Adaptive silence threshold: 12% of 75th percentile energy or minimum 0.005
                    p75 = np.percentile(rms, 75)
                    thresh = max(0.004, p75 * 0.12)
                    speech_frames = np.where(rms > thresh)[0]

                    if len(speech_frames) > 0:
                        onset_rel = max(0.0, float(speech_frames[0] * (frame_ms / 1000.0)))
                        offset_rel = min(
                            actual_duration,
                            float((speech_frames[-1] + 1) * (frame_ms / 1000.0)),
                        )

                    # Smooth RMS slightly for energy valley snapping
                    kernel = np.ones(3) / 3.0
                    smoothed_rms = np.convolve(rms, kernel, mode="same")
        except Exception:
            # Fall back gracefully to duration_fallback if audio reading encounters an issue
            pass

    active_speech_duration = max(0.15, offset_rel - onset_rel)
    words: list[WordTimestamp] = []

    curr_rel = onset_rel
    total_tokens = len(tokens)

    for i, (token, wt) in enumerate(zip(tokens, weights)):
        if i == total_tokens - 1:
            next_rel = max(curr_rel + 0.06, offset_rel)
        else:
            raw_dur = active_speech_duration * (wt / total_weight)
            next_rel = curr_rel + raw_dur

            # Acoustic valley snapping for internal boundaries
            if smoothed_rms is not None:
                target_frame = int(next_rel / (frame_ms / 1000.0))
                search_window = 4  # +/- 40ms search window
                f_start = max(0, target_frame - search_window)
                f_end = min(len(smoothed_rms), target_frame + search_window + 1)
                if f_end > f_start:
                    min_idx = f_start + int(np.argmin(smoothed_rms[f_start:f_end]))
                    snapped_rel = min_idx * (frame_ms / 1000.0)
                    # Keep boundary strictly forward with at least 60ms word duration
                    if snapped_rel > curr_rel + 0.06 and snapped_rel < offset_rel - 0.06:
                        next_rel = snapped_rel

        w_start = round(scene_offset + curr_rel, 3)
        w_end = round(scene_offset + min(actual_duration, next_rel), 3)

        # Enforce positive duration
        if w_end <= w_start:
            w_end = round(w_start + 0.05, 3)

        words.append(
            WordTimestamp(
                text=token,
                start=w_start,
                end=w_end,
                confidence=0.98 if smoothed_rms is not None else 0.90,
            )
        )
        curr_rel = next_rel

    return actual_duration, words


class AcousticSubtitleProvider(BaseSubtitleProvider):
    """High-accuracy subtitle provider aligning ground-truth narration to audio waveforms."""

    def __init__(
        self,
        model: str = "acoustic-waveform-aligner-v1",
        timeout_seconds: float = 30.0,
    ):
        super().__init__(model=model, timeout_seconds=timeout_seconds)

    @property
    def provider_name(self) -> str:
        return "acoustic"

    def generate(
        self,
        scenes: list[dict[str, Any]],
        storage_root: Path | None = None,
    ) -> SubtitleResult:
        start_time = time.perf_counter()
        segments: list[SubtitleSegment] = []
        total_words_count = 0
        cumulative_offset = 0.0

        for scene in scenes:
            scene_idx = int(scene.get("index", len(segments) + 1))
            narration = str(scene.get("narration") or "").strip()
            est_duration = float(scene.get("duration_seconds", 0.0) or 0.0)
            if est_duration <= 0.0:
                est_duration = 2.5

            audio_file_rel = scene.get("file")
            audio_abs_path = None
            if audio_file_rel and storage_root:
                audio_abs_path = Path(storage_root) / audio_file_rel

            actual_duration, words = _align_scene_audio(
                audio_path=audio_abs_path,
                narration=narration,
                duration_fallback=est_duration,
                scene_offset=cumulative_offset,
            )

            seg_start = round(cumulative_offset, 3)
            seg_end = round(cumulative_offset + actual_duration, 3)

            segments.append(
                SubtitleSegment(
                    index=scene_idx,
                    text=narration,
                    start=seg_start,
                    end=seg_end,
                    words=words,
                )
            )

            total_words_count += len(words)
            cumulative_offset += actual_duration

        latency = round(time.perf_counter() - start_time, 4)

        return SubtitleResult(
            segments=segments,
            total_words=total_words_count,
            total_duration_seconds=round(cumulative_offset, 3),
            provider=self.provider_name,
            model=self.model,
            latency_seconds=latency,
        )
