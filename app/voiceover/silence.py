"""Audio silence trimming utilities for Shorts Studio voiceover audio.

Provides robust silence detection and trimming on speech audio waveforms,
preserving natural speech onsets (breathing headroom) and trailing decay
while preventing abrupt clicks at waveform boundaries.
"""

from __future__ import annotations

import io
import re
from typing import Any
import numpy as np


def trim_silence_samples(
    samples: np.ndarray,
    sample_rate: int,
    is_terminal: bool = False,
    lead_pad_ms: float = 35.0,
    trail_pad_ms: float = 80.0,
    terminal_trail_pad_ms: float = 110.0,
) -> np.ndarray:
    """Trim excessive leading and trailing silence from speech audio samples.

    Args:
        samples: 1D or 2D numpy array of audio samples (float32).
        sample_rate: Audio sampling rate in Hz.
        is_terminal: If True, indicates scene ends in terminal punctuation (. ! ?),
            preserving slightly longer natural pause headroom (110ms vs 80ms).
        lead_pad_ms: Milliseconds of leading headroom preserved before speech onset (default: 35ms).
        trail_pad_ms: Milliseconds of trailing pause preserved for mid-sentence cuts (default: 80ms).
        terminal_trail_pad_ms: Milliseconds of trailing pause for sentence-ending cuts (default: 110ms).

    Returns:
        Cleanly trimmed numpy array with 5ms micro-fades at boundaries to prevent clicks.
    """
    if len(samples) == 0 or sample_rate <= 0:
        return samples

    # Flatten stereo to mono for energy envelope detection
    work = np.mean(samples, axis=1) if samples.ndim > 1 else samples
    abs_d = np.abs(work)
    peak = float(np.max(abs_d)) if len(abs_d) > 0 else 0.0

    if peak < 0.001:
        # Audio is completely silent or extremely faint; return unchanged
        return samples

    # Adaptive silence threshold: 1.2% of peak amplitude or minimum 0.004
    thresh = max(0.004, 0.012 * peak)
    speech = np.where(abs_d > thresh)[0]

    if len(speech) == 0:
        return samples

    first = int(speech[0])
    last = int(speech[-1])

    lead_pad = int((lead_pad_ms / 1000.0) * sample_rate)
    pad_ms = terminal_trail_pad_ms if is_terminal else trail_pad_ms
    trail_pad = int((pad_ms / 1000.0) * sample_rate)

    start_idx = max(0, first - lead_pad)
    end_idx = min(len(samples), last + 1 + trail_pad)

    # If trimming would change less than 15ms total, avoid re-slicing
    if start_idx == 0 and end_idx == len(samples):
        trimmed = samples.copy()
    else:
        trimmed = samples[start_idx:end_idx].copy()

    # Apply 5ms micro-fade in and out to guarantee zero clicks or non-zero crossing pops
    fade_len = min(len(trimmed) // 4, int(0.005 * sample_rate))
    if fade_len > 0:
        fade_in = np.linspace(0.0, 1.0, fade_len, dtype=trimmed.dtype)
        fade_out = np.linspace(1.0, 0.0, fade_len, dtype=trimmed.dtype)
        if trimmed.ndim == 1:
            trimmed[:fade_len] *= fade_in
            trimmed[-fade_len:] *= fade_out
        else:
            trimmed[:fade_len] *= fade_in[:, np.newaxis]
            trimmed[-fade_len:] *= fade_out[:, np.newaxis]

    return trimmed


def trim_silence_wav(
    wav_bytes: bytes,
    is_terminal: bool = False,
    lead_pad_ms: float = 35.0,
    trail_pad_ms: float = 80.0,
    terminal_trail_pad_ms: float = 110.0,
) -> tuple[bytes, int, float]:
    """Read a WAV byte stream, trim excessive silence, and re-export to WAV.

    Returns:
        Tuple of (trimmed_wav_bytes, sample_rate, duration_seconds).
    """
    import soundfile as sf

    buf_in = io.BytesIO(wav_bytes)
    data, sr = sf.read(buf_in, dtype="float32")

    trimmed = trim_silence_samples(
        data,
        sample_rate=sr,
        is_terminal=is_terminal,
        lead_pad_ms=lead_pad_ms,
        trail_pad_ms=trail_pad_ms,
        terminal_trail_pad_ms=terminal_trail_pad_ms,
    )

    duration = round(len(trimmed) / float(sr), 3)

    buf_out = io.BytesIO()
    sf.write(buf_out, trimmed, sr, format="WAV", subtype="PCM_16")
    return buf_out.getvalue(), sr, duration
