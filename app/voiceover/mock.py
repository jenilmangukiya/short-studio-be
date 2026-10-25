"""Mock voiceover provider for local testing and development with zero AI cost."""

import io
import math
import struct
import wave

from .base import BaseVoiceoverProvider


def create_mock_wav(
    duration_seconds: float,
    sample_rate: int = 24000,
    frequency: float = 440.0,
) -> bytes:
    """Generate a valid 16-bit PCM mono WAV byte stream using only Python standard library.

    Args:
        duration_seconds: Audio duration in seconds.
        sample_rate: Audio sample rate in Hz (default: 24000 Hz).
        frequency: Audio tone frequency in Hz (default: 440.0 Hz).
    """
    duration = max(0.1, float(duration_seconds))
    num_samples = int(duration * sample_rate)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit PCM (2 bytes per sample)
        wf.setframerate(sample_rate)

        frames = bytearray()
        for i in range(num_samples):
            # Soft sine wave tone (amplitude 800 / 32767)
            sample_val = int(800 * math.sin(2 * math.pi * frequency * (i / sample_rate)))
            frames.extend(struct.pack("<h", sample_val))

        wf.writeframes(frames)

    return buf.getvalue()


class MockVoiceoverProvider(BaseVoiceoverProvider):
    """Mock voiceover provider requiring no external dependencies, model weights, or network calls."""

    def __init__(
        self,
        model: str = "mock-kokoro-model",
        timeout_seconds: float = 60.0,
    ):
        super().__init__(
            model=model,
            timeout_seconds=timeout_seconds,
        )

    @property
    def provider_name(self) -> str:
        return "mock"

    def _execute_generate(
        self, text: str, voice: str, speed: float, target_duration: float | None = None
    ) -> tuple[bytes, int, float]:
        effective_speed = max(0.2, min(5.0, float(speed)))
        if target_duration is not None and float(target_duration) > 0:
            calculated_duration = round(float(target_duration), 3)
        else:
            words = text.split()
            calculated_duration = max(0.5, round(len(words) / (2.5 * effective_speed), 3))

        sample_rate = 24000
        audio_bytes = create_mock_wav(
            duration_seconds=calculated_duration,
            sample_rate=sample_rate,
        )

        return audio_bytes, sample_rate, calculated_duration
