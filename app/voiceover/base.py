"""Base interface and common logic for voiceover generation providers."""

import time
from abc import ABC, abstractmethod

from .types import VoiceoverResult


class BaseVoiceoverProvider(ABC):
    """Abstract base class for all voiceover generation providers."""

    def __init__(
        self,
        model: str,
        timeout_seconds: float = 60.0,
    ):
        self.model = model
        self.timeout_seconds = timeout_seconds

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """The identifier of this provider (e.g. 'mock', 'kokoro')."""
        pass

    @abstractmethod
    def _execute_generate(
        self, text: str, voice: str, speed: float, target_duration: float | None = None
    ) -> tuple[bytes, int, float]:
        """Provider-specific voice generation call.

        Returns:
            Tuple of (audio_bytes, sample_rate, duration_seconds)
        """
        pass

    def generate(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        target_duration: float | None = None,
    ) -> VoiceoverResult:
        """Generate audio from narration text with the given voice and speed multiplier.

        Returns:
            VoiceoverResult containing the audio bytes (WAV format) and metadata.
        """
        start = time.perf_counter()
        audio_bytes, sample_rate, duration = self._execute_generate(
            text=text, voice=voice, speed=speed, target_duration=target_duration
        )
        latency = time.perf_counter() - start

        return VoiceoverResult(
            audio_bytes=audio_bytes,
            sample_rate=sample_rate,
            duration_seconds=round(duration, 3),
            latency_seconds=round(latency, 4),
            provider=self.provider_name,
            model=self.model,
            voice=voice,
            speed=speed,
            inference_calls=getattr(self, "_last_inference_batches", 1) or 1,
            chunks_count=1,
        )
