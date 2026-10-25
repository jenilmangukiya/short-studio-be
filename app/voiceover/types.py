"""Types and data structures for the voiceover provider layer."""

from dataclasses import dataclass


@dataclass
class VoiceoverResult:
    """Standardized result returned by all voiceover providers.

    Attributes:
        audio_bytes: Raw bytes of the generated audio (WAV format).
        sample_rate: Sample rate in Hz (e.g. 24000).
        duration_seconds: Duration of the generated audio in seconds.
        latency_seconds: Time taken to generate the audio in seconds.
        provider: Identifier of the provider (e.g. 'mock', 'kokoro').
        model: Specific model name used.
        voice: Voice name or identifier used (e.g. 'af_heart').
        speed: Speech speed multiplier used (e.g. 1.0).
    """

    audio_bytes: bytes
    sample_rate: int
    duration_seconds: float
    latency_seconds: float
    provider: str
    model: str
    voice: str
    speed: float = 1.0
