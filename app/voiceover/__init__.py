"""Voiceover generation module for Shorts Studio."""

from .base import BaseVoiceoverProvider
from .exceptions import (
    VoiceoverConfigError,
    VoiceoverError,
    VoiceoverProviderError,
    VoiceoverRateLimitError,
    VoiceoverTimeoutError,
)
from .factory import get_voiceover_provider
from .kokoro import KokoroVoiceoverProvider
from .mock import MockVoiceoverProvider, create_mock_wav
from .silence import trim_silence_samples, trim_silence_wav
from .types import VoiceoverResult

__all__ = [
    "BaseVoiceoverProvider",
    "KokoroVoiceoverProvider",
    "MockVoiceoverProvider",
    "VoiceoverConfigError",
    "VoiceoverError",
    "VoiceoverProviderError",
    "VoiceoverRateLimitError",
    "VoiceoverResult",
    "VoiceoverTimeoutError",
    "create_mock_wav",
    "get_voiceover_provider",
    "trim_silence_samples",
    "trim_silence_wav",
]
