"""Voiceover generation module for Shorts Studio."""

from .base import BaseVoiceoverProvider
from .exceptions import (
    VoiceoverConfigError,
    VoiceoverError,
    VoiceoverProviderError,
    VoiceoverRateLimitError,
    VoiceoverTimeoutError,
)
from .continuous import (
    collect_narration_script,
    concatenate_audio_segments,
    estimate_scene_timings,
    generate_continuous_audio,
    slice_scene_audio,
    split_script_into_chunks,
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
    "collect_narration_script",
    "concatenate_audio_segments",
    "create_mock_wav",
    "estimate_scene_timings",
    "generate_continuous_audio",
    "get_voiceover_provider",
    "slice_scene_audio",
    "split_script_into_chunks",
    "trim_silence_samples",
    "trim_silence_wav",
]
