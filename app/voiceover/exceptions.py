"""Exceptions for the voiceover provider layer."""


class VoiceoverError(Exception):
    """Base exception for all voiceover provider errors."""


class VoiceoverConfigError(VoiceoverError):
    """Raised when configuration, voice model weights, or dependencies are missing/invalid."""


class VoiceoverProviderError(VoiceoverError):
    """Raised when a voiceover provider fails during inference."""


class VoiceoverRateLimitError(VoiceoverError):
    """Raised when a rate limit is exceeded."""


class VoiceoverTimeoutError(VoiceoverError):
    """Raised when voiceover generation times out."""
