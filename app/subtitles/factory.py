"""Factory for resolving subtitle providers."""

from __future__ import annotations

from typing import Any

from app.config import settings

from .base import BaseSubtitleProvider, SubtitleConfigError
from .mock import MockSubtitleProvider
from .acoustic import AcousticSubtitleProvider


def get_subtitle_provider(provider_name: str | None = None) -> BaseSubtitleProvider:
    """Instantiate and return the configured subtitle provider."""
    selected = (provider_name or getattr(settings, "subtitles_provider", "acoustic") or "acoustic").lower().strip()

    if selected == "mock":
        return MockSubtitleProvider()
    if selected in ("acoustic", "vosk", "waveform", "default"):
        return AcousticSubtitleProvider()

    # Optional Whisper integration if configured and installed
    if selected in ("whisper", "faster-whisper"):
        try:
            # If faster-whisper is installed in environment, use it
            from .whisper import WhisperSubtitleProvider

            return WhisperSubtitleProvider()
        except ImportError:
            # Graceful fallback to acoustic aligner with warning
            return AcousticSubtitleProvider()

    raise SubtitleConfigError(
        f"Unknown subtitle provider '{selected}'. Supported providers are: 'acoustic', 'mock'."
    )
