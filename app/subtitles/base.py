"""Base class and interface for subtitle providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from .types import SubtitleResult


class SubtitleError(Exception):
    """Base exception for subtitle provider failures."""


class SubtitleConfigError(SubtitleError):
    """Raised when provider configuration or required assets are missing."""


class BaseSubtitleProvider(ABC):
    """Abstract base class for all subtitle generation providers."""

    def __init__(self, model: str = "default", timeout_seconds: float = 60.0):
        self.model = model
        self.timeout_seconds = timeout_seconds

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """The identifier of this provider (e.g. 'acoustic', 'mock')."""

    @abstractmethod
    def generate(
        self,
        scenes: list[dict[str, Any]],
        storage_root: Path | None = None,
    ) -> SubtitleResult:
        """Generate synchronized subtitle data with word-level timestamps.

        Args:
            scenes: List of scene dictionaries containing:
                - 'index': Scene index (1, 2, ...)
                - 'narration': The spoken text string for this scene
                - 'duration_seconds': Duration of the scene in seconds
                - 'file': Relative path to scene audio WAV file (optional)
            storage_root: Path to base storage directory to resolve audio files.

        Returns:
            SubtitleResult containing segments and word timestamps.
        """
