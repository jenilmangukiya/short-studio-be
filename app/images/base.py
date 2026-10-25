"""Base interface and common logic for image generation providers."""

import time
from abc import ABC, abstractmethod
from typing import Any

from .exceptions import (
    ImageConfigError,
    ImageContentBlockedError,
    ImageError,
    ImageProviderError,
    ImageRateLimitError,
    ImageTimeoutError,
)
from .types import ImageResult


class BaseImageProvider(ABC):
    """Abstract base class for all image providers."""

    def __init__(
        self,
        model: str,
        timeout_seconds: float = 120.0,
        est_cost_usd: float | None = None,
    ):
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.est_cost_usd = est_cost_usd

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """The identifier of this provider (e.g. 'mock', 'openai')."""
        pass

    @abstractmethod
    def _execute_generate(
        self, prompt: str, size: str, quality: str
    ) -> tuple[bytes, int | None, int | None, int | None]:
        """Provider-specific image generation call.

        Returns:
            Tuple of (image_bytes, input_tokens, output_tokens, total_tokens)
        """
        pass

    def generate(self, prompt: str, size: str, quality: str) -> ImageResult:
        """Generate an image from a prompt with the given size and quality.

        Returns:
            ImageResult containing the image bytes and call metadata.
        """
        start = time.perf_counter()
        image_bytes, in_tokens, out_tokens, tot_tokens = self._execute_generate(
            prompt=prompt, size=size, quality=quality
        )
        latency = time.perf_counter() - start

        return ImageResult(
            image_bytes=image_bytes,
            provider=self.provider_name,
            model=self.model,
            size=size,
            quality=quality,
            latency_seconds=round(latency, 4),
            input_tokens=in_tokens,
            output_tokens=out_tokens,
            total_tokens=tot_tokens,
            est_cost_usd=self.est_cost_usd,
        )
