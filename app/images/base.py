"""Base interface and common logic for image generation providers."""

import inspect
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

    @property
    def accepts_reference_images(self) -> bool:
        """Explicitly detect whether this provider's _execute_generate method accepts reference_images."""
        if not hasattr(self, "_accepts_refs_cached"):
            try:
                sig = inspect.signature(self._execute_generate)
                params = sig.parameters
                self._accepts_refs_cached = "reference_images" in params or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
                )
            except Exception:
                self._accepts_refs_cached = False
        return self._accepts_refs_cached

    @abstractmethod
    def _execute_generate(
        self,
        prompt: str,
        size: str,
        quality: str,
        reference_images: list[bytes] | None = None,
    ) -> tuple[bytes, int | None, int | None, int | None]:
        """Provider-specific image generation call.

        When reference_images is supplied, performs reference-conditioned image edits.

        Returns:
            Tuple of (image_bytes, input_tokens, output_tokens, total_tokens)
        """
        pass

    def generate(
        self,
        prompt: str,
        size: str,
        quality: str,
        reference_images: list[bytes] | None = None,
    ) -> ImageResult:
        """Generate an image from a prompt with the given size, quality, and optional reference images.

        Uses explicit capability detection on _execute_generate rather than catching execution TypeErrors,
        ensuring internal TypeErrors from providers are never masked or retried.

        Returns:
            ImageResult containing the image bytes and call metadata.
        """
        start = time.perf_counter()

        if reference_images is not None:
            if not self.accepts_reference_images:
                raise ImageProviderError(
                    f"Provider '{self.provider_name}' does not support reference conditioning."
                )
            image_bytes, in_tokens, out_tokens, tot_tokens = self._execute_generate(
                prompt=prompt,
                size=size,
                quality=quality,
                reference_images=reference_images,
            )
        else:
            if self.accepts_reference_images:
                image_bytes, in_tokens, out_tokens, tot_tokens = self._execute_generate(
                    prompt=prompt,
                    size=size,
                    quality=quality,
                    reference_images=None,
                )
            else:
                image_bytes, in_tokens, out_tokens, tot_tokens = self._execute_generate(
                    prompt=prompt,
                    size=size,
                    quality=quality,
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
