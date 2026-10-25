"""Factory for image generation providers."""

from app.config import settings

from .base import BaseImageProvider
from .exceptions import ImageConfigError
from .mock import MockImageProvider
from .openai import OpenAIImageProvider


def get_image_provider(
    provider: str | None = None,
    model: str | None = None,
    timeout_seconds: float | None = None,
    est_cost_usd: float | None = None,
) -> BaseImageProvider:
    """Return an instance of the configured or specified image provider.

    Args:
        provider: 'mock' or 'openai'. Defaults to settings.image_provider.
        model: Optional model override. Defaults to provider settings.
        timeout_seconds: Optional timeout in seconds.
        est_cost_usd: Optional estimated cost per image in USD.
    """
    prov = (provider or settings.image_provider or "mock").lower().strip()
    timeout = timeout_seconds or settings.image_timeout_seconds
    cost = est_cost_usd if est_cost_usd is not None else settings.openai_image_est_cost_usd

    if prov == "mock":
        return MockImageProvider(
            model=model or "mock-image-model",
            timeout_seconds=timeout,
            est_cost_usd=None,
        )

    if prov == "openai":
        resolved_model = model or settings.openai_image_model
        return OpenAIImageProvider(
            model=resolved_model,
            api_key=settings.openai_image_api_key,
            timeout_seconds=timeout,
            est_cost_usd=cost,
        )

    raise ImageConfigError(
        f"Unsupported image provider '{prov}'. Supported providers are: mock, openai"
    )
