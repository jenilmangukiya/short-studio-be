"""Factory for voiceover generation providers."""

from app.config import settings

from .base import BaseVoiceoverProvider
from .exceptions import VoiceoverConfigError
from .kokoro import KokoroVoiceoverProvider
from .mock import MockVoiceoverProvider


def get_voiceover_provider(
    provider: str | None = None,
    model: str | None = None,
    timeout_seconds: float | None = None,
) -> BaseVoiceoverProvider:
    """Return an instance of the configured or specified voiceover provider.

    Args:
        provider: 'mock' or 'kokoro'. Defaults to settings.voiceover_provider.
        model: Optional model override.
        timeout_seconds: Optional timeout in seconds.
    """
    prov = (provider or getattr(settings, "voiceover_provider", "mock") or "mock").lower().strip()
    timeout = timeout_seconds or getattr(settings, "voiceover_timeout_seconds", 60.0)

    if prov == "mock":
        return MockVoiceoverProvider(
            model=model or "mock-kokoro-model",
            timeout_seconds=timeout,
        )

    if prov == "kokoro":
        model_path = getattr(settings, "kokoro_model_path", "./storage/models/kokoro/kokoro-v1.0.onnx")
        voices_path = getattr(settings, "kokoro_voices_path", "./storage/models/kokoro/voices-v1.0.bin")
        return KokoroVoiceoverProvider(
            model_path=model_path,
            voices_path=voices_path,
            model=model or "kokoro-82m-v1.0",
            timeout_seconds=timeout,
        )

    raise VoiceoverConfigError(
        f"Unsupported voiceover provider '{prov}'. Supported providers are: mock, kokoro"
    )
