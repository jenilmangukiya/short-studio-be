"""Kokoro-82M ONNX voiceover generation provider."""

import io
from pathlib import Path
import re
from typing import Any

from .base import BaseVoiceoverProvider
from .exceptions import VoiceoverConfigError, VoiceoverProviderError
from .silence import trim_silence_samples

# Module-level cache for the loaded Kokoro model instance so it persists in memory
_KOKORO_CACHE: dict[tuple[str, str], Any] = {}


def resolve_kokoro_language(voice: str) -> str:
    """Infer the ISO language code expected by Kokoro from the voice prefix."""
    v = voice.lower().strip()
    if v.startswith(("bf_", "bm_")):
        return "en-gb"
    if v.startswith(("ef_", "em_")):
        return "es"
    if v.startswith(("ff_", "fm_")):
        return "fr"
    if v.startswith(("if_", "im_")):
        return "it"
    if v.startswith(("pf_", "pm_")):
        return "pt-br"
    if v.startswith(("jf_", "jm_")):
        return "ja"
    if v.startswith(("zf_", "zm_")):
        return "zh"
    return "en-us"


class KokoroVoiceoverProvider(BaseVoiceoverProvider):
    """Voiceover provider running Kokoro-82M locally via ONNX Runtime."""

    def __init__(
        self,
        model_path: str | Path,
        voices_path: str | Path,
        model: str = "kokoro-82m-v1.0",
        timeout_seconds: float = 60.0,
    ):
        super().__init__(
            model=model,
            timeout_seconds=timeout_seconds,
        )
        self.model_path = Path(model_path)
        self.voices_path = Path(voices_path)

    @property
    def provider_name(self) -> str:
        return "kokoro"

    def _get_kokoro_instance(self) -> Any:
        """Retrieve or initialize the cached Kokoro ONNX model instance."""
        cache_key = (str(self.model_path.resolve()), str(self.voices_path.resolve()))
        if cache_key in _KOKORO_CACHE:
            return _KOKORO_CACHE[cache_key]

        try:
            from kokoro_onnx import Kokoro
        except ImportError as exc:
            raise VoiceoverConfigError(
                f"kokoro-onnx is not installed: {exc}. Run: pip install kokoro-onnx soundfile"
            ) from exc

        if not self.model_path.is_file():
            raise VoiceoverConfigError(
                f"Kokoro ONNX model file not found at '{self.model_path}'. "
                "Download model assets using: python scripts/download_kokoro_model.py"
            )

        if not self.voices_path.is_file():
            raise VoiceoverConfigError(
                f"Kokoro voices file not found at '{self.voices_path}'. "
                "Download model assets using: python scripts/download_kokoro_model.py"
            )

        try:
            instance = Kokoro(
                model_path=str(self.model_path),
                voices_path=str(self.voices_path),
            )
            _KOKORO_CACHE[cache_key] = instance
            return instance
        except Exception as exc:
            raise VoiceoverConfigError(
                f"Failed to load Kokoro ONNX model from '{self.model_path}': {exc}"
            ) from exc

    def _execute_generate(
        self, text: str, voice: str, speed: float, target_duration: float | None = None
    ) -> tuple[bytes, int, float]:
        kokoro = self._get_kokoro_instance()
        lang = resolve_kokoro_language(voice)

        try:
            import soundfile as sf
        except ImportError as exc:
            raise VoiceoverConfigError(
                f"soundfile is not installed: {exc}. Run: pip install soundfile"
            ) from exc

        try:
            samples, sample_rate = kokoro.create(
                text=text,
                voice=voice,
                speed=float(speed),
                lang=lang,
            )
        except Exception as exc:
            raise VoiceoverProviderError(
                f"Kokoro audio generation failed for voice '{voice}': {exc}"
            ) from exc

        if len(samples) == 0 or sample_rate <= 0:
            raise VoiceoverProviderError("Kokoro produced empty audio samples")

        # Trim excessive silence while preserving natural breath headroom and phrase pauses
        is_terminal = bool(re.search(r"[.!?]\s*$", text.strip()))
        samples = trim_silence_samples(samples, sample_rate, is_terminal=is_terminal)

        actual_duration = round(len(samples) / float(sample_rate), 3)

        # Convert numpy audio samples into uncompressed 24 kHz WAV format
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        audio_bytes = buf.getvalue()

        return audio_bytes, sample_rate, actual_duration
