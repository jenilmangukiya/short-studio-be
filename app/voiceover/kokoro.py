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

# Spoken expansions for abbreviations and symbols that degrade Kokoro prosody/phonemization
_ABBREVIATION_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bvs\.?(?=\s|$)", flags=re.IGNORECASE), "versus"),
    (re.compile(r"\be\.g\.?(?=\s|,|$)", flags=re.IGNORECASE), "for example"),
    (re.compile(r"\bi\.e\.?(?=\s|,|$)", flags=re.IGNORECASE), "that is"),
    (re.compile(r"\betc\.?(?=\s|,|$)", flags=re.IGNORECASE), "et cetera"),
    (re.compile(r"\bDr\.(?=\s+[A-Z])"), "Doctor"),
    (re.compile(r"\bMr\.(?=\s+[A-Z])"), "Mister"),
    (re.compile(r"\bMrs\.(?=\s+[A-Z])"), "Missus"),
    (re.compile(r"\bMs\.(?=\s+[A-Z])"), "Miz"),
    (re.compile(r"\bSt\.(?=\s+[A-Z])"), "Saint"),
    (re.compile(r"°\s*F\b"), " degrees Fahrenheit"),
    (re.compile(r"°\s*C\b"), " degrees Celsius"),
    (re.compile(r"\bmph\b", flags=re.IGNORECASE), "miles per hour"),
    (re.compile(r"\bkm/h\b", flags=re.IGNORECASE), "kilometers per hour"),
    (re.compile(r"(?<=\d)\s*%"), " percent"),
)


def normalize_text_for_tts(text: str) -> str:
    """Normalize punctuation, numbers, and abbreviations for natural Kokoro TTS prosody.

    Specifically addresses Kokoro-82M / espeak-ng phonemization artifacts:
    - Comma-separated digits (e.g. '30,000') are stripped of commas ('30000') so espeak
      says 'thirty thousand' instead of 'thirty, zero zero zero'.
    - Decimal numbers (e.g. '2.5') are expanded to '2 point 5' so the period does not
      trigger a sentence-ending pause token.
    - Curly/straight double quotation marks ('“...”') are stripped so Kokoro does not
      insert a ~240ms prosodic stall on vocab quote tokens mid-sentence.
    - Unspaced em-dashes, en-dashes, ellipses, colons, and semicolons are converted into
      clean comma/period pauses so words are not stretched or clipped.
    - Common abbreviations ('vs.', 'Dr.', 'mph', '°F') are expanded into spoken English.
    """
    if not text or not text.strip():
        return ""

    t = text.strip()

    # 1. Normalize curly apostrophes to standard ASCII apostrophe so contractions stay intact
    t = t.replace("’", "'").replace("‘", "'")

    # 2. Strip double quotation marks (“ ” ") which cause unnatural ~240ms mid-phrase stalls in Kokoro
    t = t.replace("“", "").replace("”", "").replace('"', "")

    # 3. Strip commas inside formatted numbers (e.g., 30,000 -> 30000; 1,000,000 -> 1000000)
    t = re.sub(r"(?<=\d),(?=\d{3}\b)", "", t)

    # 4. Expand decimal points between digits (e.g., 2.5 -> 2 point 5) so '.' doesn't split the sentence
    t = re.sub(r"(?<=\d)\.(?=\d)", " point ", t)

    # 5. Expand common abbreviations and unit symbols
    for pattern, replacement in _ABBREVIATION_REPLACEMENTS:
        t = pattern.sub(replacement, t)

    # 6. Convert ellipses, em-dashes, en-dashes, and spaced hyphens into clean comma pauses
    t = re.sub(r"\.{2,}|…", ", ", t)
    t = re.sub(r"\s*[—–]\s*", ", ", t)
    t = re.sub(r"\s+-\s+", ", ", t)

    # 7. Convert semicolons and colons into natural spoken pauses
    t = re.sub(r";\s*", ". ", t)
    t = re.sub(r":\s*", ", ", t)

    # 8. Clean up duplicate punctuation and collapse whitespace
    t = re.sub(r",\s*,+", ", ", t)
    t = re.sub(r"([.!?])\s*,", r"\1", t)
    t = re.sub(r",\s*([.!?])", r"\1", t)
    t = re.sub(r"\s+([,.!?])", r"\1", t)
    t = re.sub(r"\s+", " ", t).strip()

    return t


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

        tts_text = normalize_text_for_tts(text) or text.strip()
        batches: list[str] = []

        try:
            # Measure how many internal ONNX inference calls/batches Kokoro will execute
            phonemes = kokoro.tokenizer.phonemize(tts_text, lang)
            batches = kokoro._split_phonemes(phonemes)
            self._last_inference_batches = max(1, len(batches))
        except Exception:
            self._last_inference_batches = 1

        try:
            import numpy as np

            if len(batches) > 1 and hasattr(kokoro, "_create_audio") and hasattr(kokoro, "get_voice_style"):
                # When a single call exceeds 510 phonemes, synthesize each phoneme batch with trim=False,
                # apply our controlled silence trimming per batch, and insert a natural 240ms sentence breath.
                voice_style = kokoro.get_voice_style(voice)
                batch_parts: list[np.ndarray] = []
                sample_rate = 24000
                for b_idx, batch_phonemes in enumerate(batches):
                    part_samples, sample_rate = kokoro._create_audio(
                        batch_phonemes, voice_style, float(speed)
                    )
                    b_terminal = bool(re.search(r"[.!?]\s*$", batch_phonemes.strip()))
                    trimmed_part = trim_silence_samples(
                        part_samples, sample_rate, is_terminal=b_terminal
                    )
                    batch_parts.append(trimmed_part)
                    if b_idx < len(batches) - 1:
                        gap_sec = 0.24 if b_terminal else 0.09
                        batch_parts.append(np.zeros(int(gap_sec * sample_rate), dtype=np.float32))
                samples = np.concatenate(batch_parts) if batch_parts else np.array([], dtype=np.float32)
            else:
                # Pass trim=False so kokoro_onnx does not strip all trailing/leading silence via librosa.trim
                # before our own trim_silence_samples preserves natural 35ms lead and 110ms/80ms trail pads.
                try:
                    samples, sample_rate = kokoro.create(
                        text=tts_text,
                        voice=voice,
                        speed=float(speed),
                        lang=lang,
                        trim=False,
                    )
                except TypeError:
                    samples, sample_rate = kokoro.create(
                        text=tts_text,
                        voice=voice,
                        speed=float(speed),
                        lang=lang,
                    )
                is_terminal = bool(re.search(r"[.!?]\s*$", tts_text.strip()))
                samples = trim_silence_samples(samples, sample_rate, is_terminal=is_terminal)
        except Exception as exc:
            raise VoiceoverProviderError(
                f"Kokoro audio generation failed for voice '{voice}': {exc}"
            ) from exc

        if len(samples) == 0 or sample_rate <= 0:
            raise VoiceoverProviderError("Kokoro produced empty audio samples")

        actual_duration = round(len(samples) / float(sample_rate), 3)

        # Convert numpy audio samples into uncompressed 24 kHz WAV format
        buf = io.BytesIO()
        sf.write(buf, samples, sample_rate, format="WAV", subtype="PCM_16")
        audio_bytes = buf.getvalue()

        return audio_bytes, sample_rate, actual_duration
