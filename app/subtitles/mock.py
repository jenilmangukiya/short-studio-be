"""Mock subtitle provider for testing and zero-cost local development."""

from __future__ import annotations

from pathlib import Path
import re
import time
from typing import Any

from .base import BaseSubtitleProvider
from .types import SubtitleResult, SubtitleSegment, WordTimestamp


class MockSubtitleProvider(BaseSubtitleProvider):
    """Generates synthetic word-level subtitle timing without reading audio files."""

    def __init__(self, model: str = "mock-subtitle-model", timeout_seconds: float = 30.0):
        super().__init__(model=model, timeout_seconds=timeout_seconds)

    @property
    def provider_name(self) -> str:
        return "mock"

    def generate(
        self,
        scenes: list[dict[str, Any]],
        storage_root: Path | None = None,
        continuous_audio_path: Path | None = None,
        narration_script: str | None = None,
    ) -> SubtitleResult:
        start_time = time.perf_counter()
        segments: list[SubtitleSegment] = []
        total_words_count = 0
        cumulative_start = 0.0

        for scene in scenes:
            scene_idx = int(scene.get("index", len(segments) + 1))
            narration = str(scene.get("narration") or "").strip()
            duration = float(scene.get("duration_seconds", 0.0) or 0.0)
            if duration <= 0.0:
                duration = 2.0  # reasonable fallback duration

            scene_end = cumulative_start + duration
            raw_tokens = re.findall(r"\b[\w'-]+\b", narration)

            words: list[WordTimestamp] = []
            if raw_tokens:
                total_words_count += len(raw_tokens)
                # Syllable / character weighted distribution
                weights: list[float] = []
                for w in raw_tokens:
                    clean = re.sub(r"[^a-zA-Z]", "", w.lower())
                    syllables = max(1, len(re.findall(r"[aeiouy]+", clean)))
                    weights.append(syllables * 1.5 + len(clean) * 0.3)
                total_weight = sum(weights) or 1.0

                w_start = cumulative_start + 0.04  # small speech onset
                usable_duration = max(0.2, duration - 0.1)

                for token, wt in zip(raw_tokens, weights):
                    w_dur = usable_duration * (wt / total_weight)
                    w_end = min(scene_end, w_start + w_dur)
                    words.append(
                        WordTimestamp(
                            text=token,
                            start=round(w_start, 3),
                            end=round(w_end, 3),
                            confidence=1.0,
                        )
                    )
                    w_start = w_end

            segments.append(
                SubtitleSegment(
                    index=scene_idx,
                    text=narration,
                    start=round(cumulative_start, 3),
                    end=round(scene_end, 3),
                    words=words,
                )
            )
            cumulative_start = scene_end

        latency = round(time.perf_counter() - start_time, 4)
        return SubtitleResult(
            segments=segments,
            total_words=total_words_count,
            total_duration_seconds=round(cumulative_start, 3),
            provider=self.provider_name,
            model=self.model,
            latency_seconds=latency,
        )
