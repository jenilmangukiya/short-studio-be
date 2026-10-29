"""Acoustic forced alignment subtitle provider.

Uses local Vosk (Kaldi HMM-TDNN) acoustic forced alignment against the continuous
narration WAV to produce exact word-level timestamps in seconds relative to the
start of the continuous audio track.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any

import numpy as np

from app.config import settings

from .base import BaseSubtitleProvider, SubtitleConfigError, SubtitleError
from .types import SubtitleResult, SubtitleSegment, WordTimestamp

# Module-level cache for loaded Vosk Model instances so the ~40MB acoustic model
# is loaded once per worker process and reused across requests.
_VOSK_MODEL_CACHE: dict[str, Any] = {}


def _extract_display_tokens(text: str) -> list[str]:
    """Extract spoken words while preserving attached trailing punctuation and curly apostrophes."""
    normalized = text.replace("’", "'").replace("‘", "'")
    tokens = re.findall(r"\b[\w'-]+[,.;:!?—–]*", normalized.strip())
    return [t for t in tokens if re.search(r"[\w]", t)]


def _normalize_word(token: str) -> str:
    """Normalize a token to lowercase alphanumeric characters for phonetic/lexical matching."""
    normalized = token.replace("’", "'").replace("‘", "'")
    return re.sub(r"[^a-z0-9']", "", normalized.lower()).strip("'")


def _word_similarity(script_norm: str, rec_norm: str) -> float:
    """Compute similarity score [0.0, 1.0] between a script word and a recognized acoustic word."""
    if not script_norm or not rec_norm:
        return 0.0
    if script_norm == rec_norm:
        return 1.0
    # Plural / possessive / morphological prefix match (e.g. 'yawns' vs 'yawn', 'einstein's' vs 'einstein')
    if script_norm.startswith(rec_norm) or rec_norm.startswith(script_norm):
        shorter = min(len(script_norm), len(rec_norm))
        longer = max(len(script_norm), len(rec_norm))
        if shorter >= 3 and (longer - shorter) <= 3:
            return 0.92

    # Character-level similarity for homophones or numbers (e.g. 'cue' vs 'queue')
    if script_norm[0] == rec_norm[0]:
        common = sum(1 for a, b in zip(script_norm, rec_norm) if a == b)
        ratio = common / float(max(len(script_norm), len(rec_norm)))
        if ratio >= 0.5:
            return 0.65 + 0.25 * ratio

    homophones = {
        frozenset({"cue", "queue"}),
        frozenset({"to", "too", "two"}),
        frozenset({"their", "there", "they're"}),
        frozenset({"its", "it's"}),
        frozenset({"your", "you're"}),
    }
    if frozenset({script_norm, rec_norm}) in homophones:
        return 0.95

    return 0.0


def _compute_word_weights(tokens: list[str]) -> list[float]:
    """Estimate relative spoken duration weight for each word based on syllables and characters."""
    weights: list[float] = []
    for token in tokens:
        clean = re.sub(r"[^a-zA-Z0-9]", "", token.lower())
        if not clean:
            weights.append(1.0)
            continue
        vowels = len(re.findall(r"[aeiouy]+", clean))
        syllables = max(1, vowels)
        weight = syllables * 1.6 + len(clean) * 0.25
        weights.append(weight)
    return weights


def _get_vosk_model(model_path: str | Path | None = None) -> Any | None:
    """Resolve and load the local Vosk acoustic model, caching in memory."""
    try:
        import vosk
    except ImportError:
        return None

    vosk.SetLogLevel(-1)

    candidates: list[Path] = []
    if model_path:
        candidates.append(Path(model_path))

    cfg_path = getattr(settings, "vosk_model_path", None)
    if cfg_path:
        candidates.append(Path(cfg_path))

    storage_root = getattr(settings, "resolved_storage_dir", Path(settings.storage_dir).resolve())
    candidates.extend([
        storage_root / "models" / "vosk",
        storage_root / "models" / "vosk-model-small-en-us-0.15",
        Path.home() / ".cache" / "vosk" / "vosk-model-small-en-us-0.15",
    ])

    resolved_dir: Path | None = None
    for cand in candidates:
        if cand.is_dir() and ((cand / "am").is_dir() or (cand / "conf").is_dir()):
            resolved_dir = cand.resolve()
            break

    if resolved_dir is None:
        return None

    key = str(resolved_dir)
    if key not in _VOSK_MODEL_CACHE:
        _VOSK_MODEL_CACHE[key] = vosk.Model(key)
    return _VOSK_MODEL_CACHE[key]


def align_script_with_vosk(
    audio_path: Path,
    script_tokens: list[str],
    model_path: str | Path | None = None,
) -> tuple[float, list[WordTimestamp]]:
    """Run local Vosk acoustic alignment on a continuous WAV file and map to exact script tokens."""
    if not audio_path or not audio_path.is_file():
        raise SubtitleError(f"Continuous narration audio file does not exist: {audio_path}")

    try:
        import soundfile as sf
        data, sr = sf.read(str(audio_path), dtype="int16")
    except Exception as exc:
        raise SubtitleError(f"Failed to read narration WAV '{audio_path}': {exc}") from exc

    if len(data) == 0 or sr <= 0:
        raise SubtitleError(f"Narration WAV '{audio_path}' is empty or has invalid sample rate")

    if data.ndim > 1:
        data = data[:, 0]

    actual_duration = round(len(data) / float(sr), 3)
    if not script_tokens:
        return actual_duration, []

    vosk_model = _get_vosk_model(model_path)
    if vosk_model is None:
        raise SubtitleError("Local Vosk acoustic model is not available for forced alignment")

    import vosk

    rec = vosk.KaldiRecognizer(vosk_model, float(sr))
    rec.SetWords(True)

    pcm_bytes = data.tobytes()
    chunk_size = sr * 2 * 4  # 4-second chunks of 16-bit mono PCM
    recognized: list[dict[str, Any]] = []

    for offset in range(0, len(pcm_bytes), chunk_size):
        chunk = pcm_bytes[offset : offset + chunk_size]
        if rec.AcceptWaveform(chunk):
            part = json.loads(rec.Result())
            if isinstance(part.get("result"), list):
                recognized.extend(part["result"])

    final_part = json.loads(rec.FinalResult())
    if isinstance(final_part.get("result"), list):
        recognized.extend(final_part["result"])

    if not recognized:
        raise SubtitleError("Acoustic forced alignment produced zero recognized words from audio")

    return actual_duration, map_recognized_words_to_script(
        script_tokens=script_tokens,
        recognized_words=recognized,
        audio_duration=actual_duration,
    )


def map_recognized_words_to_script(
    script_tokens: list[str],
    recognized_words: list[dict[str, Any]],
    audio_duration: float,
) -> list[WordTimestamp]:
    """Align ground-truth script tokens to acoustic recognized word timestamps using dynamic programming."""
    n_script = len(script_tokens)
    n_rec = len(recognized_words)
    if n_script == 0:
        return []
    if n_rec == 0:
        raise SubtitleError("Cannot align script tokens to empty recognized words list")

    script_norms = [_normalize_word(t) for t in script_tokens]
    rec_norms = [_normalize_word(str(r.get("word", ""))) for r in recognized_words]

    gap_penalty = -0.35
    dp = np.zeros((n_script + 1, n_rec + 1), dtype=np.float32)
    back = np.zeros((n_script + 1, n_rec + 1), dtype=np.int8)

    for i in range(1, n_script + 1):
        dp[i, 0] = i * gap_penalty
        back[i, 0] = 2
    for j in range(1, n_rec + 1):
        dp[0, j] = j * gap_penalty
        back[0, j] = 3

    for i in range(1, n_script + 1):
        s_norm = script_norms[i - 1]
        for j in range(1, n_rec + 1):
            r_norm = rec_norms[j - 1]
            sim = _word_similarity(s_norm, r_norm)
            match_score = sim if sim > 0.0 else -0.25

            score_diag = dp[i - 1, j - 1] + match_score
            score_up = dp[i - 1, j] + gap_penalty
            score_left = dp[i, j - 1] + gap_penalty

            if score_diag >= score_up and score_diag >= score_left:
                dp[i, j] = score_diag
                back[i, j] = 1
            elif score_up >= score_left:
                dp[i, j] = score_up
                back[i, j] = 2
            else:
                dp[i, j] = score_left
                back[i, j] = 3

    matched_rec_idx: list[int | None] = [None] * n_script
    i, j = n_script, n_rec
    while i > 0 and j > 0:
        move = int(back[i, j])
        if move == 1:
            sim = _word_similarity(script_norms[i - 1], rec_norms[j - 1])
            if sim >= 0.5 or (n_script == n_rec):
                matched_rec_idx[i - 1] = j - 1
            i -= 1
            j -= 1
        elif move == 2:
            i -= 1
        else:
            j -= 1

    # Fill isolated 1-to-1 gaps between matched anchors
    for idx in range(n_script):
        if matched_rec_idx[idx] is None:
            prev_rec = matched_rec_idx[idx - 1] if idx > 0 else -1
            next_rec = matched_rec_idx[idx + 1] if idx + 1 < n_script else n_rec
            if prev_rec is not None and next_rec is not None and (next_rec - prev_rec) == 2:
                matched_rec_idx[idx] = prev_rec + 1

    anchored_count = sum(1 for m in matched_rec_idx if m is not None)
    if anchored_count < max(1, int(n_script * 0.4)):
        raise SubtitleError(
            f"Acoustic alignment confidence too low: only {anchored_count}/{n_script} words matched"
        )

    weights = _compute_word_weights(script_tokens)
    result_words: list[WordTimestamp | None] = [None] * n_script

    for idx, r_idx in enumerate(matched_rec_idx):
        if r_idx is not None:
            rw = recognized_words[r_idx]
            w_start = max(0.0, float(rw.get("start", 0.0)))
            w_end = min(audio_duration, max(w_start + 0.04, float(rw.get("end", w_start + 0.08))))
            conf = float(rw.get("conf", 0.95))
            result_words[idx] = WordTimestamp(
                text=script_tokens[idx],
                start=round(w_start, 3),
                end=round(w_end, 3),
                confidence=round(conf, 2),
            )

    # Interpolate any unanchored span [span_start .. span_end] between acoustic anchors
    # Also handle multi-word acoustic spans (e.g. "2015" spoken as "twenty fifteen" across 2 recognized tokens)
    idx = 0
    while idx < n_script:
        if result_words[idx] is not None:
            idx += 1
            continue

        span_start = idx
        while idx < n_script and result_words[idx] is None:
            idx += 1
        span_end = idx - 1

        prev_rec_idx = matched_rec_idx[span_start - 1] if span_start > 0 else None
        next_rec_idx = matched_rec_idx[idx] if idx < n_script else None

        # If a single unanchored script token (like "2015" or "LIGO") sits between two matched anchors
        # that have 1..3 unrecognized acoustic words between them, span across those exact acoustic words!
        if (
            span_start == span_end
            and prev_rec_idx is not None
            and next_rec_idx is not None
            and next_rec_idx > prev_rec_idx + 1
        ):
            sub_recs = recognized_words[prev_rec_idx + 1 : next_rec_idx]
            w_start = max(0.0, float(sub_recs[0].get("start", 0.0)))
            w_end = min(audio_duration, max(w_start + 0.04, float(sub_recs[-1].get("end", w_start + 0.1))))
            result_words[span_start] = WordTimestamp(
                text=script_tokens[span_start],
                start=round(w_start, 3),
                end=round(w_end, 3),
                confidence=0.92,
            )
            continue

        if span_start > 0 and result_words[span_start - 1] is not None:
            t_left = result_words[span_start - 1].end
        else:
            first_rec_start = float(recognized_words[0].get("start", 0.03))
            t_left = max(0.0, min(0.05, first_rec_start))

        if idx < n_script and result_words[idx] is not None:
            t_right = result_words[idx].start
        else:
            last_rec_end = float(recognized_words[-1].get("end", audio_duration))
            t_right = min(audio_duration, max(t_left + 0.1 * (span_end - span_start + 1), last_rec_end))

        span_count = span_end - span_start + 1
        if t_right <= t_left + 0.04 * span_count:
            t_right = min(audio_duration, t_left + 0.08 * span_count)
            if t_right <= t_left:
                t_left = max(0.0, t_right - 0.06 * span_count)

        span_weights = weights[span_start : span_end + 1]
        total_sw = sum(span_weights) or float(span_count)
        avail_dur = max(0.04 * span_count, t_right - t_left)

        curr_t = t_left
        for k, s_idx in enumerate(range(span_start, span_end + 1)):
            w_dur = avail_dur * (span_weights[k] / total_sw)
            w_end = t_right if k == span_count - 1 else (curr_t + w_dur)
            w_end = max(curr_t + 0.03, min(audio_duration, w_end))
            result_words[s_idx] = WordTimestamp(
                text=script_tokens[s_idx],
                start=round(curr_t, 3),
                end=round(w_end, 3),
                confidence=0.85,
            )
            curr_t = w_end

    final_words: list[WordTimestamp] = []
    prev_end = 0.0
    for w in result_words:
        assert w is not None
        s = max(prev_end, min(audio_duration, float(w.start)))
        e = max(s + 0.03, min(audio_duration, float(w.end)))
        if e > audio_duration:
            e = round(audio_duration, 3)
            s = max(0.0, min(s, round(e - 0.03, 3)))
        final_words.append(
            WordTimestamp(
                text=w.text,
                start=round(s, 3),
                end=round(e, 3),
                confidence=w.confidence,
            )
        )
        prev_end = final_words[-1].end

    return final_words


class AcousticSubtitleProvider(BaseSubtitleProvider):
    """Subtitle provider using local Vosk forced alignment on continuous narration WAV."""

    def __init__(
        self,
        model: str = "vosk-forced-aligner-v1",
        timeout_seconds: float = 30.0,
        vosk_model_path: str | Path | None = None,
    ):
        super().__init__(model=model, timeout_seconds=timeout_seconds)
        self.vosk_model_path = vosk_model_path

    @property
    def provider_name(self) -> str:
        return "acoustic"

    def _generate_from_continuous_audio(
        self,
        scenes: list[dict[str, Any]],
        continuous_audio_path: Path,
    ) -> SubtitleResult:
        """Align the full narration script directly against the continuous WAV asset."""
        start_time = time.perf_counter()

        scene_tokens_map: list[tuple[int, str, list[str]]] = []
        all_script_tokens: list[str] = []

        for idx_i, scene in enumerate(scenes):
            scene_idx = int(scene.get("index", idx_i + 1))
            narration = str(scene.get("narration") or "").strip()
            tokens = _extract_display_tokens(narration)
            scene_tokens_map.append((scene_idx, narration, tokens))
            all_script_tokens.extend(tokens)

        actual_duration, aligned_words = align_script_with_vosk(
            audio_path=continuous_audio_path,
            script_tokens=all_script_tokens,
            model_path=self.vosk_model_path,
        )

        segments: list[SubtitleSegment] = []
        word_cursor = 0

        for s_pos, (scene_idx, narration, tokens) in enumerate(scene_tokens_map):
            count = len(tokens)
            seg_words = aligned_words[word_cursor : word_cursor + count]
            word_cursor += count

            if seg_words:
                seg_start = 0.0 if s_pos == 0 else round(seg_words[0].start, 3)
                seg_end = (
                    round(actual_duration, 3)
                    if s_pos == len(scene_tokens_map) - 1
                    else round(seg_words[-1].end, 3)
                )
            else:
                prev_end = segments[-1].end if segments else 0.0
                seg_start = round(prev_end, 3)
                seg_end = round(min(actual_duration, prev_end + 0.1), 3)

            segments.append(
                SubtitleSegment(
                    index=scene_idx,
                    text=narration,
                    start=seg_start,
                    end=seg_end,
                    words=seg_words,
                    timing_source="forced_alignment",
                )
            )

        latency = round(time.perf_counter() - start_time, 4)
        return SubtitleResult(
            segments=segments,
            total_words=len(aligned_words),
            total_duration_seconds=round(actual_duration, 3),
            provider=self.provider_name,
            model=self.model,
            latency_seconds=latency,
            timing_source="forced_alignment",
        )

    def generate(
        self,
        scenes: list[dict[str, Any]],
        storage_root: Path | None = None,
        continuous_audio_path: Path | None = None,
        narration_script: str | None = None,
    ) -> SubtitleResult:
        if continuous_audio_path is None:
            raise SubtitleError(
                "Continuous narration WAV path is required for acoustic subtitle alignment"
            )
        if not Path(continuous_audio_path).is_file():
            raise SubtitleError(
                f"Continuous narration WAV not found at '{continuous_audio_path}'"
            )
        return self._generate_from_continuous_audio(
            scenes=scenes,
            continuous_audio_path=Path(continuous_audio_path),
        )
