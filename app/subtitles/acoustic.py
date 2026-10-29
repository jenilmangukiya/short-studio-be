"""Acoustic forced alignment and waveform subtitle provider.

Uses local Vosk (Kaldi HMM-TDNN) acoustic forced alignment against the continuous
narration WAV to produce exact word-level timestamps in seconds relative to the
start of the continuous audio track.

Preserves full backward compatibility with legacy per-scene audio projects via
scene waveform envelope alignment when a continuous narration WAV is not present.
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


def _align_scene_audio(
    audio_path: Path | None,
    narration: str,
    duration_fallback: float,
    scene_offset: float,
) -> tuple[float, list[WordTimestamp]]:
    """Analyze individual scene audio waveform (legacy fallback) and align narration words."""
    clean_narration = narration.replace("’", "'").replace("‘", "'")
    tokens = re.findall(r"\b[\w'-]+\b", clean_narration)
    if not tokens:
        return duration_fallback, []

    weights = _compute_word_weights(tokens)
    total_weight = sum(weights) or 1.0

    onset_rel = 0.04
    offset_rel = max(0.2, duration_fallback - 0.08)
    actual_duration = duration_fallback
    smoothed_rms = None
    frame_ms = 10

    if audio_path and audio_path.is_file():
        try:
            import soundfile as sf

            data, sr = sf.read(str(audio_path))
            if len(data) > 0 and sr > 0:
                actual_duration = round(len(data) / float(sr), 3)

                if data.ndim > 1:
                    data = np.mean(data, axis=1)

                frame_len = int(sr * (frame_ms / 1000.0))
                num_frames = len(data) // frame_len

                if num_frames > 4:
                    frames = data[: num_frames * frame_len].reshape(num_frames, frame_len)
                    rms = np.sqrt(np.mean(frames**2, axis=1))

                    p75 = np.percentile(rms, 75)
                    thresh = max(0.004, p75 * 0.12)
                    speech_frames = np.where(rms > thresh)[0]

                    if len(speech_frames) > 0:
                        onset_rel = max(0.0, float(speech_frames[0] * (frame_ms / 1000.0)))
                        offset_rel = min(
                            actual_duration,
                            float((speech_frames[-1] + 1) * (frame_ms / 1000.0)),
                        )

                    kernel = np.ones(3) / 3.0
                    smoothed_rms = np.convolve(rms, kernel, mode="same")
        except Exception:
            pass

    active_speech_duration = max(0.15, offset_rel - onset_rel)
    words: list[WordTimestamp] = []

    curr_rel = onset_rel
    total_tokens = len(tokens)

    for i, (token, wt) in enumerate(zip(tokens, weights)):
        if i == total_tokens - 1:
            next_rel = max(curr_rel + 0.06, offset_rel)
        else:
            raw_dur = active_speech_duration * (wt / total_weight)
            next_rel = curr_rel + raw_dur

            if smoothed_rms is not None:
                target_frame = int(next_rel / (frame_ms / 1000.0))
                search_window = 4
                f_start = max(0, target_frame - search_window)
                f_end = min(len(smoothed_rms), target_frame + search_window + 1)
                if f_end > f_start:
                    min_idx = f_start + int(np.argmin(smoothed_rms[f_start:f_end]))
                    snapped_rel = min_idx * (frame_ms / 1000.0)
                    if snapped_rel > curr_rel + 0.06 and snapped_rel < offset_rel - 0.06:
                        next_rel = snapped_rel

        w_start = round(scene_offset + curr_rel, 3)
        w_end = round(scene_offset + min(actual_duration, next_rel), 3)

        if w_end <= w_start:
            w_end = round(w_start + 0.05, 3)

        words.append(
            WordTimestamp(
                text=token,
                start=w_start,
                end=w_end,
                confidence=0.98 if smoothed_rms is not None else 0.90,
            )
        )
        curr_rel = next_rel

    return actual_duration, words


class AcousticSubtitleProvider(BaseSubtitleProvider):
    """Subtitle provider using local Vosk forced alignment on continuous narration WAV,
    with compatible fallback to scene waveform envelope alignment for legacy scene audio.
    """

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
        if continuous_audio_path is not None:
            if not Path(continuous_audio_path).is_file():
                raise SubtitleError(
                    f"Continuous narration WAV not found at '{continuous_audio_path}'"
                )
            return self._generate_from_continuous_audio(
                scenes=scenes,
                continuous_audio_path=Path(continuous_audio_path),
            )

        start_time = time.perf_counter()
        segments: list[SubtitleSegment] = []
        total_words_count = 0
        cumulative_offset = 0.0
        overall_sources: set[str] = set()

        for scene in scenes:
            scene_idx = int(scene.get("index", len(segments) + 1))
            narration = str(scene.get("narration") or "").strip()
            est_duration = float(scene.get("duration_seconds", 0.0) or 0.0)
            if est_duration <= 0.0:
                est_duration = 2.5

            audio_file_rel = scene.get("file")
            audio_abs_path = None
            if audio_file_rel and storage_root:
                audio_abs_path = Path(storage_root) / audio_file_rel

            has_scene_file = bool(audio_abs_path and audio_abs_path.is_file())
            actual_duration, words = _align_scene_audio(
                audio_path=audio_abs_path,
                narration=narration,
                duration_fallback=est_duration,
                scene_offset=cumulative_offset,
            )
            seg_source = "scene_waveform" if has_scene_file else "estimated"
            overall_sources.add(seg_source)

            seg_start = round(cumulative_offset, 3)
            seg_end = round(cumulative_offset + actual_duration, 3)

            segments.append(
                SubtitleSegment(
                    index=scene_idx,
                    text=narration,
                    start=seg_start,
                    end=seg_end,
                    words=words,
                    timing_source=seg_source,
                )
            )

            total_words_count += len(words)
            cumulative_offset += actual_duration

        latency = round(time.perf_counter() - start_time, 4)
        resolved_source = (
            "scene_waveform"
            if overall_sources == {"scene_waveform"}
            else ("estimated" if "estimated" in overall_sources else "scene_waveform")
        )

        return SubtitleResult(
            segments=segments,
            total_words=total_words_count,
            total_duration_seconds=round(cumulative_offset, 3),
            provider=self.provider_name,
            model="acoustic-waveform-aligner-v1",
            latency_seconds=latency,
            timing_source=resolved_source,
        )
