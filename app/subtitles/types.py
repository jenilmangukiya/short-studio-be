"""Types and data structures for the subtitles layer in Shorts Studio."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import re
from typing import Any


@dataclass
class WordTimestamp:
    """Word-level timing information synchronized with the video voiceover.

    Attributes:
        text: Cleaned word string (e.g. 'Why', 'airplane').
        start: Start timestamp in seconds from the start of the video.
        end: End timestamp in seconds from the start of the video.
        confidence: Alignment confidence score (0.0 to 1.0).
    """

    text: str
    start: float
    end: float
    confidence: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "confidence": round(self.confidence, 2),
        }


@dataclass
class SubtitleSegment:
    """Segment / scene-level subtitle container containing words.

    Attributes:
        index: Scene index (1-based).
        text: Full narration text for this scene/segment.
        start: Start timestamp of the scene in seconds from video start.
        end: End timestamp of the scene in seconds from video start.
        words: List of WordTimestamp objects within this segment.
        timing_source: How segment/word timings were derived ('forced_alignment', 'scene_waveform', 'estimated').
    """

    index: int
    text: str
    start: float
    end: float
    words: list[WordTimestamp] = field(default_factory=list)
    timing_source: str = "scene_waveform"

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "timing_source": self.timing_source,
            "words": [w.to_dict() for w in self.words],
        }


def _format_timestamp_srt(seconds: float) -> str:
    """Format seconds into SubRip SRT timestamp format HH:MM:SS,mmm."""
    ms = int(round(seconds * 1000))
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    secs = ms // 1000
    ms %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    """Format seconds into WebVTT timestamp format HH:MM:SS.mmm."""
    ms = int(round(seconds * 1000))
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    secs = ms // 1000
    ms %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def _format_timestamp_ass(seconds: float) -> str:
    """Format seconds into ASS timestamp format H:MM:SS.cs."""
    cs = max(0, int(round(seconds * 100)))
    hours = cs // 360000
    cs %= 360000
    minutes = cs // 6000
    cs %= 6000
    secs = cs // 100
    cs %= 100
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{cs:02d}"


def _get_chunk_sizes(total: int) -> list[int]:
    """Determine chunk partition sizes so each chunk has 3-5 words."""
    if total <= 0:
        return []
    if total <= 5:
        return [total]
    sizes: list[int] = []
    rem = total
    while rem > 0:
        if rem <= 5:
            sizes.append(rem)
            break
        if rem == 6:
            sizes.extend([3, 3])
            break
        if rem == 7:
            sizes.extend([4, 3])
            break
        if rem == 8:
            sizes.extend([4, 4])
            break
        sizes.append(4)
        rem -= 4
    return sizes


def _partition_words_into_captions(
    words: list[WordTimestamp],
    max_words: int = 4,
    max_chars: int = 28,
    pause_threshold: float = 0.28,
) -> list[list[WordTimestamp]]:
    """Group aligned words into readable caption chunks respecting punctuation, pauses, and character limits."""
    if not words:
        return []

    captions: list[list[WordTimestamp]] = []
    current_group: list[WordTimestamp] = []
    current_chars = 0

    for w in words:
        word_len = len(w.text)
        added_chars = word_len if not current_group else (word_len + 1)

        if current_group:
            prev_w = current_group[-1]
            gap = float(w.start) - float(prev_w.end)
            prev_has_terminal_punct = bool(re.search(r"[.!?;:—–]$", prev_w.text))
            prev_has_comma = bool(re.search(r"[,]$", prev_w.text))

            should_break = (
                len(current_group) >= max_words
                or (current_chars + added_chars > max_chars and len(current_group) >= 2)
                or gap >= pause_threshold
                or prev_has_terminal_punct
                or (prev_has_comma and len(current_group) >= 2)
            )
            if should_break:
                captions.append(current_group)
                current_group = []
                current_chars = 0
                added_chars = word_len

        current_group.append(w)
        current_chars += added_chars

    if current_group:
        if (
            len(current_group) == 1
            and captions
            and len(captions[-1]) < max_words + 1
            and (float(current_group[0].start) - float(captions[-1][-1].end) < pause_threshold)
            and not re.search(r"[.!?]$", captions[-1][-1].text)
            and sum(len(x.text) for x in captions[-1]) + len(current_group[0].text) + len(captions[-1]) <= max_chars + 4
        ):
            captions[-1].append(current_group[0])
        else:
            captions.append(current_group)

    return captions


@dataclass
class SubtitleResult:
    """Standardized output container for subtitle generation."""

    segments: list[SubtitleSegment]
    total_words: int
    total_duration_seconds: float
    provider: str
    model: str
    latency_seconds: float = 0.0
    timing_source: str = "scene_waveform"
    alignment_hash: str | None = None
    reused: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "total_words": self.total_words,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "format": "word_level",
            "timing_source": self.timing_source,
            "provider": self.provider,
            "model": self.model,
            "alignment_hash": self.alignment_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], reused: bool = True) -> SubtitleResult:
        """Reconstruct a SubtitleResult from a cached dictionary."""
        segments: list[SubtitleSegment] = []
        for s_dict in data.get("segments", []):
            words = [
                WordTimestamp(
                    text=str(w["text"]),
                    start=float(w["start"]),
                    end=float(w["end"]),
                    confidence=float(w.get("confidence", 1.0)),
                )
                for w in s_dict.get("words", [])
            ]
            segments.append(
                SubtitleSegment(
                    index=int(s_dict["index"]),
                    text=str(s_dict["text"]),
                    start=float(s_dict["start"]),
                    end=float(s_dict["end"]),
                    words=words,
                    timing_source=str(s_dict.get("timing_source", data.get("timing_source", "scene_waveform"))),
                )
            )
        return cls(
            segments=segments,
            total_words=int(data.get("total_words", sum(len(s.words) for s in segments))),
            total_duration_seconds=float(data.get("total_duration_seconds", 0.0)),
            provider=str(data.get("provider", "acoustic")),
            model=str(data.get("model", "default")),
            latency_seconds=0.0,
            timing_source=str(data.get("timing_source", "scene_waveform")),
            alignment_hash=data.get("alignment_hash"),
            reused=reused,
        )

    def to_srt(self) -> str:
        """Export subtitle segments as SubRip (.srt) caption text."""
        blocks: list[str] = []
        for i, seg in enumerate(self.segments, 1):
            t_start = _format_timestamp_srt(seg.start)
            t_end = _format_timestamp_srt(seg.end)
            blocks.append(f"{i}\n{t_start} --> {t_end}\n{seg.text.strip()}")
        return "\n\n".join(blocks) + "\n"

    def to_vtt(self) -> str:
        """Export subtitle segments as WebVTT (.vtt) caption text."""
        lines: list[str] = ["WEBVTT\n"]
        for i, seg in enumerate(self.segments, 1):
            t_start = _format_timestamp_vtt(seg.start)
            t_end = _format_timestamp_vtt(seg.end)
            lines.append(f"{i}\n{t_start} --> {t_end}\n{seg.text.strip()}\n")
        return "\n".join(lines)

    def to_ass(self) -> str:
        """Export subtitle segments as ASS (.ass) subtitle text with word-level highlight animation."""
        header = [
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 1080",
            "PlayResY: 1920",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            r"Style: Default,Arial,64,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,40,40,360,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ]
        lines = list(header)

        all_words: list[WordTimestamp] = []
        for seg in self.segments:
            all_words.extend(seg.words)

        if not all_words:
            return chr(10).join(lines) + chr(10)

        caption_groups = _partition_words_into_captions(all_words)

        for g_idx, chunk_words in enumerate(caption_groups):
            if not chunk_words:
                continue

            next_group_start: float | None = None
            if g_idx + 1 < len(caption_groups) and caption_groups[g_idx + 1]:
                next_group_start = float(caption_groups[g_idx + 1][0].start)

            last_w_end = float(chunk_words[-1].end)
            desired_end = max(float(chunk_words[0].start) + 0.45, last_w_end + 0.25)
            if next_group_start is not None:
                chunk_end = min(desired_end, next_group_start)
                if chunk_end < last_w_end:
                    chunk_end = min(last_w_end, next_group_start)
            else:
                chunk_end = min(
                    desired_end,
                    max(last_w_end + 0.25, float(self.total_duration_seconds or last_w_end + 0.25)),
                )

            for w_idx, w in enumerate(chunk_words):
                w_start = float(w.start)
                if w_idx + 1 < len(chunk_words):
                    next_w_start = float(chunk_words[w_idx + 1].start)
                    w_end = max(float(w.end), next_w_start)
                else:
                    w_end = max(float(w.end), chunk_end)

                if next_group_start is not None and w_idx == len(chunk_words) - 1:
                    w_end = min(w_end, next_group_start)

                if w_end <= w_start:
                    w_end = w_start + 0.08

                parts = []
                for idx, cw in enumerate(chunk_words):
                    txt = cw.text
                    if idx == w_idx:
                        parts.append(r"{\c&H0028F5FC&}" + txt + r"{\c&H00FFFFFF&}")
                    else:
                        parts.append(txt)

                line_text = " ".join(parts)
                t1 = _format_timestamp_ass(w_start)
                t2 = _format_timestamp_ass(w_end)
                if t1 == t2:
                    t2 = _format_timestamp_ass(w_start + 0.08)
                lines.append(f"Dialogue: 0,{t1},{t2},Default,,0,0,0,,{line_text}")

        return chr(10).join(lines) + chr(10)
