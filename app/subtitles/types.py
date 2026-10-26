"""Types and data structures for the subtitles layer in Shorts Studio."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
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
    """

    index: int
    text: str
    start: float
    end: float
    words: list[WordTimestamp] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "text": self.text,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
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
@dataclass
class SubtitleResult:
    """Standardized output container for subtitle generation."""

    segments: list[SubtitleSegment]
    total_words: int
    total_duration_seconds: float
    provider: str
    model: str
    latency_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "segments": [s.to_dict() for s in self.segments],
            "total_words": self.total_words,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "format": "word_level",
        }

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

        for seg in self.segments:
            words = seg.words
            if not words:
                continue
            seg_end = float(seg.end)
            sizes = _get_chunk_sizes(len(words))
            offset = 0

            for c_idx, size in enumerate(sizes):
                chunk_words = words[offset : offset + size]
                is_last_chunk = c_idx == len(sizes) - 1
                offset += size

                last_w_end = float(chunk_words[-1].end)
                if not is_last_chunk and offset < len(words):
                    next_start = float(words[offset].start)
                    chunk_end = (
                        next_start
                        if (next_start - last_w_end <= 0.4)
                        else (last_w_end + 0.35)
                    )
                else:
                    chunk_end = (
                        max(last_w_end, min(seg_end, last_w_end + 0.35))
                        if seg_end > 0
                        else (last_w_end + 0.35)
                    )

                for w_idx, w in enumerate(chunk_words):
                    w_start = float(w.start)
                    if w_idx + 1 < len(chunk_words):
                        w_end = float(chunk_words[w_idx + 1].start)
                    else:
                        w_end = chunk_end

                    if w_end <= w_start:
                        w_end = w_start + 0.1

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
                    lines.append(f"Dialogue: 0,{t1},{t2},Default,,0,0,0,,{line_text}")

        return chr(10).join(lines) + chr(10)
