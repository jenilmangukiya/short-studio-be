"""Comprehensive test suite for audio pacing requirements in Shorts Studio.

Verifies:
1. Exactly 0.75 seconds of initial silence before first spoken word.
2. Approximately 0.75 seconds of opening-hook silence after the hook.
3. Exactly 1.0 second of ending silence after the final spoken word.
4. Correct word timestamps starting after the initial silence.
5. Subtitles (SRT, VTT, ASS) do not display during artificial silences.
6. Scene boundaries and video durations include both start and ending silence.
7. No duplicated pauses when audio chunks are combined.
8. Existing legacy generation paths continue working with 0 padding when configured.
"""

import io
from pathlib import Path
import uuid
import wave
import numpy as np
import pytest
import soundfile as sf

from app.config import settings
from app.models import Video, VideoStep
from app.steps.voiceover import generate_voiceover_step
from app.subtitles import (
    SubtitleResult,
    SubtitleSegment,
    WordTimestamp,
    compute_aligned_scene_timings,
)
from app.voiceover import (
    BaseVoiceoverProvider,
    VoiceoverResult,
    collect_narration_script,
    compute_inter_chunk_gaps,
    concatenate_audio_segments,
    create_mock_wav,
    estimate_scene_timings,
    generate_continuous_audio,
    split_script_into_chunks,
)


class DummyVideo:
    def __init__(self, video_id=None, channel=None):
        self.id = video_id or uuid.uuid4()
        self.topic = "The Science of Audio Timing"
        self.channel = channel


class DummyStep:
    def __init__(self, name="voiceover", user_prompt=None, output=None, status="pending"):
        self.name = name
        self.user_prompt = user_prompt
        self.output = output
        self.status = status


class PredictableToneVoiceoverProvider(BaseVoiceoverProvider):
    """Voiceover provider producing non-zero tone audio for reliable silence boundary checks."""

    def __init__(self, sample_rate: int = 24000):
        super().__init__(model="predictable-tone-provider", timeout_seconds=10.0)
        self.sample_rate = sample_rate
        self.call_count = 0
        self.called_texts: list[str] = []

    @property
    def provider_name(self) -> str:
        return "predictable-tone-provider"

    def _execute_generate(
        self, text: str, voice: str, speed: float, target_duration: float | None = None
    ) -> tuple[bytes, int, float]:
        self.call_count += 1
        self.called_texts.append(text)
        words = text.split()
        dur = max(0.4, round(len(words) / 2.5, 3))
        n_samples = int(dur * self.sample_rate)

        # Tone with 0.3 amplitude so energy is easily distinguishable from silence zeros
        t = np.linspace(0, dur, n_samples, endpoint=False, dtype=np.float32)
        samples = 0.3 * np.sin(2 * np.pi * 440.0 * t)

        buf = io.BytesIO()
        sf.write(buf, samples, self.sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue(), self.sample_rate, dur


def _make_storyboard_prev(scenes_data: list[dict]) -> dict:
    return {
        "storyboard": {
            "storyboard": {
                "visual_style": "Modern Documentary",
                "scenes": scenes_data,
                "total_seconds": sum(s.get("est_seconds", 2.5) for s in scenes_data),
            }
        }
    }


# ==============================================================================
# Requirement 1: Exactly 0.75 seconds of initial silence
# ==============================================================================

def test_start_silence_exactly_075_seconds(tmp_path, monkeypatch):
    """Verify that the generated narration WAV begins with exactly 0.75 seconds of silence."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "voiceover_start_silence_seconds", 0.75)
    monkeypatch.setattr(settings, "voiceover_opening_hook_pause_seconds", 0.75)
    monkeypatch.setattr(settings, "voiceover_end_silence_seconds", 1.0)

    video = DummyVideo()
    scenes = [
        {"index": 1, "narration": "Why can a song give you goosebumps?", "est_seconds": 2.5},
        {"index": 2, "narration": "It starts in the brain with dopamine.", "est_seconds": 2.0},
    ]
    prev = _make_storyboard_prev(scenes)
    provider = PredictableToneVoiceoverProvider(sample_rate=24000)

    output = generate_voiceover_step(video, prev, DummyStep(), voiceover_provider=provider)
    audio_path = tmp_path / output["audio_file"]
    assert audio_path.is_file()

    data, sr = sf.read(str(audio_path), dtype="float32")
    assert sr == 24000

    # Exactly 0.75s of silence = 18,000 samples at index 0..17999
    lead_silence_samples = int(0.75 * sr)
    assert np.all(data[:lead_silence_samples] == 0.0), "Initial 0.75 second must be pure zero silence"

    # Immediately after 0.75s, speech energy must be present
    speech_onset_sample = lead_silence_samples + int(0.05 * sr)
    assert np.max(np.abs(data[lead_silence_samples:speech_onset_sample])) > 0.05, "Speech must begin after 0.75s"


# ==============================================================================
# Requirement 2: Approximately 0.75 seconds of opening-hook silence
# ==============================================================================

def test_opening_hook_pause_exactly_075_seconds():
    """Verify that approximately 0.75s of silence is inserted immediately after the opening hook."""
    chunks = [
        "Why can a song give you goosebumps?",
        "It starts in the brain with a sudden burst of dopamine.",
        "Tiny muscles pull your hair upright.",
    ]
    gaps = compute_inter_chunk_gaps(chunks, hook_pause_seconds=0.75, sentence_gap_seconds=0.24)
    assert len(gaps) == 2
    # Gap 0 (after hook) is 0.75s
    assert gaps[0] == 0.75
    # Gap 1 (after chunk 1) is normal sentence gap 0.24s
    assert gaps[1] == 0.24


# ==============================================================================
# Requirement 3: Exactly 1 second of ending silence
# ==============================================================================

def test_ending_silence_exactly_one_second(tmp_path, monkeypatch):
    """Verify that the generated narration WAV contains exactly 1.0 second of silence at the end."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "voiceover_start_silence_seconds", 0.75)
    monkeypatch.setattr(settings, "voiceover_opening_hook_pause_seconds", 0.75)
    monkeypatch.setattr(settings, "voiceover_end_silence_seconds", 1.0)

    video = DummyVideo()
    scenes = [
        {"index": 1, "narration": "Why can a song give you goosebumps?", "est_seconds": 2.5},
        {"index": 2, "narration": "Your brain is just feeling every note.", "est_seconds": 2.0},
    ]
    prev = _make_storyboard_prev(scenes)
    provider = PredictableToneVoiceoverProvider(sample_rate=24000)

    output = generate_voiceover_step(video, prev, DummyStep(), voiceover_provider=provider)
    audio_path = tmp_path / output["audio_file"]

    data, sr = sf.read(str(audio_path), dtype="float32")
    trail_silence_samples = int(1.0 * sr)

    # The last 1.0s = 24,000 samples must be pure zero silence
    assert np.all(data[-trail_silence_samples:] == 0.0), "Final 1 second must be pure zero silence"

    # Immediately before the trailing silence, speech energy must be present
    speech_offset_sample = len(data) - trail_silence_samples - int(0.05 * sr)
    assert np.max(np.abs(data[speech_offset_sample : len(data) - trail_silence_samples])) > 0.05


# ==============================================================================
# Requirement 4: Correct word timestamps after initial silence
# ==============================================================================

def test_word_timestamps_after_initial_silence():
    """Verify word-level timestamps reflect speech starting after the 1.0s initial silence."""
    from app.subtitles.acoustic import map_recognized_words_to_script

    script_tokens = ["Why", "can", "a", "song", "give", "you", "goosebumps?"]
    # Acoustic recognizer detected words starting at 1.02s
    recognized = [
        {"word": "why", "start": 1.02, "end": 1.25, "conf": 0.98},
        {"word": "can", "start": 1.26, "end": 1.35, "conf": 0.95},
        {"word": "a", "start": 1.36, "end": 1.40, "conf": 0.90},
        {"word": "song", "start": 1.41, "end": 1.65, "conf": 0.99},
        {"word": "give", "start": 1.66, "end": 1.78, "conf": 0.94},
        {"word": "you", "start": 1.79, "end": 1.90, "conf": 0.95},
        {"word": "goosebumps", "start": 1.91, "end": 2.55, "conf": 0.97},
    ]

    words = map_recognized_words_to_script(
        script_tokens=script_tokens,
        recognized_words=recognized,
        audio_duration=6.55,
    )

    assert len(words) == 7
    # First word starts around 1.02s
    assert words[0].start >= 1.00
    assert words[0].text == "Why"
    assert words[-1].text == "goosebumps?"
    assert words[-1].end <= 2.60


# ==============================================================================
# Requirement 5: Subtitles do not display during silence
# ==============================================================================

def test_subtitles_do_not_display_during_silence():
    """Verify SRT, WebVTT, and ASS subtitles contain no entries during initial, hook, or ending silences."""
    words = [
        # Scene 1: hook spoken 1.02s - 2.55s
        WordTimestamp(text="Why", start=1.02, end=1.25),
        WordTimestamp(text="can", start=1.26, end=1.35),
        WordTimestamp(text="a", start=1.36, end=1.40),
        WordTimestamp(text="song", start=1.41, end=1.65),
        WordTimestamp(text="give", start=1.66, end=1.78),
        WordTimestamp(text="you", start=1.79, end=1.90),
        WordTimestamp(text="goosebumps?", start=1.91, end=2.55),
        # Scene 2: spoken 3.30s - 5.10s (after 0.75s hook pause)
        WordTimestamp(text="It", start=3.30, end=3.45),
        WordTimestamp(text="starts", start=3.46, end=3.85),
        WordTimestamp(text="in", start=3.86, end=3.95),
        WordTimestamp(text="the", start=3.96, end=4.05),
        WordTimestamp(text="brain.", start=4.06, end=5.10),
    ]
    # Total duration is 7.10s (including 2.0s ending silence after 5.10s)
    total_dur = 7.10

    seg1 = SubtitleSegment(
        index=1,
        text="Why can a song give you goosebumps?",
        start=1.02,
        end=2.55,
        words=words[:7],
    )
    seg2 = SubtitleSegment(
        index=2,
        text="It starts in the brain.",
        start=3.30,
        end=5.10,
        words=words[7:],
    )

    result = SubtitleResult(
        segments=[seg1, seg2],
        total_words=len(words),
        total_duration_seconds=total_dur,
        provider="acoustic",
        model="test",
    )

    # 1. SRT check
    srt_text = result.to_srt()
    assert "00:00:00" not in srt_text, "SRT should not display at 00:00:00 during initial silence"
    assert "00:00:01,020 --> 00:00:02,550" in srt_text
    assert "00:00:03,300 --> 00:00:05,100" in srt_text
    # Does not display up to 7.10s
    assert "00:00:07" not in srt_text

    # 2. VTT check
    vtt_text = result.to_vtt()
    assert "00:00:00." not in vtt_text, "VTT should not display during initial silence"
    assert "00:00:01.020 --> 00:00:02.550" in vtt_text
    assert "00:00:07." not in vtt_text

    # 3. ASS check
    ass_text = result.to_ass()
    # Check all Dialogue start timestamps
    for line in ass_text.splitlines():
        if line.startswith("Dialogue:"):
            parts = line.split(",")
            start_str = parts[1]
            end_str = parts[2]
            # Convert H:MM:SS.cs to seconds
            h, m, scs = start_str.split(":")
            s, cs = scs.split(".")
            sec = int(h) * 3600 + int(m) * 60 + int(s) + int(cs) / 100.0
            assert sec >= 1.0, f"ASS dialogue started at {sec}s during initial silence!"
            # End timestamp must be <= 5.50s (well before the 7.10s ending silence)
            h_e, m_e, scs_e = end_str.split(":")
            s_e, cs_e = scs_e.split(".")
            sec_e = int(h_e) * 3600 + int(m_e) * 60 + int(s_e) + int(cs_e) / 100.0
            assert sec_e <= 5.50, f"ASS dialogue ended at {sec_e}s during ending silence!"


# ==============================================================================
# Requirement 6: Final video duration includes both start and ending silence
# ==============================================================================

def test_final_video_duration_includes_both_start_and_ending_silence():
    """Verify aligned scene timings and render bounds start at 0.0 and extend to total_duration."""
    scenes = [
        {"index": 1, "narration": "Why can a song give you goosebumps?"},
        {"index": 2, "narration": "It starts in the brain."},
    ]
    # Spoken audio from 1.02s to 5.10s, total audio duration is 7.10s
    words = [
        WordTimestamp(text="Why", start=1.02, end=1.25),
        WordTimestamp(text="goosebumps?", start=1.91, end=2.55),
        WordTimestamp(text="It", start=3.30, end=3.45),
        WordTimestamp(text="brain.", start=4.06, end=5.10),
    ]
    seg1 = SubtitleSegment(index=1, text=scenes[0]["narration"], start=1.02, end=2.55, words=words[:2])
    seg2 = SubtitleSegment(index=2, text=scenes[1]["narration"], start=3.30, end=5.10, words=words[2:])
    alignment = SubtitleResult(
        segments=[seg1, seg2],
        total_words=4,
        total_duration_seconds=7.10,
        provider="acoustic",
        model="test",
    )

    timings = compute_aligned_scene_timings(scenes, alignment, total_duration=7.10)
    assert len(timings) == 2

    # Scene 1 must start at 0.0 (visual already visible during the 1.0s initial silence)
    assert timings[0]["start_time"] == 0.0
    # Scene 1 speech start time is 1.02
    assert timings[0]["speech_start_time"] == 1.02

    # Scene 2 must end at 7.10 (visual visible during the 2.0s ending silence)
    assert timings[1]["end_time"] == 7.10
    # Scene 2 speech ends at 5.10
    assert timings[1]["speech_end_time"] == 5.10
    # Total visual duration equals total audio duration
    total_visual_dur = round(timings[1]["end_time"] - timings[0]["start_time"], 2)
    assert total_visual_dur == 7.10


# ==============================================================================
# Requirement 7: No duplicated pauses when audio chunks are combined
# ==============================================================================

def test_no_duplicated_pauses_when_audio_chunks_combined():
    """Verify that opening hook pause (0.75s) is added exactly once and subsequent gaps are 0.24s."""
    chunks = [
        "Why can a song give you goosebumps?",
        "It starts in the brain with dopamine.",
        "Your body reacts like something intense is happening.",
        "So next time music gives you chills, enjoy it.",
    ]
    # Exactly 4 chunks -> exactly 3 gaps
    gaps = compute_inter_chunk_gaps(chunks, hook_pause_seconds=0.75, sentence_gap_seconds=0.24)
    assert len(gaps) == 3
    # First gap is the hook pause
    assert gaps[0] == 0.75
    # Second and third gaps are standard 0.24s pauses, NOT duplicated hook pauses
    assert gaps[1] == 0.24
    assert gaps[2] == 0.24
    assert gaps.count(0.75) == 1, "Hook pause must be applied exactly once"


# ==============================================================================
# Requirement 8: Existing legacy paths continue working
# ==============================================================================

def test_legacy_paths_continue_working_with_zero_padding(tmp_path, monkeypatch):
    """Verify that configuring 0.0 silence durations preserves legacy unpadded behavior."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "voiceover_start_silence_seconds", 0.0)
    monkeypatch.setattr(settings, "voiceover_opening_hook_pause_seconds", 0.0)
    monkeypatch.setattr(settings, "voiceover_end_silence_seconds", 0.0)

    provider = PredictableToneVoiceoverProvider(sample_rate=24000)
    script = "Single sentence narration without any padding."

    res, count = generate_continuous_audio(
        script,
        provider=provider,
        start_silence_seconds=0.0,
        opening_hook_pause_seconds=0.0,
        end_silence_seconds=0.0,
        isolate_hook=False,
    )
    assert count == 1

    data, sr = sf.read(io.BytesIO(res.audio_bytes), dtype="float32")
    # First sample must immediately contain audio tone (no 1.0s zero pad)
    assert np.max(np.abs(data[: int(0.05 * sr)])) > 0.05
    # Last sample must immediately contain audio tone (no 2.0s zero pad)
    assert np.max(np.abs(data[-int(0.05 * sr) :])) > 0.05


def test_default_settings_values():
    """Verify that settings default to 0.75s start silence, 0.75s hook pause, and 1.0s end silence."""
    assert settings.voiceover_start_silence_seconds == 0.75
    assert settings.voiceover_opening_hook_pause_seconds == 0.75
    assert settings.voiceover_end_silence_seconds == 1.0
