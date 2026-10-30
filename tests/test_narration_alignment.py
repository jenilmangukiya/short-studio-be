"""Tests for shared narration word alignment, scene-boundary mapping, and reuse across steps.

Vosk is replaced with a deterministic fake so tests are offline and fast.
"""

from __future__ import annotations

from pathlib import Path
import uuid

import pytest

from app.config import settings
from app.steps.render import _resolve_aligned_bounds
from app.steps.subtitles import generate_subtitles_step
from app.steps.voiceover import generate_voiceover_step
from app.subtitles import (
    SubtitleError,
    SubtitleResult,
    SubtitleSegment,
    WordTimestamp,
    compute_aligned_scene_timings,
)
from app.subtitles.acoustic import AcousticSubtitleProvider
from app.voiceover import MockVoiceoverProvider, create_mock_wav


class DummyVideo:
    def __init__(self, video_id=None):
        self.id = video_id or uuid.uuid4()
        self.topic = "AI cost"
        self.channel = None
        self.output_path = None


class DummyStep:
    def __init__(self, name="voiceover", output=None):
        self.name = name
        self.user_prompt = None
        self.output = output
        self.status = "pending"


NARRATIONS = [
    "Why does answering one AI question cost so much?",
    "Because the answer takes powerful computer chips,",
    "and those chips use electricity every time they work.",
]


def _fake_vosk_factory(counter: dict, word_dur: float = 0.25, lead_in: float = 0.1):
    """Return a fake align_script_with_vosk producing realistic pauses at punctuation."""

    def fake_align(audio_path, script_tokens, model_path=None):
        counter["count"] = counter.get("count", 0) + 1
        import soundfile as sf

        total = round(float(sf.info(str(audio_path)).duration), 3)
        words = []
        t = lead_in
        for tok in script_tokens:
            words.append(WordTimestamp(text=tok, start=round(t, 3), end=round(t + word_dur, 3), confidence=0.99))
            t += word_dur
            if tok.endswith(("?", ".", "!")):
                t += 0.45
            elif tok.endswith(","):
                t += 0.25
            else:
                t += 0.03
        return total, words

    return fake_align


def _make_result(scene_words: list[list[tuple[float, float]]], total: float) -> SubtitleResult:
    segments = []
    for i, words in enumerate(scene_words, 1):
        wts = [WordTimestamp(text=f"w{i}_{k}", start=s, end=e) for k, (s, e) in enumerate(words)]
        segments.append(SubtitleSegment(index=i, text=NARRATIONS[i - 1], start=wts[0].start, end=wts[-1].end, words=wts))
    return SubtitleResult(
        segments=segments,
        total_words=sum(len(w) for w in scene_words),
        total_duration_seconds=total,
        provider="acoustic",
        model="test",
        latency_seconds=0.0,
        timing_source="forced_alignment",
    )


# ---------------------------------------------------------------------------
# compute_aligned_scene_timings
# ---------------------------------------------------------------------------


def test_aligned_timings_are_gapless_and_cut_inside_pauses():
    scenes = [{"index": i, "narration": n} for i, n in enumerate(NARRATIONS, 1)]
    result = _make_result(
        [
            [(0.10, 0.50), (0.55, 3.20)],  # sentence ends at 3.20, next speech at 3.63
            [(3.63, 4.00), (4.05, 6.00)],  # clause ends at 6.00, next speech at 6.30
            [(6.30, 7.00), (7.05, 9.10)],
        ],
        total=9.5,
    )
    timings = compute_aligned_scene_timings(scenes, result, 9.5)

    assert [t["timing_type"] for t in timings] == ["aligned"] * 3
    assert timings[0]["start_time"] == 0.0
    assert timings[-1]["end_time"] == 9.5
    for a, b in zip(timings, timings[1:]):
        assert a["end_time"] == b["start_time"]  # no gaps / overlaps

    # Cuts land in the pause: after previous scene's last word, before next scene's first word.
    assert 3.20 <= timings[0]["end_time"] <= 3.63
    assert 6.00 <= timings[1]["end_time"] <= 6.30
    # Sentence boundary cut: prev_end + min(gap*0.65, gap-0.08)
    assert timings[0]["end_time"] == pytest.approx(3.20 + min(0.43 * 0.65, 0.43 - 0.08), abs=0.002)
    # Clause boundary cut: prev_end + min(gap*0.55, gap-0.05)
    assert timings[1]["end_time"] == pytest.approx(6.00 + min(0.30 * 0.55, 0.30 - 0.05), abs=0.002)
    assert sum(t["duration_seconds"] for t in timings) == pytest.approx(9.5, abs=0.003)


def test_aligned_timings_reject_mismatched_segments():
    scenes = [{"index": 1, "narration": NARRATIONS[0]}, {"index": 2, "narration": NARRATIONS[1]}]
    result = _make_result([[(0.1, 1.0)]], total=2.0)
    with pytest.raises(SubtitleError):
        compute_aligned_scene_timings(scenes, result, 2.0)


# ---------------------------------------------------------------------------
# voiceover -> subtitles reuse
# ---------------------------------------------------------------------------


def _storyboard_prev() -> dict:
    return {
        "storyboard": {
            "storyboard": {
                "scenes": [
                    {"index": i, "narration": n, "image_prompt": "x", "shot_type": "medium", "est_seconds": 3.0}
                    for i, n in enumerate(NARRATIONS, 1)
                ]
            }
        }
    }


def test_voiceover_persists_alignment_and_subtitles_reuse_it(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    import app.subtitles.acoustic as acoustic_mod

    counter: dict = {}
    monkeypatch.setattr(acoustic_mod, "align_script_with_vosk", _fake_vosk_factory(counter))

    video = DummyVideo()
    step = DummyStep()
    out = generate_voiceover_step(
        video,
        _storyboard_prev(),
        step,
        voiceover_provider=MockVoiceoverProvider(),
        subtitle_provider=AcousticSubtitleProvider(),
    )

    assert counter["count"] == 1
    vo = out["voiceover"]
    assert vo["timing_type"] == "aligned"
    assert out["meta"]["timing_type"] == "aligned"
    alignment_hash = vo["alignment_hash"]
    assert alignment_hash
    cache_file = tmp_path / str(video.id) / "audio" / f"alignment_{alignment_hash[:12]}.json"
    assert cache_file.is_file()

    items = vo["items"]
    assert all(it["timing_type"] == "aligned" for it in items)
    assert items[0]["start_time"] == 0.0
    assert items[-1]["end_time"] == pytest.approx(out["duration_seconds"], abs=0.002)
    for a, b in zip(items, items[1:]):
        assert a["end_time"] == b["start_time"]
        # The cut never clips the previous scene's speech or the next scene's first word
        assert a["speech_end_time"] <= a["end_time"] <= b["speech_start_time"]

    # Subtitles reuse the persisted alignment: Vosk is not called again.
    raw_mp4 = tmp_path / str(video.id) / "final_video.mp4"
    raw_mp4.write_bytes(b"DUMMY")
    sub_prev = {"render": {"video_path": str(raw_mp4)}, "voiceover": out}
    sub_out = generate_subtitles_step(video, sub_prev, subtitle_provider=AcousticSubtitleProvider())
    assert counter["count"] == 1
    assert sub_out["subtitles"]["reused"] is True
    assert sub_out["subtitles"]["alignment_hash"] == alignment_hash


def test_voiceover_reuse_upgrades_estimated_timing(tmp_path, monkeypatch):
    """Existing voiceover with estimated timing is upgraded to aligned without regenerating audio."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    import app.subtitles.acoustic as acoustic_mod

    counter: dict = {}
    monkeypatch.setattr(acoustic_mod, "align_script_with_vosk", _fake_vosk_factory(counter))

    video = DummyVideo()
    step = DummyStep()
    provider = MockVoiceoverProvider()
    first = generate_voiceover_step(video, _storyboard_prev(), step, voiceover_provider=provider)
    assert first["voiceover"]["timing_type"] == "estimated"  # mock audio: alignment skipped
    assert counter.get("count", 0) == 0
    wav_path = tmp_path / first["audio_file"]
    mtime = wav_path.stat().st_mtime_ns

    second = generate_voiceover_step(
        video, _storyboard_prev(), step, voiceover_provider=provider, subtitle_provider=AcousticSubtitleProvider()
    )
    assert second["meta"]["scenes_reused"] == 3
    assert second["voiceover"]["timing_type"] == "aligned"
    assert counter["count"] == 1
    assert wav_path.stat().st_mtime_ns == mtime  # audio untouched


def test_voiceover_falls_back_to_estimated_when_alignment_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    import app.subtitles.acoustic as acoustic_mod

    def failing(*args, **kwargs):
        raise SubtitleError("no speech")

    monkeypatch.setattr(acoustic_mod, "align_script_with_vosk", failing)
    out = generate_voiceover_step(
        DummyVideo(),
        _storyboard_prev(),
        DummyStep(),
        voiceover_provider=MockVoiceoverProvider(),
        subtitle_provider=AcousticSubtitleProvider(),
    )
    assert out["voiceover"]["timing_type"] == "estimated"
    assert any("estimated scene timing" in w for w in out["warnings"])


# ---------------------------------------------------------------------------
# render timing resolution
# ---------------------------------------------------------------------------


def test_render_uses_aligned_items_directly(tmp_path):
    wav = tmp_path / "n.wav"
    wav.write_bytes(create_mock_wav(3.0, 24000))
    items = {
        1: {"index": 1, "start_time": 0.0, "end_time": 1.37, "timing_type": "aligned"},
        2: {"index": 2, "start_time": 1.37, "end_time": 3.0, "timing_type": "aligned"},
    }
    bounds, source, _, _ = _resolve_aligned_bounds(
        video=DummyVideo(),
        vo_data={"voiceover": {"items": list(items.values()), "alignment_hash": "abc"}},
        valid_audio=items,
        all_indices=[1, 2],
        continuous_audio_abs=wav,
        continuous_duration=3.0,
        storage_root=tmp_path,
    )
    assert source == "aligned"
    assert bounds == {1: (0.0, 1.37), 2: (1.37, 3.0)}


def test_render_aligns_legacy_estimated_voiceover_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "subtitles_provider", "acoustic")
    import app.subtitles.acoustic as acoustic_mod

    counter: dict = {}
    monkeypatch.setattr(acoustic_mod, "align_script_with_vosk", _fake_vosk_factory(counter))

    video = DummyVideo()
    audio_dir = tmp_path / str(video.id) / "audio"
    audio_dir.mkdir(parents=True)
    wav = audio_dir / "narration_x.wav"
    wav.write_bytes(create_mock_wav(9.0, 24000))
    items = {
        i: {"index": i, "narration": n, "start_time": (i - 1) * 3.0, "end_time": i * 3.0, "timing_type": "estimated"}
        for i, n in enumerate(NARRATIONS, 1)
    }
    vo_data = {
        "audio_file": f"{video.id}/audio/narration_x.wav",
        "narration_script": " ".join(NARRATIONS),
        "audio_hash": "x",
        "meta": {"provider": "kokoro"},
        "voiceover": {"items": list(items.values())},
    }
    kwargs = dict(
        video=video,
        vo_data=vo_data,
        valid_audio=items,
        all_indices=[1, 2, 3],
        continuous_audio_abs=wav,
        continuous_duration=9.0,
        storage_root=tmp_path,
    )
    bounds, source, alignment_hash, reused = _resolve_aligned_bounds(**kwargs)
    assert source == "aligned" and reused is False
    assert counter["count"] == 1
    assert bounds[1][0] == 0.0 and bounds[3][1] == 9.0
    assert bounds[1][1] == bounds[2][0] and bounds[2][1] == bounds[3][0]
    assert (audio_dir / f"alignment_{alignment_hash[:12]}.json").is_file()

    # Second render reuses the cached alignment
    _, _, _, reused2 = _resolve_aligned_bounds(**kwargs)
    assert reused2 is True
    assert counter["count"] == 1


def test_render_skips_alignment_for_mock_voiceover(tmp_path):
    wav = tmp_path / "n.wav"
    wav.write_bytes(create_mock_wav(2.0, 24000))
    items = {1: {"index": 1, "narration": "Hello there.", "duration_seconds": 2.0}}
    bounds, source, _, _ = _resolve_aligned_bounds(
        video=DummyVideo(),
        vo_data={"meta": {"provider": "mock"}, "voiceover": {"items": list(items.values())}},
        valid_audio=items,
        all_indices=[1],
        continuous_audio_abs=wav,
        continuous_duration=2.0,
        storage_root=tmp_path,
    )
    assert bounds is None and source == "estimated"

