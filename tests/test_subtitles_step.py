"""Unit and integration tests for the subtitles step (app/steps/subtitles.py).

Tests:
1. Data structures and export formats (WordTimestamp, SubtitleSegment, SubtitleResult, SRT, VTT).
2. Mock and Acoustic subtitle provider timestamp precision and monotonic progression.
3. Pipeline step preconditions (missing render step, missing voiceover step).
4. Full execution of generate_subtitles_step with filesystem artifact generation (.json, .vtt, .srt).
5. Pipeline ordering and rerun behavior (preserving video.output_path and upstream status).
6. FastAPI router endpoint (/videos/{video_id}/subtitles).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.config import settings
from app.db import Base
from app.models import Channel, User, Video, VideoStep
from app.pipeline import STEPS, HANDLERS, init_steps, rerun_step
from app.steps.subtitles import PipelineError, generate_subtitles_step
from app.subtitles import (
    BaseSubtitleProvider,
    SubtitleResult,
    SubtitleSegment,
    WordTimestamp,
    get_subtitle_provider,
)
from app.subtitles.acoustic import AcousticSubtitleProvider
from app.subtitles.mock import MockSubtitleProvider
from app.voiceover.mock import create_mock_wav


class DummyVideo:
    """Lightweight test video mock."""

    def __init__(self, video_id=None, output_path="/tmp/final.mp4"):
        self.id = video_id or uuid.uuid4()
        self.topic = "Why airplane windows have tiny holes"
        self.output_path = output_path


# ==============================================================================
# 1. Subtitle Data Structures & Exporters
# ==============================================================================

def test_word_timestamp_and_segment_serialization():
    w1 = WordTimestamp(text="Why", start=0.04, end=0.25, confidence=0.98)
    w2 = WordTimestamp(text="do", start=0.25, end=0.42, confidence=0.98)
    seg = SubtitleSegment(
        index=1,
        text="Why do",
        start=0.0,
        end=0.5,
        words=[w1, w2],
    )

    seg_dict = seg.to_dict()
    assert seg_dict["index"] == 1
    assert seg_dict["text"] == "Why do"
    assert seg_dict["start"] == 0.0
    assert seg_dict["end"] == 0.5
    assert len(seg_dict["words"]) == 2
    assert seg_dict["words"][0]["text"] == "Why"
    assert seg_dict["words"][0]["start"] == 0.04
    assert seg_dict["words"][0]["end"] == 0.25


def test_subtitle_result_srt_and_vtt_formatting():
    w1 = WordTimestamp(text="Why", start=1.5, end=1.8)
    seg1 = SubtitleSegment(index=1, text="Why airplane windows have holes.", start=1.0, end=3.5, words=[w1])
    res = SubtitleResult(
        segments=[seg1],
        total_words=5,
        total_duration_seconds=3.5,
        provider="test",
        model="test-model",
    )

    srt = res.to_srt()
    assert "1" in srt
    assert "00:00:01,000 --> 00:00:03,500" in srt
    assert "Why airplane windows have holes." in srt

    vtt = res.to_vtt()
    assert vtt.startswith("WEBVTT")
    assert "00:00:01.000 --> 00:00:03.500" in vtt
    assert "Why airplane windows have holes." in vtt


# ==============================================================================
# 2. Providers: Mock & Acoustic Timing Precision
# ==============================================================================

def test_mock_subtitle_provider_generates_valid_timestamps():
    provider = MockSubtitleProvider()
    scenes = [
        {"index": 1, "narration": "Why do airplane windows need a tiny hole?", "duration_seconds": 3.0},
        {"index": 2, "narration": "They balance the cabin pressure.", "duration_seconds": 2.5},
    ]
    res = provider.generate(scenes)

    assert res.provider == "mock"
    assert len(res.segments) == 2
    assert res.total_words == 13
    assert res.total_duration_seconds == 5.5

    # Check strictly increasing timestamps
    for seg in res.segments:
        assert seg.start < seg.end
        prev_end = seg.start
        for w in seg.words:
            assert w.start >= prev_end
            assert w.end > w.start
            prev_end = w.end


def test_acoustic_subtitle_provider_with_synthetic_audio(tmp_path):
    # Create two synthetic wav files
    wav1_bytes = create_mock_wav(duration_seconds=2.0)
    wav2_bytes = create_mock_wav(duration_seconds=1.5)

    audio_dir = tmp_path / "audio"
    audio_dir.mkdir(parents=True)
    f1 = audio_dir / "scene_01.wav"
    f2 = audio_dir / "scene_02.wav"
    f1.write_bytes(wav1_bytes)
    f2.write_bytes(wav2_bytes)

    scenes = [
        {"index": 1, "narration": "Why do airplane windows need a tiny hole?", "duration_seconds": 2.0, "file": "audio/scene_01.wav"},
        {"index": 2, "narration": "They balance cabin pressure.", "duration_seconds": 1.5, "file": "audio/scene_02.wav"},
    ]

    provider = AcousticSubtitleProvider()
    res = provider.generate(scenes, storage_root=tmp_path)

    assert res.provider == "acoustic"
    assert len(res.segments) == 2
    assert res.segments[0].start == 0.0
    assert res.segments[0].end == pytest.approx(2.0, abs=0.05)
    assert res.segments[1].start == pytest.approx(2.0, abs=0.05)
    assert res.segments[1].end == pytest.approx(3.5, abs=0.1)

    # Word timestamps inside each segment must be strictly non-overlapping
    for seg in res.segments:
        for i in range(len(seg.words) - 1):
            assert seg.words[i].start < seg.words[i].end
            assert seg.words[i].end <= seg.words[i + 1].start + 0.001


def test_acoustic_subtitle_provider_fallback_when_audio_missing():
    """Provider falls back cleanly to phonetic duration estimation if audio file is missing."""
    scenes = [
        {"index": 1, "narration": "Pockets on jeans tore completely off.", "duration_seconds": 2.4, "file": "missing.wav"}
    ]
    provider = AcousticSubtitleProvider()
    res = provider.generate(scenes, storage_root=Path("/nonexistent"))

    assert len(res.segments) == 1
    assert res.segments[0].end == pytest.approx(2.4, abs=0.01)
    assert len(res.segments[0].words) == 6


# ==============================================================================
# 3. Pipeline Step Preconditions & Failure Handling
# ==============================================================================

def test_subtitles_step_requires_render_step():
    video = DummyVideo(output_path=None)
    prev = {
        "voiceover": {
            "voiceover": {"items": [{"index": 1, "status": "ok", "narration": "Test"}]}
        }
    }
    with pytest.raises(PipelineError) as exc_info:
        generate_subtitles_step(video, prev)
    assert "Video render step must be completed" in str(exc_info.value)


def test_subtitles_step_requires_voiceover_step():
    video = DummyVideo(output_path="/tmp/final.mp4")
    prev = {
        "render": {"video_path": "/tmp/final.mp4", "duration_seconds": 10.0}
    }
    with pytest.raises(PipelineError) as exc_info:
        generate_subtitles_step(video, prev)
    assert "Voiceover step must be completed" in str(exc_info.value)


# ==============================================================================
# 4. Step Execution & Artifact Generation
# ==============================================================================

def test_generate_subtitles_step_full_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    video_id = uuid.uuid4()
    video = DummyVideo(video_id=video_id, output_path=str(tmp_path / "final.mp4"))

    # Prepare fake audio
    audio_dir = tmp_path / str(video_id) / "audio"
    audio_dir.mkdir(parents=True)
    f1 = audio_dir / "s1.wav"
    f1.write_bytes(create_mock_wav(2.0))

    prev = {
        "render": {"video_path": str(tmp_path / "final.mp4"), "duration_seconds": 2.0},
        "voiceover": {
            "voiceover": {
                "items": [
                    {
                        "index": 1,
                        "status": "ok",
                        "narration": "Why do airplane windows need a tiny hole?",
                        "duration_seconds": 2.0,
                        "file": f"{video_id}/audio/s1.wav",
                    }
                ]
            }
        },
    }

    step = VideoStep(video_id=video_id, name="subtitles", position=5, status="running")
    result = generate_subtitles_step(video, prev, step=step, subtitle_provider=MockSubtitleProvider())

    assert "subtitles" in result
    assert result["subtitles"]["total_words"] == 8
    assert result["subtitles"]["total_duration_seconds"] == 2.0
    assert len(result["subtitles"]["segments"]) == 1
    assert result["meta"]["task"] == "subtitles"

    # Verify artifacts on disk
    json_path = tmp_path / str(video_id) / "subtitles.json"
    vtt_path = tmp_path / str(video_id) / "subtitles.vtt"
    srt_path = tmp_path / str(video_id) / "subtitles.srt"

    assert json_path.is_file()
    assert vtt_path.is_file()
    assert srt_path.is_file()

    saved_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved_json["total_words"] == 8
    assert "WEBVTT" in vtt_path.read_text(encoding="utf-8")


# ==============================================================================
# 5. Pipeline Ordering & Retry / Rerun Behavior
# ==============================================================================

def test_pipeline_steps_includes_subtitles_at_correct_position():
    assert "subtitles" in STEPS
    assert STEPS == ["script", "storyboard", "images", "voiceover", "render", "subtitles"]
    assert "subtitles" in HANDLERS
    assert "subtitle" in HANDLERS


def test_rerun_subtitles_preserves_video_output_path():
    """Rerunning the subtitles step resets subtitles to pending without wiping video.output_path."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        user = User(email="test_sub@example.com", password_hash="hash")
        session.add(user)
        session.flush()

        channel = Channel(user_id=user.id, name="Everyday Mysteries")
        session.add(channel)
        session.flush()

        video = Video(
            channel_id=channel.id,
            topic="Submarine Cables",
            output_path="/storage/test_video.mp4",
            status="completed",
        )
        session.add(video)
        session.flush()

        # Initialize all 6 steps
        init_steps(session, video)
        assert len(video.steps) == 6
        assert video.steps[4].name == "render"
        assert video.steps[5].name == "subtitles"

        # Mark all steps done
        for s in video.steps:
            s.status = "done"
        video.output_path = "/storage/test_video.mp4"
        session.commit()

        # Rerun subtitles step
        rerun_step(session, video, "subtitles", prompt="Regenerate timing with high precision")

        # Check: subtitles is pending, upstream steps remain done, output_path is preserved!
        assert video.steps[5].status == "pending"
        assert video.steps[5].user_prompt == "Regenerate timing with high precision"
        assert video.steps[4].status == "done"  # render remains done!
        assert video.steps[3].status == "done"  # voiceover remains done!
        assert video.output_path == "/storage/test_video.mp4"  # preserved!


def test_rerun_render_invalidates_subtitles_and_video_output_path():
    """Rerunning render step invalidates output_path and marks downstream subtitles stale."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        user = User(email="test_render_rerun@example.com", password_hash="hash")
        session.add(user)
        session.flush()

        channel = Channel(user_id=user.id, name="Tech Facts")
        session.add(channel)
        session.flush()

        video = Video(
            channel_id=channel.id,
            topic="Airplane Windows",
            output_path="/storage/test_airplane.mp4",
            status="completed",
        )
        session.add(video)
        session.flush()

        init_steps(session, video)
        for s in video.steps:
            s.status = "done"
        video.output_path = "/storage/test_airplane.mp4"
        session.commit()

        # Rerun render
        rerun_step(session, video, "render", prompt=None)

        assert video.steps[4].status == "pending"  # render is pending
        assert video.steps[5].status == "stale"    # subtitles becomes stale
        assert video.output_path is None           # video output is invalidated because render is rerunning


def test_subtitle_result_ass_format():
    w1 = WordTimestamp(text="Why", start=0.04, end=0.25, confidence=0.98)
    w2 = WordTimestamp(text="do", start=0.25, end=0.42, confidence=0.98)
    w3 = WordTimestamp(text="airplanes", start=0.42, end=1.10, confidence=0.98)
    seg = SubtitleSegment(
        index=1,
        text="Why do airplanes",
        start=0.0,
        end=1.2,
        words=[w1, w2, w3],
    )
    res = SubtitleResult(
        segments=[seg],
        total_words=3,
        total_duration_seconds=1.2,
        provider="test",
        model="test",
    )

    ass_text = res.to_ass()
    assert "[Script Info]" in ass_text
    assert "PlayResX: 1080" in ass_text
    assert "PlayResY: 1920" in ass_text
    assert "[V4+ Styles]" in ass_text
    assert "[Events]" in ass_text
    assert r"{\c&H0028F5FC&}Why{\c&H00FFFFFF&} do airplanes" in ass_text
    assert r"Why {\c&H0028F5FC&}do{\c&H00FFFFFF&} airplanes" in ass_text
    assert r"Why do {\c&H0028F5FC&}airplanes{\c&H00FFFFFF&}" in ass_text


def test_generate_subtitles_step_burns_captioned_video(tmp_path, monkeypatch):
    import subprocess
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    video_id = uuid.uuid4()
    video_dir = tmp_path / str(video_id)
    video_dir.mkdir(parents=True)
    raw_mp4 = video_dir / "final_video.mp4"

    # Create a tiny 1-second silent MP4 test file using ffmpeg testsrc
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=1:size=1080x1920:rate=30",
        "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo",
        "-t", "1",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(raw_mp4),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert raw_mp4.is_file()

    video = DummyVideo(video_id=video_id, output_path=str(raw_mp4))

    audio_dir = video_dir / "audio"
    audio_dir.mkdir(parents=True)
    f1 = audio_dir / "s1.wav"
    f1.write_bytes(create_mock_wav(1.0))

    prev = {
        "render": {"video_path": str(raw_mp4), "duration_seconds": 1.0},
        "voiceover": {
            "voiceover": {
                "items": [
                    {
                        "index": 1,
                        "status": "ok",
                        "narration": "Why airplane windows?",
                        "duration_seconds": 1.0,
                        "file": f"{video_id}/audio/s1.wav",
                    }
                ]
            }
        },
    }

    step = VideoStep(video_id=video_id, name="subtitles", position=5, status="running")
    result = generate_subtitles_step(video, prev, step=step, subtitle_provider=MockSubtitleProvider())

    # Captioned video must exist and be recorded in output
    captioned_mp4 = video_dir / "final_video_captioned.mp4"
    assert captioned_mp4.is_file()
    assert captioned_mp4.stat().st_size > 0
    assert result["subtitles"]["captioned_video_path"] == str(captioned_mp4)
    assert result["captioned_video_path"] == str(captioned_mp4)
    # Original video is untouched
    assert raw_mp4.is_file()


def test_download_endpoint_captioned_and_original(tmp_path, monkeypatch):
    from app.routers.videos import download

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    video_id = uuid.uuid4()
    video_dir = tmp_path / str(video_id)
    video_dir.mkdir(parents=True)

    raw_mp4 = video_dir / "final_video.mp4"
    raw_mp4.write_bytes(b"RAW_MP4_CONTENT")

    captioned_mp4 = video_dir / "final_video_captioned.mp4"
    captioned_mp4.write_bytes(b"CAPTIONED_MP4_CONTENT")

    video = DummyVideo(video_id=video_id, output_path=str(raw_mp4))

    # 1. Download captioned (default)
    resp_cap = download(video=video, captioned=True)
    assert resp_cap.path == str(captioned_mp4)
    assert "captioned" in resp_cap.filename

    # 2. Download original
    resp_orig = download(video=video, captioned=False)
    assert resp_orig.path == str(raw_mp4)
    assert "original" in resp_orig.filename

    # 3. Fallback when captioned not yet generated
    captioned_mp4.unlink()
    resp_fallback = download(video=video, captioned=True)
    assert resp_fallback.path == str(raw_mp4)
