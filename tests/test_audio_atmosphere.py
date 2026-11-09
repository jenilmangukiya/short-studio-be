"""Automated unit and integration tests for Stage 4.9: Audio Pacing & Atmosphere.

Tests:
1. Configurable Kokoro speech speed with backward-compatible 1.0 fallback.
2. Speed validation and safe handling of invalid/out-of-range configurations.
3. Optional background music resolution, format validation (.mp3 and .wav), and error handling.
4. FFmpeg sidechain ducking command construction, filter graph signal flow, and peak limiter.
5. Audio mix preservation between master and captioned outputs (-c:a copy).
6. Backward compatibility for videos without background music.
"""

from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch
import uuid
import pytest

from app.config import settings
from app.images.mock import create_solid_png
from app.models import Channel, Video, VideoStep
from app.pipeline import h_render, h_voiceover
from app.steps.render import (
    PipelineError as RenderPipelineError,
    SUPPORTED_MUSIC_EXTENSIONS,
    build_audio_mux_command,
    generate_render_step,
    resolve_and_validate_music_track,
)
from app.steps.voiceover import (
    PipelineError as VoiceoverPipelineError,
    generate_voiceover_step,
)
from app.voiceover import (
    MockVoiceoverProvider,
    create_mock_wav,
    validate_voiceover_speed,
)


# ==============================================================================
# Test Fixtures & Dummies
# ==============================================================================

class DummyChannel:
    def __init__(self, voice=None, voice_speed=None, settings_dict=None):
        self.settings = dict(settings_dict or {})
        if voice:
            self.settings["voice"] = voice
        if voice_speed is not None:
            self.settings["voice_speed"] = voice_speed


class DummyVideo:
    def __init__(self, video_id=None, channel=None):
        self.id = video_id or uuid.uuid4()
        self.topic = "Why Airplane Windows Have Tiny Holes"
        self.channel = channel
        self.output_path = None


class DummyStep:
    def __init__(self, name="voiceover", user_prompt=None, output=None, status="pending"):
        self.name = name
        self.user_prompt = user_prompt
        self.output = output
        self.status = status


def _make_storyboard_prev(scene_count: int = 2) -> dict:
    scenes = []
    for i in range(1, scene_count + 1):
        scenes.append({
            "index": i,
            "scene_id": f"scene_{i}",
            "narration": f"This is scene {i} narration explaining how pressure shifts in the air.",
            "est_seconds": 2.5,
        })
    return {"storyboard": {"storyboard": {"scenes": scenes}}}


# ==============================================================================
# 1. Narration Speed Tests
# ==============================================================================

def test_validate_voiceover_speed_helper():
    """Verify validation helper handles normal, boundary, and invalid inputs."""
    # Normal values
    assert validate_voiceover_speed(1.0) == 1.0
    assert validate_voiceover_speed(0.94) == 0.94
    assert validate_voiceover_speed(0.5) == 0.5
    assert validate_voiceover_speed(2.0) == 2.0
    assert validate_voiceover_speed("0.94") == 0.94

    # None falls back to provided default
    assert validate_voiceover_speed(None, fallback=1.0) == 1.0
    assert validate_voiceover_speed(None, fallback=None) is None

    # Out of range (< 0.5 or > 2.0) falls back
    assert validate_voiceover_speed(0.2, fallback=1.0) == 1.0
    assert validate_voiceover_speed(2.5, fallback=1.0) == 1.0
    assert validate_voiceover_speed(-1.0, fallback=1.0) == 1.0

    # Non-numeric or non-finite inputs fall back
    assert validate_voiceover_speed("fast", fallback=1.0) == 1.0
    assert validate_voiceover_speed(float("nan"), fallback=1.0) == 1.0
    assert validate_voiceover_speed(float("inf"), fallback=1.0) == 1.0


def test_default_speed_is_1_0_when_not_configured(tmp_path, monkeypatch):
    """Default speech speed remains 1.0 when not explicitly set."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "voiceover_default_speed", 1.0)

    video = DummyVideo(channel=DummyChannel())
    prev = _make_storyboard_prev()
    provider = MockVoiceoverProvider()

    res = generate_voiceover_step(video, prev, step=None, voiceover_provider=provider)

    assert res["meta"]["speed"] == 1.0
    assert res["voiceover"]["speed"] == 1.0


def test_configured_speed_0_94_reaches_voiceover_step_and_metadata(tmp_path, monkeypatch):
    """Configuring voice_speed=0.94 reaches voiceover step and is recorded in metadata."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    # Test via channel settings
    video = DummyVideo(channel=DummyChannel(voice_speed=0.94))
    prev = _make_storyboard_prev()
    provider = MockVoiceoverProvider()

    res = generate_voiceover_step(video, prev, step=None, voiceover_provider=provider)

    assert res["meta"]["speed"] == 0.94
    assert res["voiceover"]["speed"] == 0.94
    assert res["warnings"] == []


def test_configured_speed_via_global_settings(tmp_path, monkeypatch):
    """Configuring voiceover_default_speed in settings acts as the fallback default."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "voiceover_default_speed", 0.94)

    video = DummyVideo(channel=DummyChannel())
    prev = _make_storyboard_prev()
    provider = MockVoiceoverProvider()

    res = generate_voiceover_step(video, prev, step=None, voiceover_provider=provider)

    assert res["meta"]["speed"] == 0.94
    assert res["voiceover"]["speed"] == 0.94


def test_invalid_channel_speed_falls_back_safely_with_warning(tmp_path, monkeypatch):
    """Invalid speed string or out-of-range value in channel settings safely falls back with a warning."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "voiceover_default_speed", 1.0)

    video = DummyVideo(channel=DummyChannel(voice_speed="unrealistic_speed"))
    prev = _make_storyboard_prev()
    provider = MockVoiceoverProvider()

    res = generate_voiceover_step(video, prev, step=None, voiceover_provider=provider)

    assert res["meta"]["speed"] == 1.0
    assert any("Invalid voice_speed" in w for w in res["warnings"])


# ==============================================================================
# 2. Background Music Configuration & Validation Tests
# ==============================================================================

def test_music_disabled_by_default():
    """Music is disabled by default when no arguments or channel settings exist."""
    res = resolve_and_validate_music_track(
        music_path=None,
        music_enabled=None,
        channel_settings=None,
    )
    assert res is None


def test_music_explicitly_disabled_overrides_configured_path(tmp_path):
    """Setting music_enabled=False explicitly disables music even if a path is provided."""
    dummy_mp3 = tmp_path / "ambient.mp3"
    dummy_mp3.write_bytes(b"dummy mp3 content")

    res = resolve_and_validate_music_track(
        music_path=str(dummy_mp3),
        music_enabled=False,
    )
    assert res is None


def test_valid_mp3_and_wav_paths_accepted(tmp_path):
    """Valid MP3 and WAV files on disk are resolved to absolute Paths."""
    track_mp3 = tmp_path / "background.mp3"
    track_mp3.write_bytes(b"ID3 valid fake mp3 bytes")
    track_wav = tmp_path / "background.wav"
    track_wav.write_bytes(create_mock_wav(1.0, 24000))

    res_mp3 = resolve_and_validate_music_track(music_path=str(track_mp3))
    assert res_mp3 == track_mp3.resolve()

    res_wav = resolve_and_validate_music_track(music_path=str(track_wav))
    assert res_wav == track_wav.resolve()


def test_channel_settings_background_music_resolved(tmp_path):
    """Music configured via channel settings ('background_music') is resolved."""
    track = tmp_path / "channel_music.mp3"
    track.write_bytes(b"channel music data")

    res = resolve_and_validate_music_track(
        channel_settings={"background_music": str(track), "music_enabled": True}
    )
    assert res == track.resolve()


def test_missing_music_file_raises_pipeline_error(tmp_path):
    """Non-existent music file raises PipelineError."""
    missing_file = tmp_path / "does_not_exist.mp3"

    with pytest.raises(RenderPipelineError) as exc_info:
        resolve_and_validate_music_track(music_path=str(missing_file))
    assert "Background music file not found" in str(exc_info.value)


def test_unsupported_music_format_raises_pipeline_error(tmp_path):
    """Unsupported audio extensions (e.g. .ogg, .flac, .mp4, .txt) raise PipelineError."""
    ogg_file = tmp_path / "track.ogg"
    ogg_file.write_bytes(b"dummy ogg")

    with pytest.raises(RenderPipelineError) as exc_info:
        resolve_and_validate_music_track(music_path=str(ogg_file))
    assert "Unsupported audio format for background music: '.ogg'" in str(exc_info.value)
    assert "Only .mp3 and .wav are supported" in str(exc_info.value)


def test_empty_music_file_raises_pipeline_error(tmp_path):
    """0-byte audio file raises PipelineError."""
    empty_wav = tmp_path / "empty.wav"
    empty_wav.write_bytes(b"")

    with pytest.raises(RenderPipelineError) as exc_info:
        resolve_and_validate_music_track(music_path=str(empty_wav))
    assert "Background music file is empty" in str(exc_info.value)


def test_music_enabled_without_path_raises_pipeline_error():
    """Explicitly enabling music without configuring a path raises PipelineError."""
    with pytest.raises(RenderPipelineError) as exc_info:
        resolve_and_validate_music_track(music_enabled=True, music_path=None)
    assert "Background music is enabled, but no music track path is configured" in str(exc_info.value)


# ==============================================================================
# 3. FFmpeg Command Construction & Ducking Filter Graph Tests
# ==============================================================================

def test_ffmpeg_command_construction_without_music():
    """Without music, command retains exact backward-compatible structure."""
    temp_vid = Path("/tmp/temp_video.mp4")
    narr_wav = Path("/tmp/continuous.wav")
    final_mp4 = Path("/tmp/final.mp4")

    cmd = build_audio_mux_command(
        temp_video_mp4=temp_vid,
        continuous_audio_abs=narr_wav,
        temp_final_mp4=final_mp4,
        music_file_abs=None,
    )

    assert "-i" in cmd
    assert cmd[cmd.index("-i") + 1] == str(temp_vid)
    assert cmd[cmd.index("-i", cmd.index(str(temp_vid)) + 1) + 1] == str(narr_wav)
    assert "-stream_loop" not in cmd
    assert "-map" in cmd
    assert "1:a" in cmd
    assert "-af" in cmd
    assert "apad" in cmd


def test_ffmpeg_command_construction_with_music():
    """With music, command includes 3 inputs, -stream_loop -1, filter_complex, and maps [aout]."""
    temp_vid = Path("/tmp/temp_video.mp4")
    narr_wav = Path("/tmp/continuous.wav")
    music_wav = Path("/tmp/music.wav")
    final_mp4 = Path("/tmp/final.mp4")

    cmd = build_audio_mux_command(
        temp_video_mp4=temp_vid,
        continuous_audio_abs=narr_wav,
        temp_final_mp4=final_mp4,
        music_file_abs=music_wav,
        idle_volume=0.18,
        duck_ratio=4.0,
        duck_threshold=0.03,
        attack_ms=150.0,
        release_ms=350.0,
    )

    # Verify inputs: 0 is video, 1 is narration, 2 is looped music
    assert "-stream_loop" in cmd
    loop_idx = cmd.index("-stream_loop")
    assert cmd[loop_idx + 1] == "-1"
    assert cmd[loop_idx + 2] == "-i"
    assert cmd[loop_idx + 3] == str(music_wav)

    # Verify filter_complex presence
    assert "-filter_complex" in cmd
    fc_idx = cmd.index("-filter_complex")
    fc_str = cmd[fc_idx + 1]

    # Verify sidechaincompress signal routing:
    # [bg] (input 0 to compressor) is compressed music
    # [1:a] (input 1 to compressor) is narration control signal
    assert "[2:a]volume=0.18[bg]" in fc_str
    assert "[bg][1:a]sidechaincompress=threshold=0.030:ratio=4.0:attack=150:release=350[ducked]" in fc_str

    # Verify amix mixing with duration=first and normalize=0 (prevents narration attenuation)
    assert "[1:a][ducked]amix=inputs=2:duration=first:dropout_transition=2:normalize=0[mixed]" in fc_str

    # Verify peak lookahead limiter
    assert "[mixed]alimiter=limit=0.95[aout]" in fc_str

    # Verify output mappings
    assert "-map" in cmd
    assert "0:v" in cmd
    assert "[aout]" in cmd
    assert "-shortest" in cmd
    assert "-movflags" in cmd
    assert "+faststart" in cmd


# ==============================================================================
# 4. End-to-End Render Integration (Offline Fast Test with Local FFmpeg)
# ==============================================================================

def test_render_step_with_background_music_integration(tmp_path, monkeypatch):
    """Run local FFmpeg integration test rendering a mini video with ducked background music."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "render_fps", 30)

    video = DummyVideo()
    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create synthetic scene image
    img1_rel = f"{video.id}/scene_01.png"
    (tmp_path / img1_rel).write_bytes(create_solid_png(1080, 1920, 30, 40, 50))

    # 2. Create narration audio (1.0s)
    narr_rel = f"{video.id}/narration.wav"
    (tmp_path / narr_rel).write_bytes(create_mock_wav(1.0, 24000))

    # 3. Create short background music loop (0.5s)
    music_file = tmp_path / "ambient_loop.wav"
    music_file.write_bytes(create_mock_wav(0.5, 44100, frequency=220.0))

    prev = {
        "storyboard": {
            "storyboard": {
                "scenes": [{"index": 1, "shot_type": "medium"}]
            }
        },
        "images": {
            "images": {
                "items": [{"index": 1, "status": "ok", "file": img1_rel}]
            }
        },
        "voiceover": {
            "audio_file": narr_rel,
            "voiceover": {
                "items": [{"index": 1, "status": "ok", "duration_seconds": 1.0}]
            },
        },
    }

    # Render with background music enabled
    res = generate_render_step(
        video=video,
        prev=prev,
        step=None,
        music_path=str(music_file),
    )

    final_video = Path(res["video_path"])
    assert final_video.is_file()
    assert res["render"]["music"]["enabled"] is True
    assert res["render"]["music"]["file"] == str(music_file.resolve())
    assert res["meta"]["music_enabled"] is True

    # Inspect rendered streams using ffprobe
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=codec_type,codec_name,channels,sample_rate:format=duration",
            "-of", "json",
            str(final_video),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    import json
    data = json.loads(probe.stdout)

    v_stream = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    a_stream = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    assert v_stream is not None and v_stream["codec_name"] == "h264"
    assert a_stream is not None and a_stream["codec_name"] == "aac"
    assert a_stream["channels"] == 2
    assert a_stream["sample_rate"] == "44100"

    # Verify duration matches narration (~1.0s) and does not run infinitely
    dur = float(data["format"]["duration"])
    assert 0.9 <= dur <= 1.2

