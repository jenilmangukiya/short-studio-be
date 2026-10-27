"""Unit tests for the voiceover provider layer and voiceover step (100% offline, zero network)."""

import io
import uuid
import wave
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.models import Channel, Video, VideoStep
from app.pipeline import h_voiceover
from app.steps.voiceover import PipelineError, generate_voiceover_step
from app.voiceover import (
    BaseVoiceoverProvider,
    MockVoiceoverProvider,
    VoiceoverConfigError,
    VoiceoverError,
    VoiceoverProviderError,
    VoiceoverResult,
    create_mock_wav,
    get_voiceover_provider,
    trim_silence_samples,
    trim_silence_wav,
)


# ==============================================================================
# Helpers & Test Fixtures
# ==============================================================================

class DummyChannel:
    def __init__(self, voice=None, voice_speed=None):
        self.settings = {}
        if voice:
            self.settings["voice"] = voice
        if voice_speed is not None:
            self.settings["voice_speed"] = voice_speed


class DummyVideo:
    def __init__(self, video_id=None, channel=None):
        self.id = video_id or uuid.uuid4()
        self.topic = "The Mystery of Flight Trajectories"
        self.channel = channel


class DummyStep:
    def __init__(self, name="voiceover", user_prompt=None, output=None, status="pending"):
        self.name = name
        self.user_prompt = user_prompt
        self.output = output
        self.status = status


def _make_storyboard_prev(scene_count: int = 3, est_seconds_per_scene: float = 2.5) -> dict:
    scenes = []
    for i in range(1, scene_count + 1):
        scenes.append(
            {
                "index": i,
                "narration": f"This is scene {i} explaining airplane aerodynamics clearly.",
                "image_prompt": f"Airplane flight path scene {i}",
                "shot_type": "close-up",
                "uses_anchor": (i == 1),
                "est_seconds": est_seconds_per_scene,
            }
        )
    return {
        "storyboard": {
            "storyboard": {
                "visual_style": "Clean 3D render",
                "scenes": scenes,
                "total_seconds": round(scene_count * est_seconds_per_scene, 1),
            }
        }
    }


class CountingFakeVoiceoverProvider(BaseVoiceoverProvider):
    """Fake voiceover provider for testing call counts, scene tracking, and error injections."""

    def __init__(self, model="fake-voice-model", fail_on_scene=None, error_to_raise=None):
        super().__init__(model=model, timeout_seconds=10.0)
        self.fail_on_scene = fail_on_scene
        self.error_to_raise = error_to_raise
        self.call_count = 0
        self.called_narrations = []

    @property
    def provider_name(self) -> str:
        return "fake-voice-provider"

    def _execute_generate(
        self, text: str, voice: str, speed: float, target_duration: float | None = None
    ) -> tuple[bytes, int, float]:
        self.call_count += 1
        self.called_narrations.append(text)

        if self.fail_on_scene is not None and self.call_count == self.fail_on_scene:
            if self.error_to_raise:
                raise self.error_to_raise
            raise VoiceoverProviderError(f"Simulated voiceover failure on call {self.call_count}")

        dur = float(target_duration) if target_duration else 2.5
        wav = create_mock_wav(duration_seconds=dur, sample_rate=24000)
        return wav, 24000, dur


# ==============================================================================
# Unit Tests
# ==============================================================================

def test_mock_wav_generator():
    """Verify that create_mock_wav produces a structurally valid 24 kHz WAV stream."""
    wav_bytes = create_mock_wav(duration_seconds=1.5, sample_rate=24000)
    assert len(wav_bytes) > 44  # Standard WAV header is 44 bytes

    # Verify WAV header structure using Python standard library wave module
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 24000
        n_frames = wf.getnframes()
        assert abs((n_frames / 24000) - 1.5) < 0.01


def test_mock_provider_generation():
    """Verify MockVoiceoverProvider produces expected VoiceoverResult without network."""
    provider = MockVoiceoverProvider()
    assert provider.provider_name == "mock"

    res = provider.generate("Short narration test sentence.", voice="af_heart", speed=1.0)
    assert isinstance(res, VoiceoverResult)
    assert res.provider == "mock"
    assert res.sample_rate == 24000
    assert res.duration_seconds > 0.5
    assert len(res.audio_bytes) > 44


def test_voiceover_step_multi_scene_success(tmp_path, monkeypatch):
    """Verify successful end-to-end multi-scene voiceover generation."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = DummyVideo()
    prev = _make_storyboard_prev(scene_count=3, est_seconds_per_scene=2.0)
    step = DummyStep()
    provider = CountingFakeVoiceoverProvider()

    output = generate_voiceover_step(video, prev, step, voiceover_provider=provider)

    assert "voiceover" in output
    vo = output["voiceover"]
    assert vo["complete"] is True
    assert len(vo["items"]) == 3
    assert vo["total_duration_seconds"] == 6.0
    assert output["duration_seconds"] == 6.0

    meta = output["meta"]
    assert meta["scenes_total"] == 3
    assert meta["scenes_generated"] == 3
    assert meta["scenes_reused"] == 0
    assert meta["scenes_failed"] == 0

    # Verify all audio files exist on disk
    for it in vo["items"]:
        assert it["status"] == "ok"
        assert it["reused"] is False
        assert it["error"] is None
        abs_path = tmp_path / it["file"]
        assert abs_path.is_file()
        with wave.open(str(abs_path), "rb") as wf:
            assert wf.getframerate() == 24000


def test_voiceover_step_partial_failure_and_resume(tmp_path, monkeypatch):
    """Verify that a failure on scene 2 persists scene 1, marks scene 2 failed, and allows retry."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = DummyVideo()
    prev = _make_storyboard_prev(scene_count=3, est_seconds_per_scene=2.0)
    step = DummyStep()

    # Provider fails on scene 2
    failing_provider = CountingFakeVoiceoverProvider(fail_on_scene=2)

    with pytest.raises(PipelineError) as exc_info:
        generate_voiceover_step(video, prev, step, voiceover_provider=failing_provider)

    assert "failed for scenes: 2" in str(exc_info.value)

    # Check partial persistence in step.output
    assert step.output is not None
    vo = step.output["voiceover"]
    assert vo["complete"] is False
    assert len(vo["items"]) == 3
    assert vo["items"][0]["status"] == "ok"
    assert vo["items"][1]["status"] == "failed"
    assert "Simulated voiceover failure" in vo["items"][1]["error"]

    # Scene 1 file must exist on disk
    scene_1_file = tmp_path / vo["items"][0]["file"]
    assert scene_1_file.is_file()


def test_voiceover_step_retry_failed_scene_reuses_successful(tmp_path, monkeypatch):
    """Verify retrying a failed run reuses already generated audio and generates remaining."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = DummyVideo()
    prev = _make_storyboard_prev(scene_count=3, est_seconds_per_scene=2.0)
    step = DummyStep()

    # 1. Run with failure on scene 2
    failing_prov = CountingFakeVoiceoverProvider(fail_on_scene=2)
    with pytest.raises(PipelineError):
        generate_voiceover_step(video, prev, step, voiceover_provider=failing_prov)

    # 2. Retry with a working provider
    working_prov = CountingFakeVoiceoverProvider()
    retry_output = generate_voiceover_step(video, prev, step, voiceover_provider=working_prov)

    vo = retry_output["voiceover"]
    assert vo["complete"] is True
    # Scene 1 was reused
    assert vo["items"][0]["status"] == "ok"
    assert vo["items"][0]["reused"] is True
    # Scene 2 was newly generated
    assert vo["items"][1]["status"] == "ok"
    assert vo["items"][1]["reused"] is False
    # Scene 3 succeeded in run 1 and was reused in retry
    assert vo["items"][2]["status"] == "ok"
    assert vo["items"][2]["reused"] is True

    # Provider was called ONLY for the failed scene 2 (call_count == 1)
    assert working_prov.call_count == 1
    assert retry_output["meta"]["scenes_reused"] == 2
    assert retry_output["meta"]["scenes_generated"] == 1


def test_voiceover_step_fatal_config_error_aborts_immediately(tmp_path, monkeypatch):
    """Verify that a fatal VoiceoverConfigError aborts immediately and does not process subsequent scenes."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = DummyVideo()
    prev = _make_storyboard_prev(scene_count=3, est_seconds_per_scene=2.0)
    step = DummyStep()

    # Provider raises fatal VoiceoverConfigError on scene 2
    fatal_prov = CountingFakeVoiceoverProvider(
        fail_on_scene=2, error_to_raise=VoiceoverConfigError("Missing Kokoro ONNX model")
    )
    with pytest.raises(PipelineError) as exc_info:
        generate_voiceover_step(video, prev, step, voiceover_provider=fatal_prov)

    assert "Missing Kokoro ONNX model" in str(exc_info.value)
    # Only scene 1 was saved; scene 2 failed fatally and scene 3 was not attempted
    vo = step.output["voiceover"]
    assert len(vo["items"]) == 1
    assert vo["items"][0]["status"] == "ok"

    # Retry with working provider: scene 1 is reused, scenes 2 and 3 are generated
    working_prov = CountingFakeVoiceoverProvider()
    retry_output = generate_voiceover_step(video, prev, step, voiceover_provider=working_prov)
    assert retry_output["voiceover"]["complete"] is True
    assert retry_output["meta"]["scenes_reused"] == 1
    assert retry_output["meta"]["scenes_generated"] == 2
    assert working_prov.call_count == 2


def test_voiceover_step_reuse_when_narration_unchanged(tmp_path, monkeypatch):
    """Verify that re-running voiceover with identical narration reuses all scenes with 0 provider calls."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = DummyVideo()
    prev = _make_storyboard_prev(scene_count=2, est_seconds_per_scene=2.0)
    step = DummyStep()

    # Initial successful run
    prov1 = CountingFakeVoiceoverProvider()
    out1 = generate_voiceover_step(video, prev, step, voiceover_provider=prov1)
    assert prov1.call_count == 2
    step.output = out1

    # Second run with same input
    prov2 = CountingFakeVoiceoverProvider()
    out2 = generate_voiceover_step(video, prev, step, voiceover_provider=prov2)

    assert prov2.call_count == 0  # Reused all!
    assert out2["meta"]["scenes_reused"] == 2
    assert out2["meta"]["scenes_generated"] == 0
    assert all(it["reused"] is True for it in out2["voiceover"]["items"])


def test_voiceover_step_regenerates_when_voice_changes(tmp_path, monkeypatch):
    """Verify changing channel voice setting alters hash and regenerates audio."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    channel = DummyChannel(voice="af_heart")
    video = DummyVideo(channel=channel)
    prev = _make_storyboard_prev(scene_count=2)
    step = DummyStep()

    # 1. Initial run with af_heart
    prov1 = CountingFakeVoiceoverProvider()
    out1 = generate_voiceover_step(video, prev, step, voiceover_provider=prov1)
    step.output = out1
    assert out1["meta"]["voice"] == "af_heart"

    # 2. Change channel voice to am_adam
    channel.settings["voice"] = "am_adam"
    prov2 = CountingFakeVoiceoverProvider()
    out2 = generate_voiceover_step(video, prev, step, voiceover_provider=prov2)

    assert prov2.call_count == 2
    assert out2["meta"]["voice"] == "am_adam"
    assert out2["meta"]["scenes_reused"] == 0
    assert all(it["reused"] is False for it in out2["voiceover"]["items"])


def test_audio_serving_endpoint(tmp_path, monkeypatch):
    """Verify GET /videos/{video_id}/audio/{index} endpoint authentication, headers, and security."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video_id = uuid.uuid4()
    channel_id = uuid.uuid4()
    channel = Channel(id=channel_id, user_id=uuid.uuid4(), name="Test Channel")
    video = Video(id=video_id, channel_id=channel_id, topic="Serving test")
    video.channel = channel

    # Create dummy audio file on disk
    rel_path = f"{video_id}/audio/scene_01_testhash.wav"
    abs_path = tmp_path / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    wav_bytes = create_mock_wav(1.0, 24000)
    abs_path.write_bytes(wav_bytes)

    vo_step = VideoStep(
        video_id=video_id,
        name="voiceover",
        status="done",
        output={
            "voiceover": {
                "items": [
                    {
                        "index": 1,
                        "status": "ok",
                        "file": rel_path,
                        "audio_hash": "testhash123456",
                    },
                    {
                        "index": 2,
                        "status": "failed",
                        "file": None,
                        "audio_hash": "failhash",
                    },
                ]
            }
        },
    )
    video.steps = [vo_step]

    from app.deps import get_owned_video

    client = TestClient(app)

    # 1. Successful fetch by authorized owner
    app.dependency_overrides[get_owned_video] = lambda: video
    try:
        resp = client.get(f"/videos/{video_id}/audio/1")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert resp.headers["etag"] == '"testhash123456"'
        assert "no-cache" in resp.headers["cache-control"]
        assert resp.content == wav_bytes

        # 2. Failed scene returns 404
        resp_failed = client.get(f"/videos/{video_id}/audio/2")
        assert resp_failed.status_code == 404
        assert "No generated audio available" in resp_failed.json()["detail"]

        # 3. Non-existent scene index returns 404
        resp_missing = client.get(f"/videos/{video_id}/audio/99")
        assert resp_missing.status_code == 404

        # 4. Unauthorized / other user returns 404
        from fastapi import HTTPException

        def mock_other_user():
            raise HTTPException(404, "Video not found")

        app.dependency_overrides[get_owned_video] = mock_other_user
        resp_unowned = client.get(f"/videos/{video_id}/audio/1")
        assert resp_unowned.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_audio_serving_path_traversal_forbidden(tmp_path, monkeypatch):
    """Ensure path traversal attacks on audio serving are rejected with 403."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = Video(id=uuid.uuid4(), channel_id=uuid.uuid4(), topic="Traversal test")
    vo_step = VideoStep(
        video_id=video.id,
        name="voiceover",
        status="done",
        output={
            "voiceover": {
                "items": [
                    {
                        "index": 1,
                        "status": "ok",
                        "file": "../../etc/passwd",
                        "audio_hash": "traversalhash",
                    }
                ]
            }
        },
    )
    video.steps = [vo_step]

    from app.deps import get_owned_video
    app.dependency_overrides[get_owned_video] = lambda: video
    client = TestClient(app)
    try:
        resp = client.get(f"/videos/{video.id}/audio/1")
        assert resp.status_code == 403
        assert "forbidden" in resp.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()


def test_precondition_missing_storyboard():
    """Ensure voiceover raises PipelineError if storyboard output is missing."""
    video = DummyVideo()
    step = DummyStep()
    with pytest.raises(PipelineError) as exc_info:
        generate_voiceover_step(video, prev={}, step=step)
    assert "Storyboard step must be completed" in str(exc_info.value)


def test_h_voiceover_pipeline_handler(tmp_path, monkeypatch):
    """Verify h_voiceover integration conforms to pipeline handler signature."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "voiceover_provider", "mock")

    video = DummyVideo()
    prev = _make_storyboard_prev(scene_count=2, est_seconds_per_scene=2.5)

    res = h_voiceover(video, prev, None)
    assert res["mock"] is True
    assert res["duration_seconds"] == 5.0
    assert "voiceover" in res
    assert len(res["voiceover"]["items"]) == 2


# ==============================================================================
# Silence Trimming & Audio Prosody Tests
# ==============================================================================

def test_trim_silence_samples_terminal_and_non_terminal():
    """Verify that excessive leading/trailing silence is cleanly trimmed with proper headroom."""
    import numpy as np

    sr = 24000
    # 400ms lead silence + 1000ms 440Hz tone + 600ms trail silence = 2.0s
    lead = np.zeros(int(0.4 * sr), dtype=np.float32)
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * np.linspace(0, 1.0, int(1.0 * sr), dtype=np.float32))
    trail = np.zeros(int(0.6 * sr), dtype=np.float32)
    full = np.concatenate([lead, tone, trail])

    # 1. Terminal sentence trimming (. ! ?) keeps ~35ms lead + ~110ms trail = 1.145s
    trimmed_term = trim_silence_samples(full, sample_rate=sr, is_terminal=True)
    dur_term = len(trimmed_term) / sr
    assert 1.13 <= dur_term <= 1.16, f"Expected ~1.145s terminal duration, got {dur_term:.3f}s"

    # 2. Non-terminal clause trimming (, ; :) keeps ~35ms lead + ~80ms trail = 1.115s
    trimmed_nonterm = trim_silence_samples(full, sample_rate=sr, is_terminal=False)
    dur_nonterm = len(trimmed_nonterm) / sr
    assert 1.10 <= dur_nonterm <= 1.13, f"Expected ~1.115s non-terminal duration, got {dur_nonterm:.3f}s"

    # 3. Micro-fades prevent edge clicks: boundaries should smoothly taper towards 0
    assert abs(trimmed_term[0]) < 0.05
    assert abs(trimmed_term[-1]) < 0.05


def test_trim_silence_samples_edge_cases():
    """Verify edge cases: empty audio, pure silence, and ultra-short signals."""
    import numpy as np

    sr = 24000
    # Empty audio
    empty = np.array([], dtype=np.float32)
    assert len(trim_silence_samples(empty, sr)) == 0

    # Pure silence (no speech energy above threshold)
    silence = np.zeros(sr, dtype=np.float32)
    trimmed_silence = trim_silence_samples(silence, sr)
    assert len(trimmed_silence) == len(silence)


def test_trim_silence_wav_round_trip():
    """Verify WAV byte stream silence trimming produces valid WAV format and accurate duration."""
    import io
    import soundfile as sf
    import numpy as np

    sr = 24000
    lead = np.zeros(int(0.3 * sr), dtype=np.float32)
    tone = 0.4 * np.sin(2 * np.pi * 440.0 * np.linspace(0, 0.8, int(0.8 * sr), dtype=np.float32))
    trail = np.zeros(int(0.5 * sr), dtype=np.float32)
    raw_audio = np.concatenate([lead, tone, trail])

    buf = io.BytesIO()
    sf.write(buf, raw_audio, sr, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    trimmed_bytes, out_sr, dur = trim_silence_wav(wav_bytes, is_terminal=True)
    assert out_sr == sr
    # Expected duration: 0.035 lead + 0.8 tone + 0.110 trail = ~0.945s
    assert 0.93 <= dur <= 0.96
    assert len(trimmed_bytes) < len(wav_bytes)

    # Verify output bytes can be parsed by soundfile
    read_data, read_sr = sf.read(io.BytesIO(trimmed_bytes))
    assert read_sr == sr
    assert round(len(read_data) / float(read_sr), 3) == dur


def test_silence_trimming_subtitle_synchronization(tmp_path):
    """Verify that trimming silence from audio maintains exact acoustic subtitle synchronization."""
    import io
    import soundfile as sf
    import numpy as np
    from app.subtitles.acoustic import _align_scene_audio

    sr = 24000
    lead = np.zeros(int(0.35 * sr), dtype=np.float32)
    # 3 tone pulses representing 3 words: "airplane", "window", "hole"
    w1 = 0.5 * np.sin(2 * np.pi * 300.0 * np.linspace(0, 0.3, int(0.3 * sr), dtype=np.float32))
    pause1 = np.zeros(int(0.08 * sr), dtype=np.float32)
    w2 = 0.5 * np.sin(2 * np.pi * 400.0 * np.linspace(0, 0.25, int(0.25 * sr), dtype=np.float32))
    pause2 = np.zeros(int(0.08 * sr), dtype=np.float32)
    w3 = 0.5 * np.sin(2 * np.pi * 500.0 * np.linspace(0, 0.3, int(0.3 * sr), dtype=np.float32))
    trail = np.zeros(int(0.55 * sr), dtype=np.float32)

    raw_audio = np.concatenate([lead, w1, pause1, w2, pause2, w3, trail])
    buf = io.BytesIO()
    sf.write(buf, raw_audio, sr, format="WAV", subtype="PCM_16")

    trimmed_bytes, out_sr, trimmed_dur = trim_silence_wav(buf.getvalue(), is_terminal=True)

    wav_file = tmp_path / "scene_trimmed.wav"
    wav_file.write_bytes(trimmed_bytes)

    narration = "Airplane window hole."
    actual_dur, word_ts = _align_scene_audio(
        audio_path=wav_file,
        narration=narration,
        duration_fallback=trimmed_dur,
        scene_offset=0.0,
    )

    # 1. Actual duration matches trimmed audio duration
    assert round(actual_dur, 3) == round(trimmed_dur, 3)

    # 2. Word timestamps are within trimmed duration bounds
    assert len(word_ts) == 3
    assert word_ts[0].text == "Airplane"
    assert word_ts[1].text == "window"
    assert word_ts[2].text == "hole"

    # 3. First word starts near lead pad headroom (~0.035s)
    assert 0.0 <= word_ts[0].start <= 0.08
    # 4. Final word ends before total trimmed duration
    assert word_ts[2].end <= actual_dur
