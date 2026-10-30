"""Unit and integration tests for continuous narration generation in Shorts Studio.

Verifies:
1. Collecting narration from scenes in the correct order.
2. Rejecting empty or invalid narration.
3. Producing a single valid WAV asset from a multi-scene script.
4. Correctly combining chunks when chunking is necessary.
5. Preserving scene-to-narration associations and timing metadata.
6. Handling TTS failures without incorrectly marking the step successful.
7. Ensuring retries do not create inconsistent pipeline state and support idempotent reuse.
8. Preserving compatibility with downstream pipeline stages (subtitles, render, API serving).
"""

import io
from pathlib import Path
import uuid
import wave
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.models import Channel, Video, VideoStep
from app.steps.voiceover import PipelineError, generate_voiceover_step
from app.voiceover import (
    BaseVoiceoverProvider,
    VoiceoverConfigError,
    VoiceoverProviderError,
    VoiceoverResult,
    collect_narration_script,
    concatenate_audio_segments,
    create_mock_wav,
    estimate_scene_timings,
    generate_continuous_audio,
    split_script_into_chunks,
)


class DummyVideo:
    def __init__(self, video_id=None, channel=None):
        self.id = video_id or uuid.uuid4()
        self.topic = "The Mystery of Airplane Bleed Holes"
        self.channel = channel


class DummyStep:
    def __init__(self, name="voiceover", user_prompt=None, output=None, status="pending"):
        self.name = name
        self.user_prompt = user_prompt
        self.output = output
        self.status = status


class MockContinuousProvider(BaseVoiceoverProvider):
    """Predictable mock voiceover provider for unit testing continuous audio generation."""

    def __init__(self, model="mock-kokoro-model", fail_on_call=None, error_to_raise=None):
        super().__init__(model=model, timeout_seconds=10.0)
        self.fail_on_call = fail_on_call
        self.error_to_raise = error_to_raise
        self.call_count = 0
        self.called_narrations: list[str] = []

    @property
    def provider_name(self) -> str:
        return "mock-continuous-provider"

    def _execute_generate(
        self, text: str, voice: str, speed: float, target_duration: float | None = None
    ) -> tuple[bytes, int, float]:
        self.call_count += 1
        self.called_narrations.append(text)

        if self.fail_on_call is not None and self.call_count == self.fail_on_call:
            if self.error_to_raise:
                raise self.error_to_raise
            raise VoiceoverProviderError(f"Simulated voiceover error on call {self.call_count}")

        if target_duration is not None and float(target_duration) > 0:
            dur = float(target_duration)
        else:
            words = text.split()
            dur = max(0.5, round(len(words) / 2.5, 3))

        sample_rate = 24000
        wav_bytes = create_mock_wav(duration_seconds=dur, sample_rate=sample_rate)
        return wav_bytes, sample_rate, dur


def _make_storyboard_scenes(scenes_data: list[dict]) -> dict:
    return {
        "storyboard": {
            "storyboard": {
                "visual_style": "Cinematic 3D",
                "scenes": scenes_data,
                "total_seconds": sum(s.get("est_seconds", 2.5) for s in scenes_data),
            }
        }
    }


# ==============================================================================
# Requirement 1: Collecting narration from scenes in correct order
# ==============================================================================

def test_collect_narration_script_order_and_preservation():
    """Verify that narration is collected strictly in ascending scene index order,
    preserving exact wording, punctuation, and scene boundaries."""
    scenes = [
        {"index": 3, "narration": "That tiny opening is called a bleed hole.", "est_seconds": 2.5},
        {"index": 1, "narration": "Why do airplane windows have tiny holes?", "est_seconds": 2.5},
        {"index": 2, "narration": "It is not a manufacturing defect.", "est_seconds": 2.0},
    ]

    full_script, validated = collect_narration_script(scenes)

    # Verify scene order is strictly 1, 2, 3
    assert [s["index"] for s in validated] == [1, 2, 3]

    expected_script = (
        "Why do airplane windows have tiny holes? "
        "It is not a manufacturing defect. "
        "That tiny opening is called a bleed hole."
    )
    assert full_script == expected_script
    assert full_script.startswith("Why do airplane windows")
    assert full_script.endswith("called a bleed hole.")


# ==============================================================================
# Requirement 2: Rejecting empty or invalid narration
# ==============================================================================

def test_reject_empty_or_invalid_narration():
    """Verify that scenes with missing or empty narration text are rejected with PipelineError."""
    # 1. Empty string narration
    scenes_empty = [
        {"index": 1, "narration": "First scene looks great.", "est_seconds": 2.0},
        {"index": 2, "narration": "   ", "est_seconds": 2.0},
    ]
    with pytest.raises(PipelineError) as exc_info:
        generate_voiceover_step(DummyVideo(), _make_storyboard_scenes(scenes_empty))
    assert "Scene 2 narration text is empty or missing" in str(exc_info.value)

    # 2. None narration
    scenes_none = [
        {"index": 1, "narration": None, "est_seconds": 2.0},
    ]
    with pytest.raises(PipelineError) as exc_info:
        generate_voiceover_step(DummyVideo(), _make_storyboard_scenes(scenes_none))
    assert "Scene 1 narration text is empty or missing" in str(exc_info.value)

    # 3. Missing storyboard scenes
    with pytest.raises(PipelineError) as exc_info:
        generate_voiceover_step(DummyVideo(), {"storyboard": {"storyboard": {"scenes": []}}})
    assert "contain scenes" in str(exc_info.value)

    # 4. Missing storyboard step entirely
    with pytest.raises(PipelineError) as exc_info:
        generate_voiceover_step(DummyVideo(), {})
    assert "Storyboard step must be completed" in str(exc_info.value)


# ==============================================================================
# Requirement 3: Producing a single valid WAV asset from a multi-scene script
# ==============================================================================

def test_produces_single_valid_continuous_wav_asset(tmp_path, monkeypatch):
    """Verify that a multi-scene storyboard produces a single primary continuous WAV file on disk."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = DummyVideo()
    scenes = [
        {"index": 1, "narration": "Why do airplane windows have tiny holes?", "est_seconds": 2.5},
        {"index": 2, "narration": "It is not a manufacturing flaw.", "est_seconds": 2.0},
        {"index": 3, "narration": "It balances atmospheric pressure at 30,000 feet.", "est_seconds": 3.0},
    ]
    prev = _make_storyboard_scenes(scenes)
    provider = MockContinuousProvider()

    output = generate_voiceover_step(video, prev, step=DummyStep(), voiceover_provider=provider)

    # Exactly 1 provider call for the entire video
    assert provider.call_count == 1
    assert "voiceover" in output
    vo = output["voiceover"]

    assert vo["continuous"] is True
    assert vo["complete"] is True
    assert vo["total_duration_seconds"] == 7.5
    assert output["duration_seconds"] == 7.5

    # Check primary continuous WAV file
    cont_rel = vo["audio_file"]
    assert cont_rel == output["audio_file"]
    cont_abs = tmp_path / cont_rel
    assert cont_abs.is_file()

    with wave.open(str(cont_abs), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 24000
        n_frames = wf.getnframes()
        assert n_frames > 0
        actual_dur = round(n_frames / 24000.0, 3)
        assert abs(actual_dur - 7.5) < 0.05


# ==============================================================================
# Requirement 4: Correctly combining chunks when chunking is necessary
# ==============================================================================

def test_chunking_and_audio_concatenation():
    """Verify linguistic chunking splits at natural sentence/paragraph boundaries
    and audio concatenation combines chunks into a single continuous stream."""
    script = (
        "Why do airplane windows have tiny holes? It is not a manufacturing defect.\n\n"
        "That opening is called a bleed hole. It keeps the cabin pressurized and prevents fogging."
    )

    chunks = split_script_into_chunks(script, max_words=10)
    assert len(chunks) >= 2

    # Verify no words are lost or duplicated across chunks
    reconstructed_words = " ".join(chunks).split()
    original_words = script.split()
    assert reconstructed_words == original_words

    # Test audio concatenation
    sr = 24000
    seg1 = create_mock_wav(duration_seconds=1.0, sample_rate=sr)
    seg2 = create_mock_wav(duration_seconds=1.5, sample_rate=sr)

    comb_bytes, out_sr, dur = concatenate_audio_segments([(seg1, sr), (seg2, sr)])
    assert out_sr == sr
    assert dur == 2.5
    assert len(comb_bytes) == len(seg1) + len(seg2) - 44  # Single 44-byte header for concatenated data

    with wave.open(io.BytesIO(comb_bytes), "rb") as wf:
        assert wf.getframerate() == sr
        assert wf.getnchannels() == 1
        assert round(wf.getnframes() / float(sr), 1) == 2.5


# ==============================================================================
# Requirement 5: Preserving scene-to-narration associations
# ==============================================================================

def test_preserve_scene_to_narration_associations(tmp_path, monkeypatch):
    """Verify that scene timing metadata is preserved, start/end times are monotonic,
    and no separate per-scene WAV files are written to disk."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = DummyVideo()
    scenes = [
        {"index": 1, "id": "sc-001", "narration": "Why do airplane windows have tiny holes?", "est_seconds": 2.5},
        {"index": 2, "id": "sc-002", "narration": "It is not a manufacturing defect.", "est_seconds": 2.0},
        {"index": 3, "id": "sc-003", "narration": "That tiny opening is called a bleed hole.", "est_seconds": 2.5},
    ]
    prev = _make_storyboard_scenes(scenes)
    provider = MockContinuousProvider()

    output = generate_voiceover_step(video, prev, step=DummyStep(), voiceover_provider=provider)
    items = output["voiceover"]["items"]
    assert len(items) == 3

    # Check timing progression
    assert items[0]["index"] == 1
    assert items[0]["scene_id"] == "sc-001"
    assert items[0]["start_time"] == 0.0
    assert items[0]["end_time"] > 0.0
    assert items[0]["timing_type"] == "estimated"

    assert items[1]["index"] == 2
    assert items[1]["start_time"] == items[0]["end_time"]
    assert items[1]["end_time"] > items[1]["start_time"]

    assert items[2]["index"] == 3
    assert items[2]["start_time"] == items[1]["end_time"]
    assert items[2]["end_time"] == 7.0  # Total duration (2.5 + 2.0 + 2.5)

    # Confirm no per-scene WAV files are referenced or created on disk
    for it in items:
        assert "file" not in it

    audio_dir = tmp_path / str(video.id) / "audio"
    wav_files = list(audio_dir.glob("*.wav"))
    assert len(wav_files) == 1
    assert wav_files[0].name.startswith("narration_")


# ==============================================================================
# Requirement 6: Handling TTS failures without marking step successful
# ==============================================================================

def test_tts_failure_marks_step_failed(tmp_path, monkeypatch):
    """Verify that a provider inference failure raises PipelineError and persists failure status."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = DummyVideo()
    scenes = [
        {"index": 1, "narration": "Why do airplane windows have tiny holes?", "est_seconds": 2.5},
    ]
    prev = _make_storyboard_scenes(scenes)
    step = DummyStep()

    failing_provider = MockContinuousProvider(fail_on_call=1)

    with pytest.raises(PipelineError) as exc_info:
        generate_voiceover_step(video, prev, step=step, voiceover_provider=failing_provider)

    assert "Simulated voiceover error" in str(exc_info.value)
    # Must persist failure state in step.output without marking complete
    assert step.output is not None
    assert step.output["voiceover"]["complete"] is False
    assert "Simulated voiceover error" in step.output["error"]


# ==============================================================================
# Requirement 7: Retries do not create inconsistent state & reuse is idempotent
# ==============================================================================

def test_retry_after_failure_succeeds_consistently(tmp_path, monkeypatch):
    """Verify that retrying a failed step succeeds cleanly and subsequent runs reuse cached assets."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = DummyVideo()
    scenes = [
        {"index": 1, "narration": "Why do airplane windows have tiny holes?", "est_seconds": 2.5},
        {"index": 2, "narration": "It is not a manufacturing defect.", "est_seconds": 2.0},
    ]
    prev = _make_storyboard_scenes(scenes)
    step = DummyStep()

    # 1. Initial attempt fails
    failing_provider = MockContinuousProvider(fail_on_call=1)
    with pytest.raises(PipelineError):
        generate_voiceover_step(video, prev, step=step, voiceover_provider=failing_provider)

    # 2. Retry with healthy provider succeeds cleanly
    working_provider = MockContinuousProvider()
    retry_output = generate_voiceover_step(video, prev, step=step, voiceover_provider=working_provider)

    assert working_provider.call_count == 1
    assert retry_output["voiceover"]["complete"] is True
    assert retry_output["meta"]["scenes_generated"] == 2
    assert retry_output["meta"]["scenes_reused"] == 0

    # 3. Third run with unchanged storyboard reuses cached audio with 0 provider calls
    third_provider = MockContinuousProvider()
    third_output = generate_voiceover_step(video, prev, step=step, voiceover_provider=third_provider)

    assert third_provider.call_count == 0
    assert third_output["voiceover"]["complete"] is True
    assert third_output["meta"]["scenes_reused"] == 2
    assert third_output["meta"]["scenes_generated"] == 0
    assert all(it["reused"] is True for it in third_output["voiceover"]["items"])


# ==============================================================================
# Requirement 8: Compatibility with downstream pipeline stages & API endpoints
# ==============================================================================

def test_downstream_compatibility_subtitles_and_endpoints(tmp_path, monkeypatch):
    """Verify that continuous narration output is served via GET /videos/{video_id}/audio
    and legacy per-scene audio endpoint /videos/{video_id}/audio/{index} is removed."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video_id = uuid.uuid4()
    channel_id = uuid.uuid4()
    channel = Channel(id=channel_id, user_id=uuid.uuid4(), name="Test Aerodynamics")
    video = Video(id=video_id, channel_id=channel_id, topic="Bleed Holes")
    video.channel = channel

    scenes = [
        {"index": 1, "narration": "Why do airplane windows have tiny holes?", "est_seconds": 2.5},
        {"index": 2, "narration": "It is not a defect.", "est_seconds": 2.0},
    ]
    prev = _make_storyboard_scenes(scenes)
    provider = MockContinuousProvider()

    vo_step = VideoStep(video_id=video_id, name="voiceover", status="pending")
    vo_output = generate_voiceover_step(video, prev, step=vo_step, voiceover_provider=provider)
    vo_step.output = vo_output
    vo_step.status = "done"
    video.steps = [vo_step]

    # Verify GET /videos/{video_id}/audio serves continuous WAV
    from app.deps import get_owned_video
    app.dependency_overrides[get_owned_video] = lambda: video
    client = TestClient(app)

    try:
        resp_cont = client.get(f"/videos/{video_id}/audio")
        assert resp_cont.status_code == 200
        assert resp_cont.headers["content-type"] == "audio/wav"
        with wave.open(io.BytesIO(resp_cont.content), "rb") as wf:
            assert wf.getframerate() == 24000
            assert wf.getnchannels() == 1

        # Verify legacy GET /videos/{video_id}/audio/1 endpoint is removed (404)
        resp_sc1 = client.get(f"/videos/{video_id}/audio/1")
        assert resp_sc1.status_code == 404
    finally:
        app.dependency_overrides.clear()


# ==============================================================================
# Requirement 9: Balanced chunking, inter-chunk breath gaps, and TTS normalization
# ==============================================================================

def test_balanced_chunking_avoids_orphan_tail_chunk():
    """Verify split_script_into_chunks balances multi-chunk splits instead of leaving a tiny tail chunk."""
    sentences = [
        "Why do people yawn when someone else yawns?",
        "Seeing a friend yawn can make your own jaw stretch within seconds.",
        "For years people blamed low oxygen in the blood, but experiments proved extra oxygen does not stop it.",
        "Instead, your brain automatically mirrors the facial actions of the people around you.",
        "When you spot a yawn, social mirroring networks fire up before you even think about it.",
        "That is why you are much more likely to catch a yawn from a close friend than a stranger.",
        "Even dogs can catch a yawn from the person sitting beside them.",
        "Next time you catch a yawn, your brain is quietly syncing with the room.",
    ]
    script = " ".join(sentences)
    chunks = split_script_into_chunks(script, max_words=65, max_chars=380)

    # 118 words / 685 chars should split into 2 balanced chunks (each ~50-65 words), not 3 with a tiny orphan
    assert len(chunks) == 2
    w_counts = [len(c.split()) for c in chunks]
    assert all(40 <= wc <= 65 for wc in w_counts)
    assert " ".join(chunks).split() == script.split()


def test_compute_inter_chunk_gaps_and_multi_chunk_pause_insertion():
    """Verify natural inter-chunk breath pauses are inserted when joining multiple chunks."""
    from app.voiceover import compute_inter_chunk_gaps

    chunks = [
        "First sentence group ends with a period.",
        "Second clause group ends with a comma,",
        "Final sentence group finishes the story.",
    ]
    gaps = compute_inter_chunk_gaps(chunks)
    assert gaps == [0.24, 0.09]

    provider = MockContinuousProvider()
    res, count = generate_continuous_audio(
        " ".join(chunks), provider=provider, max_chunk_words=8
    )
    assert count == 3
    # Sum of chunk durations (7/2.5=2.8 + 7/2.5=2.8 + 6/2.5=2.4 = 8.0s) + inter-chunk gaps (0.24 + 0.09 = 0.33s)
    expected_unpadded = sum(round(len(c.split()) / 2.5, 3) for c in chunks)
    assert res.duration_seconds > expected_unpadded
    assert abs(res.duration_seconds - (expected_unpadded + 0.33)) < 0.02


def test_normalize_text_for_tts():
    """Verify Kokoro TTS text normalization handles comma numbers, decimals, quotes, dashes, and abbreviations."""
    from app.voiceover import normalize_text_for_tts

    raw = (
        'At 30,000 feet—or 2.5 miles up—temperatures drop to 40°F vs. 70°C! '
        'Scientists call it a “diamond planet”... traveling at 1,200 mph (100%).'
    )
    norm = normalize_text_for_tts(raw)

    assert "30000" in norm
    assert "30,000" not in norm
    assert "2 point 5" in norm
    assert "degrees Fahrenheit" in norm
    assert "versus" in norm
    assert "degrees Celsius" in norm
    assert "“" not in norm and "”" not in norm and '"' not in norm
    assert "—" not in norm and "..." not in norm
    assert "1200 miles per hour" in norm
    assert "100 percent" in norm

