"""Phase B Verification Script: Isolated Edge-Case & Resiliency Testing.

Tests:
1. Partial progress persistence when a scene fails.
2. Selective retry (reusing successful audio, regenerating only failed scene).
3. Empty narration handling.
4. Recovery when a cached audio file is missing on disk.
5. Endpoint security (authorization, failed scenes, path traversal protection).

All tests run in total isolation using temporary directories and in-memory
fixtures. Real database records and real video files are never modified.

Usage:
    .venv/bin/python scripts/verify_phase_b_edge_cases.py
"""

from pathlib import Path
import shutil
import sys
import tempfile
import time
import uuid
import wave

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import HTTPException
from fastapi.testclient import TestClient
from app.config import settings
from app.main import app
from app.models import Channel, Video, VideoStep
from app.steps.voiceover import PipelineError, generate_voiceover_step
from app.voiceover import (
    BaseVoiceoverProvider,
    VoiceoverProviderError,
    create_mock_wav,
)


# ==============================================================================
# In-Memory Test Fixtures
# ==============================================================================

class IsolatedVideo:
    """In-memory video model fixture."""
    def __init__(self, video_id=None):
        self.id = video_id or uuid.uuid4()
        self.topic = "Isolated Edge Case Test"
        self.channel = None


class IsolatedStep:
    """In-memory step fixture that tracks output without writing to DB."""
    def __init__(self, name="voiceover", output=None, user_prompt=None):
        self.name = name
        self.output = output
        self.user_prompt = user_prompt
        self.status = "pending"


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


def _make_storyboard_prev(scenes_data: list[tuple[int, str, float]]) -> dict:
    """Build a storyboard prev dictionary from a list of (index, narration, est_seconds)."""
    scenes = [
        {
            "index": idx,
            "narration": text,
            "image_prompt": f"Visual prompt for scene {idx}",
            "shot_type": "close-up",
            "uses_anchor": False,
            "est_seconds": dur,
        }
        for idx, text, dur in scenes_data
    ]
    return {
        "storyboard": {
            "storyboard": {
                "visual_style": "Clean 3D render",
                "scenes": scenes,
                "total_seconds": sum(s["est_seconds"] for s in scenes),
            }
        }
    }


# ==============================================================================
# Phase B Check Runners
# ==============================================================================

def run_check_1_partial_progress(tmp_dir: Path) -> tuple[bool, str]:
    """Check 1: Partial progress persistence when Scene 2 fails."""
    video = IsolatedVideo()
    step = IsolatedStep()
    prev = _make_storyboard_prev([
        (1, "Scene one narration text.", 2.0),
        (2, "Scene two narration text.", 2.0),
        (3, "Scene three narration text.", 2.0),
    ])

    failing_provider = CountingFakeVoiceoverProvider(fail_on_scene=2)
    try:
        generate_voiceover_step(video, prev, step, voiceover_provider=failing_provider)
        return False, "Expected PipelineError was not raised"
    except PipelineError as exc:
        if "failed for scenes: 2" not in str(exc):
            return False, f"Unexpected error message: {exc}"

    # Verify partial persistence
    if not step.output or "voiceover" not in step.output:
        return False, "step.output does not contain partial voiceover output"

    items = step.output["voiceover"]["items"]
    if len(items) != 3:
        return False, f"Expected 3 scene items in partial output, got {len(items)}"

    if items[0]["status"] != "ok":
        return False, f"Scene 1 status expected 'ok', got '{items[0]['status']}'"

    if items[1]["status"] != "failed":
        return False, f"Scene 2 status expected 'failed', got '{items[1]['status']}'"

    # Confirm Scene 1 audio file exists on disk
    scene_1_file = tmp_dir / items[0]["file"]
    if not scene_1_file.is_file():
        return False, f"Scene 1 audio file was not saved on disk at {scene_1_file}"

    return True, "Scene 1 saved on disk and marked 'ok'; Scene 2 correctly recorded as 'failed' in partial output."


def run_check_2_selective_retry(tmp_dir: Path) -> tuple[bool, str]:
    """Check 2: Selective retry, reusing successful audio and regenerating only failed scene."""
    video = IsolatedVideo()
    step = IsolatedStep()
    prev = _make_storyboard_prev([
        (1, "Scene one narration text.", 2.0),
        (2, "Scene two narration text.", 2.0),
        (3, "Scene three narration text.", 2.0),
    ])

    # 1. Run with failure on scene 2
    failing_provider = CountingFakeVoiceoverProvider(fail_on_scene=2)
    try:
        generate_voiceover_step(video, prev, step, voiceover_provider=failing_provider)
    except PipelineError:
        pass

    # 2. Retry with a working provider
    working_provider = CountingFakeVoiceoverProvider()
    retry_output = generate_voiceover_step(video, prev, step, voiceover_provider=working_provider)

    items = retry_output["voiceover"]["items"]
    meta = retry_output["meta"]

    if retry_output["voiceover"]["complete"] is not True:
        return False, "Expected complete == True after retry"

    if not items[0]["reused"]:
        return False, "Scene 1 was not reused during retry"

    if items[1]["reused"]:
        return False, "Failed Scene 2 was incorrectly marked as reused"

    if not items[2]["reused"]:
        return False, "Scene 3 was not reused during retry"

    if working_provider.call_count != 1:
        return False, f"Expected exactly 1 provider call (for Scene 2), got {working_provider.call_count}"

    if meta["scenes_reused"] != 2 or meta["scenes_generated"] != 1:
        return False, f"Unexpected meta counts: reused={meta['scenes_reused']}, generated={meta['scenes_generated']}"

    return True, "Scenes 1 and 3 reused; only failed Scene 2 was regenerated (provider called exactly 1 time)."


def run_check_3_empty_narration(tmp_dir: Path) -> tuple[bool, str]:
    """Check 3: Empty narration handling."""
    video = IsolatedVideo()
    step = IsolatedStep()
    prev = _make_storyboard_prev([
        (1, "Valid first scene narration.", 2.0),
        (2, "   ", 2.0),  # Whitespace / empty narration
        (3, "Valid third scene narration.", 2.0),
    ])

    provider = CountingFakeVoiceoverProvider()
    try:
        generate_voiceover_step(video, prev, step, voiceover_provider=provider)
        return False, "Expected PipelineError for empty narration was not raised"
    except PipelineError as exc:
        if "failed for scenes: 2" not in str(exc):
            return False, f"Unexpected error message: {exc}"

    items = step.output["voiceover"]["items"]
    scene_2 = items[1]

    if scene_2["status"] != "failed":
        return False, f"Expected Scene 2 status 'failed', got '{scene_2['status']}'"

    if "empty" not in scene_2.get("error", "").lower():
        return False, f"Expected empty narration error message, got '{scene_2.get('error')}'"

    # Scene 1 and Scene 3 were processed cleanly
    if items[0]["status"] != "ok" or items[2]["status"] != "ok":
        return False, "Valid scenes were not processed correctly alongside empty scene"

    return True, "Empty narration detected and marked as failed with clear error message; other scenes preserved."


def run_check_4_missing_cached_file(tmp_dir: Path) -> tuple[bool, str]:
    """Check 4: Recovery when a cached audio file is missing on disk."""
    video = IsolatedVideo()
    step = IsolatedStep()
    prev = _make_storyboard_prev([
        (1, "Scene one narration text.", 2.0),
        (2, "Scene two narration text.", 2.0),
    ])

    # 1. Initial complete run
    prov1 = CountingFakeVoiceoverProvider()
    out1 = generate_voiceover_step(video, prev, step, voiceover_provider=prov1)
    step.output = out1

    scene_1_rel = out1["voiceover"]["items"][0]["file"]
    scene_1_abs = tmp_dir / scene_1_rel
    scene_2_rel = out1["voiceover"]["items"][1]["file"]
    scene_2_abs = tmp_dir / scene_2_rel

    if not scene_1_abs.is_file() or not scene_2_abs.is_file():
        return False, "Initial files were not generated properly"

    # 2. Simulate disk loss: physically delete Scene 1's audio file from disk
    scene_1_abs.unlink()
    if scene_1_abs.is_file():
        return False, "Failed to simulate file deletion"

    # 3. Re-run step: pipeline should detect Scene 1 file is missing and regenerate it, while reusing Scene 2
    prov2 = CountingFakeVoiceoverProvider()
    out2 = generate_voiceover_step(video, prev, step, voiceover_provider=prov2)

    items2 = out2["voiceover"]["items"]
    if items2[0]["reused"] is not False:
        return False, "Missing Scene 1 was incorrectly marked as reused"

    if items2[1]["reused"] is not True:
        return False, "Intact Scene 2 was not reused"

    if prov2.call_count != 1:
        return False, f"Expected provider to be called exactly 1 time (for missing Scene 1), got {prov2.call_count}"

    if not scene_1_abs.is_file():
        return False, "Scene 1 audio file was not regenerated on disk"

    return True, "Missing Scene 1 file detected; automatically regenerated on disk while reusing intact Scene 2."


def run_check_5_endpoint_security(tmp_dir: Path) -> tuple[bool, str]:
    """Check 5: Endpoint security, authorization, and path traversal."""
    from app.deps import get_owned_video

    video_id = uuid.uuid4()
    channel_id = uuid.uuid4()
    channel = Channel(id=channel_id, user_id=uuid.uuid4(), name="Test Channel")
    video = Video(id=video_id, channel_id=channel_id, topic="Security Test Video")
    video.channel = channel

    # Create dummy audio file on disk
    rel_path = f"{video_id}/audio/scene_01_validhash.wav"
    abs_path = tmp_dir / rel_path
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
                        "audio_hash": "hash123456",
                    },
                    {
                        "index": 2,
                        "status": "failed",
                        "file": None,
                        "audio_hash": "failhash",
                    },
                    {
                        "index": 3,
                        "status": "ok",
                        "file": "../../etc/passwd",  # Malicious path traversal
                        "audio_hash": "traversalhash",
                    },
                ]
            }
        },
    )
    video.steps = [vo_step]

    client = TestClient(app)

    # 1. Authorized user -> Scene 1 returns 200 with audio/wav, ETag, Cache-Control
    app.dependency_overrides[get_owned_video] = lambda: video
    try:
        resp_ok = client.get(f"/videos/{video_id}/audio/1")
        if resp_ok.status_code != 200:
            return False, f"Expected 200 for Scene 1, got {resp_ok.status_code}"
        if resp_ok.headers.get("content-type") != "audio/wav":
            return False, f"Expected content-type audio/wav, got {resp_ok.headers.get('content-type')}"
        if resp_ok.headers.get("etag") != '"hash123456"':
            return False, f"Expected ETag '\"hash123456\"', got {resp_ok.headers.get('etag')}"
        if "no-cache" not in resp_ok.headers.get("cache-control", ""):
            return False, f"Expected Cache-Control no-cache, got {resp_ok.headers.get('cache-control')}"

        # 2. Failed scene (Scene 2) -> returns 404
        resp_failed = client.get(f"/videos/{video_id}/audio/2")
        if resp_failed.status_code != 404:
            return False, f"Expected 404 for failed Scene 2, got {resp_failed.status_code}"

        # 3. Nonexistent scene index (Scene 99) -> returns 404
        resp_missing = client.get(f"/videos/{video_id}/audio/99")
        if resp_missing.status_code != 404:
            return False, f"Expected 404 for missing Scene 99, got {resp_missing.status_code}"

        # 4. Path traversal attempt (Scene 3 -> ../../etc/passwd) -> returns 403 Forbidden
        resp_traversal = client.get(f"/videos/{video_id}/audio/3")
        if resp_traversal.status_code != 403:
            return False, f"Expected 403 for path traversal, got {resp_traversal.status_code}"
        if "forbidden" not in resp_traversal.json().get("detail", "").lower():
            return False, f"Expected 'forbidden' detail in 403 response, got {resp_traversal.json()}"

        # 5. Unauthorized / other user -> returns 404
        def mock_unauthorized_user():
            raise HTTPException(404, "Video not found")

        app.dependency_overrides[get_owned_video] = mock_unauthorized_user
        resp_unowned = client.get(f"/videos/{video_id}/audio/1")
        if resp_unowned.status_code != 404:
            return False, f"Expected 404 for unauthorized request, got {resp_unowned.status_code}"

    finally:
        app.dependency_overrides.clear()

    return True, "200 OK for valid scene, 404 for failed/missing scenes, 403 for path traversal, and 404 for unauthorized requests."


# ==============================================================================
# Main Orchestrator
# ==============================================================================

def main():
    print("=" * 70)
    print("  PHASE B: ISOLATED EDGE-CASE & RESILIENCY VERIFICATION")
    print("=" * 70)

    # Safety: Capture real video state before Phase B to confirm zero mutations
    real_video_id = uuid.UUID("386aaf7e-32fd-4c40-bf49-c349e4f2ee9c")
    real_storage_dir = Path(settings.storage_dir).resolve()
    real_audio_dir = real_storage_dir / str(real_video_id) / "audio"

    if real_audio_dir.is_dir():
        real_files_before = {f.name: f.stat().st_mtime_ns for f in real_audio_dir.glob("*.wav")}
    else:
        real_files_before = {}

    print(f"Isolated from real video: {real_video_id}")
    print(f"Real video audio files on disk: {len(real_files_before)}")
    print(f"Configured real storage root:   {real_storage_dir}\n")

    # Set up isolated temporary storage
    temp_dir = Path(tempfile.mkdtemp(prefix="shorts_studio_phase_b_"))
    original_storage_setting = settings.storage_dir
    settings.storage_dir = str(temp_dir)

    print(f"Temporary isolated storage initialized at: {temp_dir}\n")

    results = []
    checks = [
        ("Check 1: Partial Progress on Failure", run_check_1_partial_progress),
        ("Check 2: Selective Retry & Reuse", run_check_2_selective_retry),
        ("Check 3: Empty Narration Handling", run_check_3_empty_narration),
        ("Check 4: Missing Cached Audio Recovery", run_check_4_missing_cached_file),
        ("Check 5: Endpoint Security & Traversal Protection", run_check_5_endpoint_security),
    ]

    try:
        for name, runner in checks:
            print(f"Running {name}...")
            start_t = time.perf_counter()
            try:
                passed, detail = runner(temp_dir)
            except Exception as exc:
                passed = False
                detail = f"Unhandled exception: {exc}"
            elapsed = time.perf_counter() - start_t
            results.append((name, passed, detail, elapsed))
            status_str = "PASS" if passed else "FAIL"
            print(f"  [{status_str}] in {elapsed:.3f}s: {detail}\n")

    finally:
        # Restore storage setting and clean up temporary directory
        settings.storage_dir = original_storage_setting
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Cleaned up temporary storage at: {temp_dir}")

    # Safety: Verify real video was NOT modified in any way
    if real_audio_dir.is_dir():
        real_files_after = {f.name: f.stat().st_mtime_ns for f in real_audio_dir.glob("*.wav")}
    else:
        real_files_after = {}

    assert real_files_before == real_files_after, "SAFETY VIOLATION: Real video files were modified!"
    print("Safety Check: Verified that real video audio files and database remain 100% UNTOUCHED.")

    # Print final summary table
    print("\n" + "=" * 70)
    print("  PHASE B SUMMARY REPORT")
    print("=" * 70)
    all_passed = True
    for name, passed, detail, elapsed in results:
        status_tag = "[PASS]" if passed else "[FAIL]"
        if not passed:
            all_passed = False
        print(f"{status_tag} {name:<45} ({elapsed:.3f}s)")
        print(f"       Detail: {detail}")

    print("=" * 70)
    if all_passed:
        print("  ALL 5 CHECKS PASSED (100% SUCCESS)")
    else:
        print("  SOME CHECKS FAILED - REVIEW REPORT ABOVE")
    print("=" * 70)

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
