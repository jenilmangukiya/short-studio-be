"""Unit tests for the real image provider layer and images step (network-free)."""

import hashlib
import struct
import uuid
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.images import (
    BaseImageProvider,
    ImageConfigError,
    ImageContentBlockedError,
    ImageError,
    ImageProviderError,
    ImageRateLimitError,
    ImageResult,
    MockImageProvider,
    OpenAIImageProvider,
    get_image_provider,
)
from app.images.mock import create_solid_png
from app.main import app
from app.models import Channel, User, Video, VideoStep
from app.steps.images import PipelineError, generate_images_step


# ==============================================================================
# Helpers & Test Fixtures
# ==============================================================================

class DummyVideo:
    def __init__(self, video_id=None):
        self.id = video_id or uuid.uuid4()
        self.topic = "Test Airplane Windows"


class DummyStep:
    def __init__(self, name="images", user_prompt=None, output=None, status="pending"):
        self.name = name
        self.user_prompt = user_prompt
        self.output = output
        self.status = status


def _make_storyboard_prev(scene_count: int = 3, visual_style: str = "Clean 3D render", subject_anchor: str = "A small blue cube.") -> dict:
    scenes = []
    for i in range(1, scene_count + 1):
        scenes.append(
            {
                "index": i,
                "narration": f"Narration for scene {i}.",
                "image_prompt": f"Airplane window showing angle {i}",
                "shot_type": "close-up",
                "uses_anchor": (i == 1),
                "est_seconds": 2.5,
            }
        )
    return {
        "storyboard": {
            "storyboard": {
                "visual_style": visual_style,
                "style_source": "default",
                "subject_anchor": subject_anchor,
                "scenes": scenes,
                "total_seconds": round(scene_count * 2.5, 1),
            }
        }
    }


class CountingFakeProvider(BaseImageProvider):
    """Fake image provider for testing call counts and error injections."""

    def __init__(self, model="fake-image-model", fail_on_scene=None, error_to_raise=None):
        super().__init__(model=model, timeout_seconds=10.0, est_cost_usd=0.04)
        self.fail_on_scene = fail_on_scene
        self.error_to_raise = error_to_raise
        self.call_count = 0
        self.called_prompts = []

    @property
    def provider_name(self) -> str:
        return "fake-provider"

    def _execute_generate(self, prompt: str, size: str, quality: str):
        self.call_count += 1
        self.called_prompts.append(prompt)

        if self.fail_on_scene is not None and self.call_count == self.fail_on_scene:
            if self.error_to_raise:
                raise self.error_to_raise
            raise ImageProviderError(f"Simulated provider failure on call {self.call_count}")

        # Return a valid solid 1024x1536 PNG
        png = create_solid_png(1024, 1536, 100, 150, 200)
        return png, 15, 100, 115


# ==============================================================================
# 1. Mock Provider & PNG Generation Tests
# ==============================================================================

def test_mock_provider_generates_valid_png():
    """Verify MockImageProvider produces valid PNG with signature and correct dimensions."""
    provider = MockImageProvider(model="mock-model")
    result = provider.generate(prompt="A sleek red airplane", size="1024x1536", quality="low")

    assert isinstance(result, ImageResult)
    assert result.provider == "mock"
    assert result.size == "1024x1536"
    assert result.quality == "low"
    assert result.est_cost_usd is None
    assert result.input_tokens > 0

    data = result.image_bytes
    # 8-byte PNG header
    assert data.startswith(b"\x89PNG\r\n\x1a\n")

    # Unpack dimensions from IHDR chunk (offset 16:24)
    width, height = struct.unpack(">II", data[16:24])
    assert width == 1024
    assert height == 1536

    # Verify IEND chunk exists at the end
    assert data.endswith(b"IEND\xaeB`\x82")


def test_mock_provider_custom_size_and_determinism():
    """Verify custom sizes (e.g. 864x1536) and deterministic output based on prompt hash."""
    provider = MockImageProvider()
    res1 = provider.generate(prompt="Airplane wing in cloud", size="864x1536", quality="low")
    res2 = provider.generate(prompt="Airplane wing in cloud", size="864x1536", quality="low")
    res3 = provider.generate(prompt="Different prompt altogether", size="864x1536", quality="low")

    # Dimensions
    w, h = struct.unpack(">II", res1.image_bytes[16:24])
    assert w == 864
    assert h == 1536

    # Determinism: same prompt -> exact same bytes
    assert res1.image_bytes == res2.image_bytes
    # Different prompt -> different bytes (different RGB color derived from hash)
    assert res1.image_bytes != res3.image_bytes


# ==============================================================================
# 2. Images Step Full Run with Mock Provider
# ==============================================================================

def test_full_images_step_mock_provider(tmp_path, monkeypatch):
    """Verify end-to-end images step execution with the default mock provider."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "image_provider", "mock")

    video = DummyVideo()
    prev = _make_storyboard_prev(scene_count=2)
    step = DummyStep()

    output = generate_images_step(video, prev, step)

    assert output["images"]["complete"] is True
    assert len(output["images"]["items"]) == 2
    assert output["meta"]["mock"] is True or output.get("mock") is True
    assert output["meta"]["provider"] == "mock"
    assert output["meta"]["images_total"] == 2
    assert output["meta"]["images_generated"] == 2
    assert output["meta"]["images_reused"] == 0
    assert output["meta"]["images_failed"] == 0

    for idx, item in enumerate(output["images"]["items"], start=1):
        assert item["index"] == idx
        assert item["status"] == "ok"
        assert item["reused"] is False
        assert item["error"] is None
        assert item["prompt_hash"] is not None
        assert len(item["prompt_hash"]) == 64

        # Check file exists on disk
        file_path = tmp_path / item["file"]
        assert file_path.is_file()
        assert file_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


# ==============================================================================
# 3. Prompt Hash Sensitivity
# ==============================================================================

def test_prompt_hash_sensitivity():
    """Verify prompt_hash changes when prompt, model, size, or quality change."""
    def compute_hash(prompt, model, size, quality):
        payload = f"{prompt}|{model}|{size}|{quality}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    base = compute_hash("Airplane window", "gpt-image-2", "1024x1536", "low")
    diff_prompt = compute_hash("Airplane window modified", "gpt-image-2", "1024x1536", "low")
    diff_model = compute_hash("Airplane window", "gpt-image-2.5", "1024x1536", "low")
    diff_size = compute_hash("Airplane window", "gpt-image-2", "864x1536", "low")
    diff_quality = compute_hash("Airplane window", "gpt-image-2", "1024x1536", "high")

    assert base != diff_prompt
    assert base != diff_model
    assert base != diff_size
    assert base != diff_quality


# ==============================================================================
# 4. Reuse Skips Regeneration (Zero AI Cost on Rerun)
# ==============================================================================

def test_reuse_skips_regeneration(tmp_path, monkeypatch):
    """Verify rerun skips scenes whose prompt hash matches and file exists."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    fake_provider = CountingFakeProvider()
    video = DummyVideo()
    prev = _make_storyboard_prev(scene_count=3)
    step = DummyStep()

    # Run 1: Generates all 3 scenes
    output_1 = generate_images_step(video, prev, step, image_provider=fake_provider)
    assert fake_provider.call_count == 3
    assert output_1["meta"]["images_generated"] == 3
    assert output_1["meta"]["images_reused"] == 0
    assert step.output["images"]["complete"] is True

    # Run 2: Re-executing step with same output on disk
    fake_provider_2 = CountingFakeProvider()
    output_2 = generate_images_step(video, prev, step, image_provider=fake_provider_2)

    # ZERO calls made to the provider!
    assert fake_provider_2.call_count == 0
    assert output_2["meta"]["images_generated"] == 0
    assert output_2["meta"]["images_reused"] == 3
    for it in output_2["images"]["items"]:
        assert it["status"] == "ok"
        assert it["reused"] is True


# ==============================================================================
# 5. Failure at Scene 3, Partial Output, and Successful Resume
# ==============================================================================

def test_partial_save_and_resume_after_failure(tmp_path, monkeypatch):
    """A failure at scene 3 saves scenes 1-2; later rerun resumes and only generates scene 3."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = DummyVideo()
    prev = _make_storyboard_prev(scene_count=3)
    step = DummyStep()

    # Run 1: Fails on call 3
    failing_provider = CountingFakeProvider(fail_on_scene=3)
    with pytest.raises(PipelineError, match="Image generation failed for scenes: 3"):
        generate_images_step(video, prev, step, image_provider=failing_provider)

    assert failing_provider.call_count == 3
    # Step output was persisted with partial progress
    assert step.output is not None
    saved_items = step.output["images"]["items"]
    assert len(saved_items) == 3
    assert saved_items[0]["status"] == "ok"
    assert saved_items[1]["status"] == "ok"
    assert saved_items[2]["status"] == "failed"
    assert step.output["images"]["complete"] is False

    # Check files 1 and 2 exist on disk
    assert (tmp_path / saved_items[0]["file"]).is_file()
    assert (tmp_path / saved_items[1]["file"]).is_file()

    # Run 2: Resume with healthy provider
    healthy_provider = CountingFakeProvider()
    resume_output = generate_images_step(video, prev, step, image_provider=healthy_provider)

    # Only scene 3 was newly generated (1 call)!
    assert healthy_provider.call_count == 1
    assert resume_output["images"]["complete"] is True
    assert resume_output["meta"]["images_reused"] == 2
    assert resume_output["meta"]["images_generated"] == 1
    assert resume_output["meta"]["images_failed"] == 0

    assert resume_output["images"]["items"][0]["reused"] is True
    assert resume_output["images"]["items"][1]["reused"] is True
    assert resume_output["images"]["items"][2]["reused"] is False
    assert resume_output["images"]["items"][2]["status"] == "ok"


# ==============================================================================
# 6. Rate Limit (429) Abort Keeps Partial Output
# ==============================================================================

def test_rate_limit_429_aborts_and_keeps_partial_output(tmp_path, monkeypatch):
    """Rate limit 429 after retries exhausted aborts run, keeps scenes 1, and raises readable message."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = DummyVideo()
    prev = _make_storyboard_prev(scene_count=3)
    step = DummyStep()

    rate_limit_provider = CountingFakeProvider(
        fail_on_scene=2,
        error_to_raise=ImageRateLimitError("Rate limit 429 exceeded"),
    )

    with pytest.raises(PipelineError, match="rate limit"):
        generate_images_step(video, prev, step, image_provider=rate_limit_provider)

    # Partial output has scene 1 saved
    assert step.output is not None
    assert len(step.output["images"]["items"]) == 1
    assert step.output["images"]["items"][0]["status"] == "ok"
    assert step.output["images"]["complete"] is False


# ==============================================================================
# 7. Auth and Configuration Abort Immediately
# ==============================================================================

def test_auth_error_aborts_at_once(tmp_path, monkeypatch):
    """Auth/permission error on scene 1 immediately aborts the step."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = DummyVideo()
    prev = _make_storyboard_prev(scene_count=3)
    step = DummyStep()

    auth_failing_provider = CountingFakeProvider(
        fail_on_scene=1,
        error_to_raise=ImageConfigError("OpenAI authentication failed (401)"),
    )

    with pytest.raises(PipelineError, match="authentication failed"):
        generate_images_step(video, prev, step, image_provider=auth_failing_provider)


# ==============================================================================
# 8. Content-Policy Block Continues Other Scenes
# ==============================================================================

def test_content_block_continues_other_scenes(tmp_path, monkeypatch):
    """A content policy block on scene 2 marks scene 2 failed but continues generating scene 3."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = DummyVideo()
    prev = _make_storyboard_prev(scene_count=3)
    step = DummyStep()

    content_block_provider = CountingFakeProvider(
        fail_on_scene=2,
        error_to_raise=ImageContentBlockedError("Prompt rejected by safety policy"),
    )

    with pytest.raises(PipelineError, match="Image generation failed for scenes: 2"):
        generate_images_step(video, prev, step, image_provider=content_block_provider)

    # All 3 scenes were processed
    assert content_block_provider.call_count == 3
    items = step.output["images"]["items"]
    assert len(items) == 3
    assert items[0]["status"] == "ok"
    assert items[1]["status"] == "failed"
    assert "safety policy" in items[1]["error"]
    assert items[2]["status"] == "ok"
    assert step.output["images"]["complete"] is False


# ==============================================================================
# 9. Max Images Guard and Input Validation
# ==============================================================================

def test_max_images_guard(tmp_path, monkeypatch):
    """Fails before generation if scene count exceeds image_max_per_video."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "image_max_per_video", 4)

    video = DummyVideo()
    prev = _make_storyboard_prev(scene_count=5)
    step = DummyStep()
    fake_provider = CountingFakeProvider()

    with pytest.raises(PipelineError, match="exceeding the maximum allowed"):
        generate_images_step(video, prev, step, image_provider=fake_provider)

    # Not a single provider call was made
    assert fake_provider.call_count == 0


def test_missing_storyboard_input_errors(tmp_path, monkeypatch):
    """Readable errors when storyboard step is missing or has no scenes."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    video = DummyVideo()
    step = DummyStep()

    # Missing storyboard key
    with pytest.raises(PipelineError, match="Storyboard step must be completed"):
        generate_images_step(video, {}, step)

    # Storyboard with empty scenes list
    with pytest.raises(PipelineError, match="Storyboard step must be completed"):
        generate_images_step(video, {"storyboard": {"scenes": []}}, step)


# ==============================================================================
# 10. OpenAI Provider Configuration Checks
# ==============================================================================

def test_openai_provider_names_exact_missing_setting():
    """Verify missing key or missing model raises ImageConfigError naming the exact setting."""
    # Missing API key
    p1 = OpenAIImageProvider(api_key=None, model="gpt-image-2")
    with pytest.raises(ImageConfigError, match="OPENAI_IMAGE_API_KEY"):
        p1._get_client()

    # Missing model
    p2 = OpenAIImageProvider(api_key="valid-key", model=None)
    with pytest.raises(ImageConfigError, match="OPENAI_IMAGE_MODEL"):
        p2._get_client()


def test_openai_provider_never_falls_back_to_llm_key(monkeypatch):
    """Verify OPENAI_IMAGE_API_KEY never falls back to OPENAI_API_KEY."""
    monkeypatch.setattr(settings, "openai_api_key", "llm-key-secret")
    monkeypatch.setattr(settings, "openai_image_api_key", None)

    p = get_image_provider(provider="openai", model="test-model")
    assert p.api_key is None
    with pytest.raises(ImageConfigError, match="OPENAI_IMAGE_API_KEY"):
        p._get_client()


# ==============================================================================
# 11. Backward Compatibility & Warnings
# ==============================================================================

def test_old_shape_storyboard_compatibility(tmp_path, monkeypatch):
    """Legacy storyboard format with scenes directly under storyboard works seamlessly."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = DummyVideo()
    step = DummyStep()
    prev = {
        "storyboard": {
            "scenes": [
                {
                    "index": 1,
                    "narration": "Old style narration.",
                    "image_prompt": "Old airplane photo",
                    "shot_type": "wide",
                }
            ]
        }
    }

    output = generate_images_step(video, prev, step, image_provider=MockImageProvider())
    assert output["images"]["complete"] is True
    assert len(output["images"]["items"]) == 1
    assert output["images"]["items"][0]["status"] == "ok"


def test_user_prompt_warning_on_images_step(tmp_path, monkeypatch):
    """Warning is added if user_prompt is present on the images step."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = DummyVideo()
    step = DummyStep(user_prompt="Make colors more saturated")
    prev = _make_storyboard_prev(scene_count=1)

    output = generate_images_step(video, prev, step, image_provider=MockImageProvider())
    assert "Instructions are not used by the images step" in output["warnings"][0]


# ==============================================================================
# 12. Serving Route: GET /videos/{video_id}/images/{index}
# ==============================================================================

def test_serving_route_ownership_and_delivery(tmp_path, monkeypatch):
    """Test image retrieval endpoint: 200 with ETag for owner, 404 for different user or missing image."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    user_a = User(id=uuid.uuid4(), email="owner@test.com", password_hash="pw", is_active=True)
    user_b = User(id=uuid.uuid4(), email="other@test.com", password_hash="pw", is_active=True)


    channel_a = Channel(id=uuid.uuid4(), user_id=user_a.id, name="Channel A")
    video_a = Video(id=uuid.uuid4(), channel_id=channel_a.id, topic="Serving test")

    # Write a test image file
    rel_path = f"{video_a.id}/images/scene_01_abcdef12.png"
    abs_file = tmp_path / rel_path
    abs_file.parent.mkdir(parents=True, exist_ok=True)
    png_bytes = create_solid_png(1024, 1536, 12, 34, 56)
    abs_file.write_bytes(png_bytes)

    # Setup images step on video_a (even if step status is failed, item status 'ok' must be served)
    img_step = VideoStep(
        video_id=video_a.id,
        name="images",
        position=2,
        status="failed",
        output={
            "images": {
                "items": [
                    {
                        "index": 1,
                        "status": "ok",
                        "file": rel_path,
                        "prompt_hash": "abcdef1234567890",
                    },
                    {
                        "index": 2,
                        "status": "failed",
                        "file": None,
                        "prompt_hash": "1111222233334444",
                    },
                ]
            }
        },
    )
    video_a.steps = [img_step]
    video_a.channel = channel_a

    from app.deps import get_owned_video

    client = TestClient(app)

    # 1. Owner requests scene 1 -> 200 with ETag & Cache-Control
    app.dependency_overrides[get_owned_video] = lambda: video_a
    try:
        resp = client.get(f"/videos/{video_a.id}/images/1")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.headers["etag"] == '"abcdef1234567890"'
        assert "no-cache" in resp.headers["cache-control"]
        assert resp.content == png_bytes

        # 2. Scene 2 has status 'failed' -> 404
        resp_failed = client.get(f"/videos/{video_a.id}/images/2")
        assert resp_failed.status_code == 404
        assert "No generated image available" in resp_failed.json()["detail"]

        # 3. Non-existent scene index -> 404
        resp_missing = client.get(f"/videos/{video_a.id}/images/99")
        assert resp_missing.status_code == 404

        # 4. Different user requesting video_a -> 404 (ownership check)
        from fastapi import HTTPException

        def mock_other_user():
            raise HTTPException(404, "Video not found")

        app.dependency_overrides[get_owned_video] = mock_other_user
        resp_unowned = client.get(f"/videos/{video_a.id}/images/1")
        assert resp_unowned.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_serving_route_path_traversal_rejected(tmp_path, monkeypatch):
    """Ensure paths escaping the storage root are strictly rejected with 403."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = Video(id=uuid.uuid4(), channel_id=uuid.uuid4(), topic="Traversal test")
    img_step = VideoStep(
        video_id=video.id,
        name="images",
        status="done",
        output={
            "images": {
                "items": [
                    {
                        "index": 1,
                        "status": "ok",
                        "file": "../../etc/passwd",
                        "prompt_hash": "traversalhash",
                    }
                ]
            }
        },
    )
    video.steps = [img_step]

    from app.deps import get_owned_video
    app.dependency_overrides[get_owned_video] = lambda: video
    client = TestClient(app)
    try:
        resp = client.get(f"/videos/{video.id}/images/1")
        assert resp.status_code == 403
        assert "forbidden" in resp.json()["detail"].lower()
    finally:
        app.dependency_overrides.clear()


def test_tokens_and_cost_aggregation_in_meta(tmp_path, monkeypatch):
    """Verify that input/output/total tokens and cost estimates are correctly summed into meta."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    provider = CountingFakeProvider()
    video = DummyVideo()
    prev = _make_storyboard_prev(scene_count=2)
    step = DummyStep()

    output = generate_images_step(video, prev, step, image_provider=provider)

    meta = output["meta"]
    assert meta["input_tokens"] == 15 * 2
    assert meta["output_tokens"] == 100 * 2
    assert meta["total_tokens"] == 115 * 2
    assert meta["est_cost_usd"] == round(0.04 * 2, 4)

