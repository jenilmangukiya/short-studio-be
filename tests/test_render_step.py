"""Unit and integration tests for the video rendering step (app/steps/render.py).

Tests:
1. Precondition validation (missing images, missing voiceover, mismatched scenes).
2. File existence validation (missing asset on disk).
3. Subprocess failure and timeout handling with clean temp file cleanup.
4. Fast end-to-end rendering of a miniature synthetic clip using local FFmpeg in tmp_path.
"""

from pathlib import Path
import subprocess
from unittest.mock import MagicMock, patch
import uuid
import pytest

from app.config import settings
from app.images.mock import create_solid_png
from app.models import Video, VideoStep
from app.pipeline import h_render
from app.steps.render import PipelineError, generate_render_step
from app.voiceover.mock import create_mock_wav


class DummyVideo:
    def __init__(self, video_id=None):
        self.id = video_id or uuid.uuid4()
        self.topic = "Test Jeans Rivets"
        self.output_path = None


def test_missing_images_precondition():
    """Verify error when images step is missing from prev."""
    video = DummyVideo()
    prev = {
        "voiceover": {
            "voiceover": {
                "items": [{"index": 1, "status": "ok", "file": "dummy/audio.wav"}]
            }
        }
    }
    with pytest.raises(PipelineError) as exc_info:
        generate_render_step(video, prev)
    assert "Images step must be completed" in str(exc_info.value)


def test_missing_voiceover_precondition():
    """Verify error when voiceover step is missing from prev."""
    video = DummyVideo()
    prev = {
        "images": {
            "images": {
                "items": [{"index": 1, "status": "ok", "file": "dummy/img.png"}]
            }
        }
    }
    with pytest.raises(PipelineError) as exc_info:
        generate_render_step(video, prev)
    assert "Voiceover step must be completed" in str(exc_info.value)


def test_scene_alignment_mismatch():
    """Verify error when scenes between images and audio are mismatched."""
    video = DummyVideo()
    prev = {
        "images": {
            "images": {
                "items": [
                    {"index": 1, "status": "ok", "file": "dummy/scene1.png"},
                    {"index": 2, "status": "ok", "file": "dummy/scene2.png"},
                ]
            }
        },
        "voiceover": {
            "voiceover": {
                "items": [
                    {"index": 1, "status": "ok", "file": "dummy/scene1.wav"},
                    # Scene 2 audio missing!
                ]
            }
        },
    }
    with pytest.raises(PipelineError) as exc_info:
        generate_render_step(video, prev)
    assert "scenes missing audio: [2]" in str(exc_info.value)


def test_asset_file_missing_on_disk(tmp_path, monkeypatch):
    """Verify error when an asset in metadata does not physically exist on disk."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    video = DummyVideo()
    prev = {
        "images": {
            "images": {
                "items": [{"index": 1, "status": "ok", "file": "nonexistent/img.png"}]
            }
        },
        "voiceover": {
            "voiceover": {
                "items": [{"index": 1, "status": "ok", "file": "nonexistent/audio.wav", "duration_seconds": 2.0}]
            }
        },
    }
    with pytest.raises(PipelineError) as exc_info:
        generate_render_step(video, prev)
    assert "does not exist" in str(exc_info.value)


def test_ffmpeg_subprocess_error_cleanup(tmp_path, monkeypatch):
    """Verify that FFmpeg failure raises PipelineError and cleans up temp directories."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    video = DummyVideo()

    # Create dummy files
    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)
    img_path = video_dir / "scene_01.png"
    img_path.write_bytes(create_solid_png(100, 150, 200, 100, 50))
    vo_path = video_dir / "scene_01.wav"
    vo_path.write_bytes(create_mock_wav(1.0, 24000))

    prev = {
        "images": {
            "images": {
                "items": [{"index": 1, "status": "ok", "file": f"{video.id}/scene_01.png"}]
            }
        },
        "voiceover": {
            "voiceover": {
                "items": [{"index": 1, "status": "ok", "file": f"{video.id}/scene_01.wav", "duration_seconds": 1.0}]
            }
        },
    }

    # Simulate FFmpeg failure
    def mock_run(*args, **kwargs):
        raise subprocess.CalledProcessError(1, "ffmpeg", stderr="Simulated FFmpeg encoding failure")

    with patch("subprocess.run", side_effect=mock_run):
        with pytest.raises(PipelineError) as exc_info:
            generate_render_step(video, prev)

    assert "Simulated FFmpeg encoding failure" in str(exc_info.value)

    # Verify all temporary directories were cleaned up
    tmp_dirs = list(video_dir.glob("tmp_render_*"))
    assert len(tmp_dirs) == 0, f"Expected 0 temp directories, found {tmp_dirs}"


def test_fast_real_render_synthetic_clip(tmp_path, monkeypatch):
    """Fast end-to-end integration test rendering a 0.5s synthetic clip with local FFmpeg."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "render_fps", 30)

    video = DummyVideo()
    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic 1024x1536 solid PNG
    img_bytes = create_solid_png(1024, 1536, 40, 80, 160)
    img_rel = f"{video.id}/scene_01.png"
    (tmp_path / img_rel).write_bytes(img_bytes)

    # Create synthetic 0.5s 24kHz WAV
    wav_bytes = create_mock_wav(0.5, 24000)
    wav_rel = f"{video.id}/scene_01.wav"
    (tmp_path / wav_rel).write_bytes(wav_bytes)

    prev = {
        "images": {
            "images": {
                "items": [{"index": 1, "status": "ok", "file": img_rel}]
            }
        },
        "voiceover": {
            "voiceover": {
                "items": [{"index": 1, "status": "ok", "file": wav_rel, "duration_seconds": 0.5}]
            }
        },
    }

    res = h_render(video, prev, None)

    # Verify returned metadata
    assert "render" in res
    assert "video_path" in res
    assert res["render"]["resolution"] == "1080x1920"
    assert res["render"]["codec"] == "h264"
    assert res["render"]["audio_codec"] == "aac"
    assert res["render"]["fps"] == 30
    assert res["render"]["scenes_rendered"] == 1
    assert abs(res["duration_seconds"] - 0.5) < 0.1

    # Verify final video file exists on disk and is non-empty
    final_file = Path(res["video_path"])
    assert final_file.is_file()
    assert final_file.stat().st_size > 1000
    assert video.output_path == str(final_file)

    # Verify zero temporary directories remain
    tmp_dirs = list(video_dir.glob("tmp_render_*"))
    assert len(tmp_dirs) == 0


def test_render_ken_burns_motion_validation(tmp_path, monkeypatch):
    """Verify Ken Burns camera motion is applied with frame-to-frame pixel differences."""
    from PIL import Image, ImageDraw
    import numpy as np

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "render_fps", 30)

    video = DummyVideo()
    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)

    # Create synthetic pattern PNG with distinct details to detect zoom motion
    img1 = Image.new("RGB", (1024, 1536), color=(20, 30, 40))
    draw1 = ImageDraw.Draw(img1)
    draw1.rectangle([300, 400, 700, 800], fill=(220, 180, 50))
    img1_rel = f"{video.id}/scene_01.png"
    img1.save(tmp_path / img1_rel)

    img2 = Image.new("RGB", (1024, 1536), color=(40, 20, 50))
    draw2 = ImageDraw.Draw(img2)
    draw2.ellipse([350, 450, 650, 750], fill=(50, 200, 220))
    img2_rel = f"{video.id}/scene_02.png"
    img2.save(tmp_path / img2_rel)

    wav1_rel = f"{video.id}/scene_01.wav"
    (tmp_path / wav1_rel).write_bytes(create_mock_wav(0.6, 24000))
    wav2_rel = f"{video.id}/scene_02.wav"
    (tmp_path / wav2_rel).write_bytes(create_mock_wav(0.6, 24000))

    prev = {
        "storyboard": {
            "storyboard": {
                "scenes": [
                    {"index": 1, "shot_type": "wide"},
                    {"index": 2, "shot_type": "close-up"},
                ]
            }
        },
        "images": {
            "images": {
                "items": [
                    {"index": 1, "status": "ok", "file": img1_rel},
                    {"index": 2, "status": "ok", "file": img2_rel},
                ]
            }
        },
        "voiceover": {
            "voiceover": {
                "items": [
                    {"index": 1, "status": "ok", "file": wav1_rel, "duration_seconds": 0.6},
                    {"index": 2, "status": "ok", "file": wav2_rel, "duration_seconds": 0.6},
                ]
            }
        },
    }

    res = h_render(video, prev, None)
    final_video = Path(res["video_path"])
    assert final_video.is_file()
    assert res["render"]["scenes_rendered"] == 2
    assert abs(res["duration_seconds"] - 1.2) < 0.15

    # Extract two frames from scene 1 at different timestamps (t=0.1s and t=0.45s)
    f1_path = tmp_path / "f1.png"
    f2_path = tmp_path / "f2.png"
    subprocess.run(
        ["ffmpeg", "-y", "-ss", "0.1", "-i", str(final_video), "-frames:v", "1", str(f1_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-ss", "0.45", "-i", str(final_video), "-frames:v", "1", str(f2_path)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    im1 = np.array(Image.open(f1_path))
    im2 = np.array(Image.open(f2_path))

    assert im1.shape == (1920, 1080, 3)
    assert im2.shape == (1920, 1080, 3)

    # In a static video, im1 and im2 would have difference == 0.
    # With Ken Burns motion, the frame content has changed due to zoom:
    pixel_diff = float(np.mean(np.abs(im1.astype(float) - im2.astype(float))))
    assert pixel_diff > 0.2, f"Expected Ken Burns motion between frames, got diff {pixel_diff}"
