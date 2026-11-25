"""Unit and integration tests for Phase 4C: Visual Style System Renderer & Motion Integration.

Verifies:
1. Legacy scene zoom: 1.050.
2. Legacy visual beat zoom: 1.050 for <= 4 seconds and 1.070 for > 4 seconds.
3. Stickman scene and beat scales: 1.030 (<= 4s) and 1.042 (> 4s).
4. Stylized 2D scene and beat scales: 1.040 (<= 4s) and 1.056 (> 4s).
5. Zoom-out scales are the inverse of the corresponding zoom-in scales.
6. Static scenes remain static across all styles (1.000 -> 1.000).
7. Frozen snapshot takes precedence over registry mutation.
8. Legacy NULL-style videos preserve historical motion.
9. Missing optional render profile and motion scale use schema defaults.
10. Invalid motion scale values fail closed.
11. Malformed snapshots fail closed.
12. Unknown explicit style IDs and incompatible versions fail closed.
13. Renderer compiles the exact supplied plan scales without double scaling.
14. Fallback Ken Burns retains its constant 0.05 delta.
15. Render telemetry reports actual calibrated animation scales, motion scale, and visual style ID.
16. Boundary tests at exactly 4.0s and just above 4.0s (4.01s).
"""

from pathlib import Path
import subprocess
from unittest.mock import MagicMock
import uuid
import pytest

from app.animation.crop import calculate_portrait_crop, plan_beat_camera_motion
from app.animation.planner import plan_scene_animation
from app.animation.renderer import build_fallback_ken_burns_filter, build_ffmpeg_zoompan_filter
from app.animation.types import AnimationPlan
from app.config import settings
from app.images.mock import create_solid_png
from app.models import Video
from app.steps.render import PipelineError, generate_render_step
from app.styles import (
    LEGACY_DEFAULT_PRESET,
    STICKMAN_TACTICAL_EXPLAINER_PRESET,
    STYLIZED_2D_NARRATIVE_PRESET,
    STYLES_REGISTRY,
    VisualStyleResolutionError,
    canonical_snapshot_hash,
)
from app.voiceover.mock import create_mock_wav


class DummyVideo:
    """Mock video model instance for unit and render testing."""

    def __init__(
        self,
        video_id=None,
        visual_style_id=None,
        visual_style_version=None,
        visual_style_snapshot=None,
    ):
        self.id = video_id or uuid.uuid4()
        self.topic = "Phase 4C Motion Test"
        self.output_path = None
        self.visual_style_id = visual_style_id
        self.visual_style_version = visual_style_version
        self.visual_style_snapshot = visual_style_snapshot


# ==============================================================================
# 1. Planner Unit Tests: Motion Scaling Formula & Scale Matrix
# ==============================================================================


def test_planner_scene_zoom_in_scale_matrix():
    """Verify scene zoom-in scales match the expected scale matrix across all styles."""
    # Legacy default (motion_scale = 1.0) -> end_scale = 1.050
    plan_legacy = plan_scene_animation(
        scene_data={"shot_type": "medium"},
        duration_seconds=3.0,
        motion_scale=1.0,
    )
    assert plan_legacy.animation_type == "zoom-in"
    assert plan_legacy.start_scale == 1.000
    assert plan_legacy.end_scale == 1.050

    # Stickman tactical explainer (motion_scale = 0.6) -> end_scale = 1.030
    plan_stickman = plan_scene_animation(
        scene_data={"shot_type": "medium"},
        duration_seconds=3.0,
        motion_scale=0.6,
    )
    assert plan_stickman.animation_type == "zoom-in"
    assert plan_stickman.start_scale == 1.000
    assert plan_stickman.end_scale == 1.030

    # Stylized 2D narrative drama (motion_scale = 0.8) -> end_scale = 1.040
    plan_stylized = plan_scene_animation(
        scene_data={"shot_type": "medium"},
        duration_seconds=3.0,
        motion_scale=0.8,
    )
    assert plan_stylized.animation_type == "zoom-in"
    assert plan_stylized.start_scale == 1.000
    assert plan_stylized.end_scale == 1.040


def test_planner_scene_zoom_out_inverse_scaling():
    """Verify zoom-out scales are the exact inverse of zoom-in scales."""
    # Close-up triggers zoom-out (pull-out)
    plan_legacy = plan_scene_animation(
        scene_data={"shot_type": "close-up"},
        duration_seconds=3.0,
        motion_scale=1.0,
    )
    assert plan_legacy.animation_type == "zoom-out"
    assert plan_legacy.start_scale == 1.050
    assert plan_legacy.end_scale == 1.000

    plan_stickman = plan_scene_animation(
        scene_data={"shot_type": "close-up"},
        duration_seconds=3.0,
        motion_scale=0.6,
    )
    assert plan_stickman.animation_type == "zoom-out"
    assert plan_stickman.start_scale == 1.030
    assert plan_stickman.end_scale == 1.000

    plan_stylized = plan_scene_animation(
        scene_data={"shot_type": "close-up"},
        duration_seconds=3.0,
        motion_scale=0.8,
    )
    assert plan_stylized.animation_type == "zoom-out"
    assert plan_stylized.start_scale == 1.040
    assert plan_stylized.end_scale == 1.000


def test_planner_static_scenes_remain_static_across_all_styles():
    """Verify diagram and schematic scenes remain static (1.000 -> 1.000) for every motion scale."""
    for ms in [0.0, 0.6, 0.8, 1.0, 1.5, 2.0]:
        plan = plan_scene_animation(
            scene_data={"shot_type": "diagram", "visual_purpose": "schematic cutaway"},
            duration_seconds=3.0,
            motion_scale=ms,
        )
        assert plan.animation_type == "static"
        assert plan.start_scale == 1.000
        assert plan.end_scale == 1.000


# ==============================================================================
# 2. Visual Beat Duration Scaling & Boundary Tests
# ==============================================================================


def test_beat_camera_motion_duration_categories_and_boundary():
    """Verify visual beat scaling for duration <= 4s vs > 4s, including exact 4.0s boundary."""
    crop_reg = calculate_portrait_crop(source_width=1080, source_height=1920, scale_factor=1.0)

    # 1. Exact 4.0s boundary (uses <= 4s formula: 0.05 * motion_scale)
    b_leg_40 = plan_beat_camera_motion(
        beat={"duration_seconds": 4.0, "shot_type": "medium"},
        crop_region=crop_reg,
        source_width=1080,
        source_height=1920,
        duration_seconds=4.0,
        motion_scale=1.0,
    )
    assert b_leg_40.animation_type == "zoom-in"
    assert b_leg_40.end_crop.scale_factor == 1.050

    b_stick_40 = plan_beat_camera_motion(
        beat={"duration_seconds": 4.0, "shot_type": "medium"},
        crop_region=crop_reg,
        source_width=1080,
        source_height=1920,
        duration_seconds=4.0,
        motion_scale=0.6,
    )
    assert b_stick_40.end_crop.scale_factor == 1.030

    b_styl_40 = plan_beat_camera_motion(
        beat={"duration_seconds": 4.0, "shot_type": "medium"},
        crop_region=crop_reg,
        source_width=1080,
        source_height=1920,
        duration_seconds=4.0,
        motion_scale=0.8,
    )
    assert b_styl_40.end_crop.scale_factor == 1.040

    # 2. Just above boundary: 4.01s (uses > 4s formula: 0.07 * motion_scale)
    b_leg_401 = plan_beat_camera_motion(
        beat={"duration_seconds": 4.01, "shot_type": "medium"},
        crop_region=crop_reg,
        source_width=1080,
        source_height=1920,
        duration_seconds=4.01,
        motion_scale=1.0,
    )
    assert b_leg_401.end_crop.scale_factor == 1.070

    b_stick_401 = plan_beat_camera_motion(
        beat={"duration_seconds": 4.01, "shot_type": "medium"},
        crop_region=crop_reg,
        source_width=1080,
        source_height=1920,
        duration_seconds=4.01,
        motion_scale=0.6,
    )
    assert b_stick_401.end_crop.scale_factor == 1.042

    b_styl_401 = plan_beat_camera_motion(
        beat={"duration_seconds": 4.01, "shot_type": "medium"},
        crop_region=crop_reg,
        source_width=1080,
        source_height=1920,
        duration_seconds=4.01,
        motion_scale=0.8,
    )
    assert b_styl_401.end_crop.scale_factor == 1.056

    # 3. Short beat (< 2.0s) defaults to static hold
    b_short = plan_beat_camera_motion(
        beat={"duration_seconds": 1.5, "shot_type": "medium"},
        crop_region=crop_reg,
        source_width=1080,
        source_height=1920,
        duration_seconds=1.5,
        motion_scale=0.6,
    )
    assert b_short.animation_type == "static"
    assert b_short.start_crop.scale_factor == 1.000
    assert b_short.end_crop.scale_factor == 1.000


def test_beat_camera_motion_zoom_out_duration_categories():
    """Verify visual beat zoom-out (pull-out) inverse scaling for duration <= 4s and > 4s across all styles."""
    crop_reg = calculate_portrait_crop(source_width=1080, source_height=1920, scale_factor=1.0)

    # Category 1: Duration <= 4.0s (delta = 0.05 * motion_scale; start_scale = 1.0 + delta, end_scale = 1.000)
    # Legacy default (motion_scale = 1.0) -> 1.050 -> 1.000
    b_leg_le4 = plan_beat_camera_motion(
        beat={"duration_seconds": 3.0, "shot_type": "close-up"},
        crop_region=crop_reg,
        source_width=1080,
        source_height=1920,
        duration_seconds=3.0,
        motion_scale=1.0,
    )
    assert b_leg_le4.animation_type == "zoom-out"
    assert b_leg_le4.start_crop.scale_factor == 1.050
    assert b_leg_le4.end_crop.scale_factor == 1.000

    # Stickman (motion_scale = 0.6) -> 1.030 -> 1.000
    b_stick_le4 = plan_beat_camera_motion(
        beat={"duration_seconds": 3.0, "shot_type": "close-up"},
        crop_region=crop_reg,
        source_width=1080,
        source_height=1920,
        duration_seconds=3.0,
        motion_scale=0.6,
    )
    assert b_stick_le4.animation_type == "zoom-out"
    assert b_stick_le4.start_crop.scale_factor == 1.030
    assert b_stick_le4.end_crop.scale_factor == 1.000

    # Stylized 2D (motion_scale = 0.8) -> 1.040 -> 1.000
    b_styl_le4 = plan_beat_camera_motion(
        beat={"duration_seconds": 3.0, "shot_type": "close-up"},
        crop_region=crop_reg,
        source_width=1080,
        source_height=1920,
        duration_seconds=3.0,
        motion_scale=0.8,
    )
    assert b_styl_le4.animation_type == "zoom-out"
    assert b_styl_le4.start_crop.scale_factor == 1.040
    assert b_styl_le4.end_crop.scale_factor == 1.000

    # Category 2: Duration > 4.0s (delta = 0.07 * motion_scale; start_scale = 1.0 + delta, end_scale = 1.000)
    # Legacy default (motion_scale = 1.0) -> 1.070 -> 1.000
    b_leg_gt4 = plan_beat_camera_motion(
        beat={"duration_seconds": 4.5, "shot_type": "close-up"},
        crop_region=crop_reg,
        source_width=1080,
        source_height=1920,
        duration_seconds=4.5,
        motion_scale=1.0,
    )
    assert b_leg_gt4.animation_type == "zoom-out"
    assert b_leg_gt4.start_crop.scale_factor == 1.070
    assert b_leg_gt4.end_crop.scale_factor == 1.000

    # Stickman (motion_scale = 0.6) -> 1.042 -> 1.000
    b_stick_gt4 = plan_beat_camera_motion(
        beat={"duration_seconds": 4.5, "shot_type": "close-up"},
        crop_region=crop_reg,
        source_width=1080,
        source_height=1920,
        duration_seconds=4.5,
        motion_scale=0.6,
    )
    assert b_stick_gt4.animation_type == "zoom-out"
    assert b_stick_gt4.start_crop.scale_factor == 1.042
    assert b_stick_gt4.end_crop.scale_factor == 1.000

    # Stylized 2D (motion_scale = 0.8) -> 1.056 -> 1.000
    b_styl_gt4 = plan_beat_camera_motion(
        beat={"duration_seconds": 4.5, "shot_type": "close-up"},
        crop_region=crop_reg,
        source_width=1080,
        source_height=1920,
        duration_seconds=4.5,
        motion_scale=0.8,
    )
    assert b_styl_gt4.animation_type == "zoom-out"
    assert b_styl_gt4.start_crop.scale_factor == 1.056
    assert b_styl_gt4.end_crop.scale_factor == 1.000


# ==============================================================================
# 3. Renderer Compilation & Fallback Ken Burns (No Double Scaling)
# ==============================================================================


def test_renderer_compiles_exact_plan_without_double_scaling():
    """Verify build_ffmpeg_zoompan_filter compiles the exact scales from AnimationPlan."""
    # Stickman plan with end_scale = 1.030
    plan = AnimationPlan(
        animation_type="zoom-in",
        focal_x=0.5,
        focal_y=0.42,
        start_scale=1.000,
        end_scale=1.030,
        duration_seconds=3.0,
    )
    z_expr, x_expr, y_expr, fc, resolved = build_ffmpeg_zoompan_filter(
        plan=plan,
        scene_dur=3.0,
        target_fps=30,
        target_width=1080,
        target_height=1920,
    )
    assert resolved["start_scale"] == 1.000
    assert resolved["end_scale"] == 1.030
    # The zoompan expression must contain the exact +0.030 delta (not doubled or re-multiplied)
    assert "+0.030" in z_expr


def test_fallback_ken_burns_retains_constant_delta():
    """Verify emergency fallback Ken Burns filter retains constant 0.050 delta."""
    z_expr, x_expr, y_expr, fc = build_fallback_ken_burns_filter(
        shot_type="medium",
        scene_idx=1,
        scene_dur=3.0,
        target_fps=30,
        target_width=1080,
        target_height=1920,
    )
    assert "1.0+0.050" in z_expr


# ==============================================================================
# 4. End-to-End Render Integration Tests with Mock Assets
# ==============================================================================


def _setup_mock_render_pipeline(tmp_path: Path, video: DummyVideo):
    """Helper to write mock image and voiceover assets to disk."""
    video_dir = tmp_path / str(video.id)
    images_dir = video_dir / "images"
    audio_dir = video_dir / "audio"
    images_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    img1 = images_dir / "scene_01.png"
    img2 = images_dir / "scene_02.png"
    img1.write_bytes(create_solid_png(1080, 1920, 100, 150, 200))
    img2.write_bytes(create_solid_png(1080, 1920, 200, 150, 100))

    audio_file = audio_dir / "narration_full.wav"
    audio_file.write_bytes(create_mock_wav(6.0, 24000))

    prev = {
        "storyboard": {
            "storyboard": {
                "scenes": [
                    {
                        "index": 1,
                        "shot_type": "medium",
                        "visual_description": "Astronomer focusing eyepiece",
                        "duration_seconds": 3.0,
                    },
                    {
                        "index": 2,
                        "shot_type": "close-up",
                        "visual_description": "Close up of eyepiece lens",
                        "duration_seconds": 3.0,
                    },
                ]
            }
        },
        "images": {
            "images": {
                "items": [
                    {"index": 1, "status": "ok", "file": str(img1.relative_to(tmp_path))},
                    {"index": 2, "status": "ok", "file": str(img2.relative_to(tmp_path))},
                ]
            }
        },
        "voiceover": {
            "audio_file": str(audio_file.relative_to(tmp_path)),
            "voiceover": {
                "items": [
                    {"index": 1, "status": "ok", "duration_seconds": 3.0},
                    {"index": 2, "status": "ok", "duration_seconds": 3.0},
                ]
            },
        },
    }
    return prev


def _mock_ffmpeg_execution(monkeypatch):
    """Mock subprocess.run for fast ffmpeg execution in unit tests."""
    def mock_run(cmd, *args, **kwargs):
        cmd_strs = [str(c) for c in cmd]
        # Touch output file
        out_file = Path(cmd_strs[-1])
        out_file.parent.mkdir(parents=True, exist_ok=True)
        if not out_file.exists():
            out_file.write_bytes(b"dummy_mp4_bytes")
        return MagicMock(returncode=0)

    monkeypatch.setattr(subprocess, "run", mock_run)
    monkeypatch.setattr(
        "app.steps.render._probe_video_file",
        MagicMock(
            return_value={
                "streams": [
                    {"codec_type": "video", "codec_name": "h264", "pix_fmt": "yuv420p", "width": 1080, "height": 1920, "duration": 6.0},
                    {"codec_type": "audio", "codec_name": "aac", "duration": 6.0},
                ],
                "format": {"duration": "6.0", "size": "10000"},
            }
        ),
    )


def test_render_stickman_motion_scale_integration(tmp_path: Path, monkeypatch):
    """Verify Stickman preset calibrates render animation scales to 1.030 and records telemetry."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    _mock_ffmpeg_execution(monkeypatch)

    video = DummyVideo(
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    prev = _setup_mock_render_pipeline(tmp_path, video)

    res = generate_render_step(video=video, prev=prev)

    assert "animations" in res["render"]
    anims = res["render"]["animations"]
    assert len(anims) == 2

    # Scene 1 (medium shot -> zoom-in): 1.000 -> 1.030
    assert anims[0]["animation_type"] == "zoom-in"
    assert anims[0]["start_scale"] == 1.000
    assert anims[0]["end_scale"] == 1.030

    # Scene 2 (close-up -> zoom-out): 1.030 -> 1.000
    assert anims[1]["animation_type"] == "zoom-out"
    assert anims[1]["start_scale"] == 1.030
    assert anims[1]["end_scale"] == 1.000

    # Telemetry checks
    assert res["meta"]["visual_style_id"] == "stickman_tactical_explainer"
    assert res["meta"]["motion_scale"] == 0.6


def test_render_stylized_2d_motion_scale_integration(tmp_path: Path, monkeypatch):
    """Verify Stylized 2D preset calibrates render animation scales to 1.040 and records telemetry."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    _mock_ffmpeg_execution(monkeypatch)

    video = DummyVideo(
        visual_style_id="stylized_2d_narrative",
        visual_style_version=1,
        visual_style_snapshot=STYLIZED_2D_NARRATIVE_PRESET.model_dump(),
    )
    prev = _setup_mock_render_pipeline(tmp_path, video)

    res = generate_render_step(video=video, prev=prev)

    anims = res["render"]["animations"]
    assert anims[0]["end_scale"] == 1.040
    assert anims[1]["start_scale"] == 1.040
    assert res["meta"]["visual_style_id"] == "stylized_2d_narrative"
    assert res["meta"]["motion_scale"] == 0.8


def test_render_legacy_null_style_fields_integration(tmp_path: Path, monkeypatch):
    """Verify videos with NULL style fields preserve exact legacy default scales (1.050)."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    _mock_ffmpeg_execution(monkeypatch)

    video = DummyVideo(
        visual_style_id=None,
        visual_style_version=None,
        visual_style_snapshot=None,
    )
    prev = _setup_mock_render_pipeline(tmp_path, video)

    res = generate_render_step(video=video, prev=prev)

    anims = res["render"]["animations"]
    assert anims[0]["end_scale"] == 1.050
    assert anims[1]["start_scale"] == 1.050
    assert res["meta"]["visual_style_id"] is None
    assert res["meta"]["motion_scale"] == 1.0


def test_frozen_snapshot_precedence_over_registry_mutation(tmp_path: Path, monkeypatch):
    """Verify frozen snapshot's motion_scale takes precedence even if registry is mutated."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    _mock_ffmpeg_execution(monkeypatch)

    # Frozen snapshot with custom motion_scale: 0.5 (zoom-in should be 1.025)
    snap = STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump()
    snap["render_profile"]["motion_scale"] = 0.5

    # Mutate registry in-memory to 0.9
    orig_registry_preset = STYLES_REGISTRY["stickman_tactical_explainer"]
    mutated_preset = orig_registry_preset.model_copy(deep=True)
    mutated_preset.render_profile.motion_scale = 0.9
    monkeypatch.setitem(STYLES_REGISTRY, "stickman_tactical_explainer", mutated_preset)

    video = DummyVideo(
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=snap,
    )
    prev = _setup_mock_render_pipeline(tmp_path, video)

    res = generate_render_step(video=video, prev=prev)

    anims = res["render"]["animations"]
    # 1.0 + (0.05 * 0.5) = 1.025, NOT 1.0 + (0.05 * 0.9) = 1.045
    assert anims[0]["end_scale"] == 1.025
    assert res["meta"]["motion_scale"] == 0.5


def test_missing_optional_render_profile_uses_defaults(tmp_path: Path, monkeypatch):
    """Verify valid snapshot with omitted render_profile OR render_profile: {} uses schema default motion_scale (1.0)."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    _mock_ffmpeg_execution(monkeypatch)

    # Case A: render_profile key completely omitted
    snap_a = STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump()
    snap_a.pop("render_profile", None)

    video_a = DummyVideo(
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=snap_a,
    )
    prev_a = _setup_mock_render_pipeline(tmp_path, video_a)

    res_a = generate_render_step(video=video_a, prev=prev_a)
    assert res_a["render"]["animations"][0]["end_scale"] == 1.050
    assert res_a["meta"]["motion_scale"] == 1.0

    # Case B: render_profile is an empty dict {} (motion_scale omitted inside profile)
    snap_b = STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump()
    snap_b["render_profile"] = {}

    video_b = DummyVideo(
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=snap_b,
    )
    prev_b = _setup_mock_render_pipeline(tmp_path, video_b)

    res_b = generate_render_step(video=video_b, prev=prev_b)
    assert res_b["render"]["animations"][0]["end_scale"] == 1.050
    assert res_b["meta"]["motion_scale"] == 1.0


def test_invalid_motion_scale_fails_closed(tmp_path: Path, monkeypatch):
    """Verify invalid motion scale values (negative, zero, > 2.0, non-numeric) fail closed."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    _mock_ffmpeg_execution(monkeypatch)

    # 1. Negative motion scale
    snap_neg = STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump()
    snap_neg["render_profile"]["motion_scale"] = -0.5
    v_neg = DummyVideo(visual_style_id="stickman_tactical_explainer", visual_style_snapshot=snap_neg)
    prev = _setup_mock_render_pipeline(tmp_path, v_neg)
    with pytest.raises(PipelineError) as exc_neg:
        generate_render_step(video=v_neg, prev=prev)
    assert "Unsupported motion_scale" in str(exc_neg.value)

    # 2. Zero motion scale
    snap_zero = STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump()
    snap_zero["render_profile"]["motion_scale"] = 0.0
    v_zero = DummyVideo(visual_style_id="stickman_tactical_explainer", visual_style_snapshot=snap_zero)
    with pytest.raises(PipelineError) as exc_zero:
        generate_render_step(video=v_zero, prev=prev)
    assert "Unsupported motion_scale" in str(exc_zero.value)

    # 3. Excessive motion scale (> 2.0)
    snap_high = STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump()
    snap_high["render_profile"]["motion_scale"] = 3.5
    v_high = DummyVideo(visual_style_id="stickman_tactical_explainer", visual_style_snapshot=snap_high)
    with pytest.raises(PipelineError) as exc_high:
        generate_render_step(video=v_high, prev=prev)
    assert "Unsupported motion_scale" in str(exc_high.value)

    # 4. Non-numeric motion scale
    snap_str = STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump()
    snap_str["render_profile"]["motion_scale"] = "fast"
    v_str = DummyVideo(visual_style_id="stickman_tactical_explainer", visual_style_snapshot=snap_str)
    with pytest.raises(VisualStyleResolutionError):
        generate_render_step(video=v_str, prev=prev)


def test_malformed_frozen_snapshot_fails_closed(tmp_path: Path, monkeypatch):
    """Verify malformed frozen snapshot raises VisualStyleResolutionError and does not render."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    _mock_ffmpeg_execution(monkeypatch)

    v_corrupt = DummyVideo(
        visual_style_id="stickman_tactical_explainer",
        visual_style_snapshot={"corrupt": "data", "missing_required_id": True},
    )
    prev = _setup_mock_render_pipeline(tmp_path, v_corrupt)
    with pytest.raises(VisualStyleResolutionError):
        generate_render_step(video=v_corrupt, prev=prev)


def test_unknown_style_id_and_incompatible_version_fail_closed(tmp_path: Path, monkeypatch):
    """Verify unknown style ID or incompatible version when snapshot is absent fails closed."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    _mock_ffmpeg_execution(monkeypatch)

    # 1. Unknown style ID
    v_unknown = DummyVideo(visual_style_id="nonexistent_style_xyz", visual_style_snapshot=None)
    prev = _setup_mock_render_pipeline(tmp_path, v_unknown)
    with pytest.raises(VisualStyleResolutionError) as exc_unk:
        generate_render_step(video=v_unknown, prev=prev)
    assert "Unknown visual style ID" in str(exc_unk.value)

    # 2. Incompatible version
    v_incompat = DummyVideo(
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=99,
        visual_style_snapshot=None,
    )
    with pytest.raises(VisualStyleResolutionError) as exc_inc:
        generate_render_step(video=v_incompat, prev=prev)
    assert "Incompatible visual style version" in str(exc_inc.value)


def test_render_telemetry_records_style_id_motion_scale_and_calibrated_scales(tmp_path: Path, monkeypatch):
    """Verify render output metadata and animations telemetry records style ID, motion scale, and calibrated scales.

    Tests Stickman, Stylized 2D, and Legacy NULL-style rendering explicitly.
    """
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    _mock_ffmpeg_execution(monkeypatch)

    # 1. Stickman Tactical Explainer
    v_stickman = DummyVideo(
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=1,
        visual_style_snapshot=STICKMAN_TACTICAL_EXPLAINER_PRESET.model_dump(),
    )
    prev_stickman = _setup_mock_render_pipeline(tmp_path, v_stickman)
    res_stickman = generate_render_step(video=v_stickman, prev=prev_stickman)

    assert res_stickman["meta"]["visual_style_id"] == "stickman_tactical_explainer"
    assert res_stickman["meta"]["motion_scale"] == 0.6
    assert len(res_stickman["render"]["animations"]) == 2
    assert res_stickman["render"]["animations"][0]["end_scale"] == 1.030
    assert res_stickman["render"]["animations"][1]["start_scale"] == 1.030

    # 2. Stylized 2D Narrative Drama
    v_stylized = DummyVideo(
        visual_style_id="stylized_2d_narrative",
        visual_style_version=1,
        visual_style_snapshot=STYLIZED_2D_NARRATIVE_PRESET.model_dump(),
    )
    prev_stylized = _setup_mock_render_pipeline(tmp_path, v_stylized)
    res_stylized = generate_render_step(video=v_stylized, prev=prev_stylized)

    assert res_stylized["meta"]["visual_style_id"] == "stylized_2d_narrative"
    assert res_stylized["meta"]["motion_scale"] == 0.8
    assert len(res_stylized["render"]["animations"]) == 2
    assert res_stylized["render"]["animations"][0]["end_scale"] == 1.040
    assert res_stylized["render"]["animations"][1]["start_scale"] == 1.040

    # 3. Legacy NULL-style rendering
    v_null = DummyVideo(
        visual_style_id=None,
        visual_style_version=None,
        visual_style_snapshot=None,
    )
    prev_null = _setup_mock_render_pipeline(tmp_path, v_null)
    res_null = generate_render_step(video=v_null, prev=prev_null)

    assert res_null["meta"]["visual_style_id"] is None
    assert res_null["meta"]["motion_scale"] == 1.0
    assert len(res_null["render"]["animations"]) == 2
    assert res_null["render"]["animations"][0]["end_scale"] == 1.050
    assert res_null["render"]["animations"][1]["start_scale"] == 1.050

