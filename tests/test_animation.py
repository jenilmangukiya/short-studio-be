"""Unit and integration tests for conservative AI-directed camera animation.

Tests:
1. AnimationPlan data model, coordinate clamping, zoom scale limits, and subtitle safety zone.
2. Local image visual composition analysis (edge energy, centroids, dispersion).
3. Conservative animation planning: smooth zoom default, static diagrams, pull-outs for close-ups,
   and preservation of natural repetition without forced pans.
4. FFmpeg zoompan filter compilation with smoothstep Hermite easing and safe fallback.
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import pytest

from app.animation.analyzer import analyze_image_composition
from app.animation.planner import (
    _infer_semantic_focal_point,
    plan_scene_animation,
)
from app.animation.renderer import (
    build_fallback_ken_burns_filter,
    build_ffmpeg_zoompan_filter,
)
from app.animation.types import (
    ALLOWED_ANIMATION_TYPES,
    DEFAULT_SUBJECT_Y,
    SUBTITLE_SAFETY_MAX_Y,
    AnimationPlan,
)


# ---------------------------------------------------------------------------
# 1. AnimationPlan Data Model & Validation Tests
# ---------------------------------------------------------------------------


def test_animation_plan_defaults():
    """Verify default values meet conservative motion parameters."""
    plan = AnimationPlan()
    assert plan.animation_type == "zoom-in"
    assert plan.focal_x == 0.5
    assert plan.focal_y == DEFAULT_SUBJECT_Y
    assert plan.start_scale == 1.00
    assert plan.end_scale == 1.05
    assert plan.intensity == "subtle"


def test_animation_plan_clamping_scales():
    """Verify scales are clamped strictly within [1.00, 1.15] to prevent blur or extreme zoom."""
    # Scale too high clamped to 1.15
    plan_high = AnimationPlan(start_scale=1.50, end_scale=2.00).validate_and_clamp()
    assert plan_high.start_scale == 1.15
    assert plan_high.end_scale == 1.15

    # Scale too low clamped to 1.00
    plan_low = AnimationPlan(start_scale=0.80, end_scale=0.90).validate_and_clamp()
    assert plan_low.start_scale == 1.00
    assert plan_low.end_scale == 1.00

    # Static animation locks zoom to exactly 1.00
    plan_static = AnimationPlan(
        animation_type="static", start_scale=1.10, end_scale=1.12
    ).validate_and_clamp()
    assert plan_static.start_scale == 1.00
    assert plan_static.end_scale == 1.00


def test_animation_plan_clamping_coordinates_and_subtitle_safe_zone():
    """Verify viewport center never exceeds frame boundaries or intrudes into subtitle safe zone."""
    # Negative and excessive coordinates clamped
    plan = AnimationPlan(
        focal_x=-0.2,
        focal_y=0.95,  # In the lower-third subtitle area
        start_x=-0.5,
        start_y=0.99,
        end_x=1.8,
        end_y=0.85,
    ).validate_and_clamp()

    # X clamped to [0.05, 0.95]
    assert plan.focal_x == 0.05
    assert plan.start_x == 0.05
    assert plan.end_x == 0.95

    # Y clamped to max SUBTITLE_SAFETY_MAX_Y (0.68)
    assert plan.focal_y <= SUBTITLE_SAFETY_MAX_Y
    assert plan.start_y <= SUBTITLE_SAFETY_MAX_Y
    assert plan.end_y <= SUBTITLE_SAFETY_MAX_Y


def test_animation_plan_invalid_type_fallback():
    """Verify invalid or unknown animation type string safely reverts to 'zoom-in'."""
    plan = AnimationPlan(animation_type="invalid-spin-360").validate_and_clamp()
    assert plan.animation_type == "zoom-in"


def test_animation_plan_serialization_roundtrip():
    """Verify serialization to and from dictionary retains values accurately."""
    data = {
        "animation_type": "zoom-in",
        "focal_point": "vintage rivet detail",
        "focal_coords": [0.48, 0.42],
        "start_coords": [0.48, 0.42],
        "end_coords": [0.48, 0.42],
        "start_scale": 1.00,
        "end_scale": 1.05,
        "intensity": "subtle",
        "duration_seconds": 3.2,
        "reasoning": "Subtle push-in toward copper rivet.",
    }
    plan = AnimationPlan.from_dict(data)
    assert plan.animation_type == "zoom-in"
    assert plan.focal_x == 0.48
    assert plan.focal_y == 0.42
    assert plan.start_scale == 1.00
    assert plan.end_scale == 1.05
    assert plan.duration_seconds == 3.2

    serialized = plan.to_dict()
    assert serialized["animation_type"] == "zoom-in"
    assert serialized["focal_coords"] == [0.48, 0.42]
    assert serialized["start_coords"] == [0.48, 0.42]
    assert serialized["end_coords"] == [0.48, 0.42]


# ---------------------------------------------------------------------------
# 2. Local Image Composition Analysis Tests
# ---------------------------------------------------------------------------


def test_analyze_image_composition_missing_file():
    """Verify missing file returns safe default metrics."""
    res = analyze_image_composition(None)
    assert res["centroid_x"] == 0.5
    assert res["centroid_y"] == DEFAULT_SUBJECT_Y
    assert res["composition_type"] == "balanced_scene"

    res_missing = analyze_image_composition(Path("/nonexistent/image.png"))
    assert res_missing["composition_type"] == "balanced_scene"


def test_analyze_image_composition_diffuse_blank(tmp_path):
    """Verify flat solid color image is classified as diffuse low contrast."""
    blank = Image.new("RGB", (640, 960), color=(128, 128, 128))
    img_path = tmp_path / "blank.png"
    blank.save(img_path)

    res = analyze_image_composition(img_path)
    assert res["composition_type"] == "diffuse_low_contrast"
    assert res["centroid_x"] == 0.5
    assert res["centroid_y"] == DEFAULT_SUBJECT_Y


def test_analyze_image_composition_left_focal_point(tmp_path):
    """Verify image with high contrast on left side yields left-shifted centroid."""
    img = Image.new("RGB", (600, 900), color=(10, 10, 10))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 150, 200, 350], fill=(255, 255, 255))
    draw.line([50, 150, 200, 350], fill=(0, 0, 0), width=3)
    draw.line([50, 350, 200, 150], fill=(0, 0, 0), width=3)

    img_path = tmp_path / "left_subject.png"
    img.save(img_path)

    res = analyze_image_composition(img_path)
    assert res["centroid_x"] < 0.40, f"Expected left centroid, got {res['centroid_x']}"
    assert res["centroid_y"] < 0.45


def test_analyze_image_composition_panoramic_horizontal_spread(tmp_path):
    """Verify wide horizontal horizon with narrow vertical profile classifies as horizontal_panoramic."""
    img = Image.new("RGB", (600, 900), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 380, 590, 420], fill=(240, 240, 240))

    img_path = tmp_path / "panoramic.png"
    img.save(img_path)

    res = analyze_image_composition(img_path)
    assert res["composition_type"] == "horizontal_panoramic"
    assert res["spread_x"] > 0.25


def test_analyze_image_composition_vertical_stack(tmp_path):
    """Verify tall vertical pillar classifies as vertical_stack."""
    img = Image.new("RGB", (600, 900), color=(20, 20, 20))
    draw = ImageDraw.Draw(img)
    draw.rectangle([280, 50, 320, 850], fill=(250, 250, 250))

    img_path = tmp_path / "vertical.png"
    img.save(img_path)

    res = analyze_image_composition(img_path)
    assert res["composition_type"] == "vertical_stack"
    assert res["spread_y"] > 0.25


# ---------------------------------------------------------------------------
# 3. Intelligent Conservative Animation Planning Engine Tests
# ---------------------------------------------------------------------------


def test_semantic_focal_point_inference():
    """Verify semantic textual cues steer normalized focal coordinates within restrained bounds."""
    # Top-left cues
    x_tl, y_tl = _infer_semantic_focal_point({
        "focal_point": "upper-left window corner",
        "image_prompt": "high ceiling on the left-side",
    })
    assert x_tl == 0.47
    assert y_tl == 0.38

    # Bottom-right cues
    x_br, y_br = _infer_semantic_focal_point({
        "focal_point": "ground pebble on the right-hand side",
        "image_prompt": "floor shadow on the bottom",
    })
    assert x_br == 0.53
    assert y_br == 0.46


def test_plan_scene_animation_extreme_closeup():
    """Verify extreme close-up generates gentle pull-out (zoom-out) so edges are not cropped."""
    scene = {
        "shot_type": "extreme-close-up",
        "focal_point": "tiny breather hole in window pane",
        "image_prompt": "Macro extreme close up of acrylic edge",
    }
    plan = plan_scene_animation(scene_data=scene, duration_seconds=3.0)
    assert plan.animation_type == "zoom-out"
    assert plan.start_scale > plan.end_scale
    assert plan.end_scale == 1.00
    assert "pull-out" in plan.reasoning.lower()


def test_plan_scene_animation_wide_uses_smooth_zoom_default():
    """Verify wide landscape defaults to smooth zoom-in rather than awkward sideways panning."""
    scene = {
        "shot_type": "wide",
        "focal_point": "vast desert horizon and sunset",
        "image_prompt": "Cinematic wide angle horizon view",
    }
    plan = plan_scene_animation(scene_data=scene, duration_seconds=4.0)
    # Default is smooth zoom-in, NOT horizontal pan
    assert plan.animation_type == "zoom-in"
    assert plan.start_scale == 1.00
    assert plan.end_scale == 1.05
    assert plan.start_x == plan.end_x  # No horizontal sliding
    assert "push-in" in plan.reasoning.lower()


def test_plan_scene_animation_diagram_static():
    """Verify technical diagram or schematic uses static hold to ensure text readability."""
    scene = {
        "shot_type": "diagram",
        "focal_point": "airflow pressure arrows",
        "image_prompt": "Cross section schematic showing airflow and pressure equalization",
    }
    plan = plan_scene_animation(scene_data=scene, duration_seconds=3.0)
    assert plan.animation_type == "static"
    assert plan.start_scale == 1.00
    assert plan.end_scale == 1.00
    assert "static hold" in plan.reasoning.lower()


def test_plan_scene_animation_cutaway_static():
    """Verify exploded cutaway diagram uses static hold for maximum legibility."""
    scene = {
        "shot_type": "cutaway",
        "focal_point": "three-pane acrylic layer assembly",
        "image_prompt": "Exploded technical cutaway diagram of window assembly layers",
    }
    plan = plan_scene_animation(scene_data=scene, duration_seconds=3.5)
    assert plan.animation_type == "static"
    assert plan.start_scale == 1.00
    assert plan.end_scale == 1.00


def test_plan_scene_animation_natural_repetition_preserved():
    """Verify engine preserves natural repetition across scenes without forced sideways drift."""
    scene1 = {
        "shot_type": "medium",
        "focal_point": "passenger sitting",
        "image_prompt": "Medium interior passenger shot",
    }
    plan1 = plan_scene_animation(scene1, duration_seconds=2.5, prev_anim_type=None)
    assert plan1.animation_type == "zoom-in"

    scene2 = {
        "shot_type": "medium",
        "focal_point": "flight attendant smiling",
        "image_prompt": "Medium shot of attendant",
    }
    # Scene 2 natural repetition is preserved (remains zoom-in, no awkward forced variety)
    plan2 = plan_scene_animation(scene2, duration_seconds=2.5, prev_anim_type="zoom-in")
    assert plan2.animation_type == "zoom-in"
    assert plan2.start_scale == 1.00
    assert plan2.end_scale == 1.05


def test_plan_scene_animation_preserves_explicit_metadata():
    """Verify that if upstream storyboard already provided an explicit animation plan, it is preserved."""
    explicit_plan = {
        "animation_type": "pan-up",
        "focal_point": "cabin ceiling air vent",
        "focal_coords": [0.50, 0.38],
        "start_coords": [0.50, 0.42],
        "end_coords": [0.50, 0.35],
        "start_scale": 1.05,
        "end_scale": 1.05,
        "intensity": "subtle",
        "reasoning": "Explicit storyboard instruction.",
    }
    scene = {
        "shot_type": "close-up",
        "animation": explicit_plan,
    }
    plan = plan_scene_animation(scene, duration_seconds=2.8)
    assert plan.animation_type == "pan-up"
    assert plan.focal_point == "cabin ceiling air vent"


# ---------------------------------------------------------------------------
# 4. FFmpeg Zoompan Filter Compilation & Smooth Easing Tests
# ---------------------------------------------------------------------------


def test_build_fallback_ken_burns_filter_smoothstep():
    """Verify safe fallback filter compiles valid FFmpeg expressions with smoothstep easing."""
    z, x, y, fc = build_fallback_ken_burns_filter(
        shot_type="wide",
        scene_idx=1,
        scene_dur=2.0,
        target_fps=30,
        target_width=1080,
        target_height=1920,
    )
    assert "zoompan" in fc
    # 2x lanczos oversampling + 4:4:4 zoompan output eliminates sub-pixel shake
    assert "scale=2160:3840" in fc
    assert "flags=lanczos" in fc
    assert fc.endswith("format=yuv444p")
    assert "s=1080x1920" in fc
    assert "fps=30" in fc
    # Verify smoothstep Hermite formula is present in zoom expression
    assert "(on/60)*(on/60)*(3-2*(on/60))" in z
    # Verify Phase 1 horizontal and vertical centering
    assert x == "iw/2-(iw/zoom/2)+0.5"
    assert "0.42" in y


def test_build_ffmpeg_zoompan_filter_none_input_fallback():
    """Verify None plan safely uses fallback without raising an exception."""
    z, x, y, fc, meta = build_ffmpeg_zoompan_filter(
        plan=None,
        scene_dur=2.0,
        target_fps=30,
        target_width=1080,
        target_height=1920,
        fallback_shot_type="close-up",
        fallback_scene_idx=2,
    )
    assert meta["animation_type"] == "fallback-ken-burns"
    assert "zoompan=" in fc


def test_build_ffmpeg_zoompan_filter_smooth_zoom_in():
    """Verify zoom-in expression uses smoothstep easing and Phase 1 centered coordinates."""
    plan = AnimationPlan(
        animation_type="zoom-in",
        focal_x=0.50,
        focal_y=0.42,
        start_scale=1.00,
        end_scale=1.05,
        duration_seconds=2.0,
    )
    z_expr, x_expr, y_expr, fc, meta = build_ffmpeg_zoompan_filter(
        plan=plan,
        scene_dur=2.0,
        target_fps=30,
        target_width=1080,
        target_height=1920,
    )

    # Smoothstep easing present
    assert "(on/60)*(on/60)*(3-2*(on/60))" in z_expr
    assert "(1.000+(+0.050)*" in z_expr
    # Zoom is quantized to even crop widths (prevents odd/even parity wobble)
    assert z_expr.startswith("(iw/(2*round(iw/(2*")

    # X and Y match Phase 1 Ken Burns anchor
    assert x_expr == "iw/2-(iw/zoom/2)+0.5"
    assert "0.42" in y_expr


def test_build_ffmpeg_zoompan_filter_static():
    """Verify static hold outputs constant scale 1.000 with centered crop."""
    plan = AnimationPlan(
        animation_type="static",
        focal_x=0.50,
        focal_y=0.42,
        start_scale=1.00,
        end_scale=1.00,
    )
    z_expr, x_expr, y_expr, fc, meta = build_ffmpeg_zoompan_filter(
        plan=plan,
        scene_dur=2.5,
        target_fps=30,
    )
    assert z_expr == "1.000"
    assert x_expr == "iw/2-(iw/zoom/2)+0.5"


# ---------------------------------------------------------------------------
# 5. Phase 1 stability fixes: pan coordinate preservation, drift, and no shake
# ---------------------------------------------------------------------------


def _write_centered_image(path: Path) -> Path:
    img = Image.new("RGB", (1024, 1536), color=(20, 25, 35))
    draw = ImageDraw.Draw(img)
    draw.rectangle([412, 545, 612, 745], fill=(230, 200, 80))
    img.save(path)
    return path


def test_planner_preserves_explicit_pan_coordinates_with_default_focal(tmp_path):
    """Planned pan start/end must not be overwritten when focal is the default anchor."""
    img = _write_centered_image(tmp_path / "pan.png")
    plan = plan_scene_animation(
        scene_data={
            "animation": {
                "animation_type": "pan-right",
                "focal_x": 0.5,
                "focal_y": DEFAULT_SUBJECT_Y,
                "start_x": 0.40,
                "end_x": 0.60,
                "start_y": 0.42,
                "end_y": 0.42,
                "start_scale": 1.08,
                "end_scale": 1.08,
            }
        },
        image_path=img,
        duration_seconds=3.0,
    )
    assert plan.animation_type == "pan-right"
    assert plan.start_x < plan.end_x
    assert abs(plan.end_x - plan.start_x) >= 0.1
    # Orthogonal axis is locked (no cross-axis wobble) and overscan allows translation
    assert plan.start_y == plan.end_y
    assert plan.start_scale >= 1.05 and plan.end_scale >= 1.05


def test_planner_zoom_locks_position_and_static_locks_scale(tmp_path):
    img = _write_centered_image(tmp_path / "zoom.png")
    zoom = plan_scene_animation(
        scene_data={
            "animation": {
                "animation_type": "zoom-in",
                "focal_x": 0.5,
                "focal_y": 0.42,
                "start_x": 0.3,
                "end_x": 0.7,
                "start_scale": 1.0,
                "end_scale": 1.05,
            }
        },
        image_path=img,
        duration_seconds=3.0,
    )
    assert zoom.start_x == zoom.end_x == zoom.focal_x
    assert zoom.start_y == zoom.end_y == zoom.focal_y
    assert zoom.end_scale > zoom.start_scale

    static = plan_scene_animation(
        scene_data={"animation": {"animation_type": "static", "start_scale": 1.1, "end_scale": 1.2}},
        image_path=img,
        duration_seconds=3.0,
    )
    assert static.start_scale == static.end_scale == 1.0


def test_planner_centroid_deadzone_snaps_near_center(tmp_path):
    """Near-centered subjects snap to the exact default anchor (no micro-offset jitter)."""
    img = _write_centered_image(tmp_path / "center.png")
    plan = plan_scene_animation(
        scene_data={"narration": "A glowing chip.", "shot_type": "medium"},
        image_path=img,
        duration_seconds=3.0,
    )
    assert plan.focal_x == 0.5
    assert plan.focal_y in (DEFAULT_SUBJECT_Y, round(plan.focal_y, 2))


def test_renderer_subtle_drift_interpolates_in_bounds():
    """subtle-drift must interpolate position with in-bounds proportional anchors."""
    plan = AnimationPlan(
        animation_type="subtle-drift",
        focal_x=0.5,
        focal_y=0.42,
        start_x=0.46,
        end_x=0.54,
        start_y=0.42,
        end_y=0.42,
        start_scale=1.03,
        end_scale=1.06,
        duration_seconds=3.0,
    )
    _, x_expr, y_expr, _, meta = build_ffmpeg_zoompan_filter(
        plan=plan, scene_dur=3.0, target_fps=30, target_width=1080, target_height=1920
    )
    assert meta["animation_type"] == "subtle-drift"
    assert "(iw-trunc(iw/zoom))*(0.460+(+0.080)*" in x_expr
    assert "clip(" in x_expr and "iw-iw/zoom" in x_expr
    # Legacy out-of-bounds formula is gone
    assert "*iw)-(iw/zoom/2)" not in x_expr
    assert "(ih-trunc(ih/zoom))*0.420" in y_expr


@pytest.mark.parametrize("anim", ["zoom-in", "zoom-out"])
def test_rendered_zoom_has_no_back_and_forth_shake(tmp_path, anim):
    """Render a real zoom and verify the image centre never oscillates frame-to-frame."""
    import shutil
    import subprocess

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not available")

    # High-detail pattern so sub-pixel motion is measurable
    w, h = 512, 768
    yy, xx = np.mgrid[0:h, 0:w]
    pattern = (((xx // 16) + (yy // 16)) % 2 * 200 + 30).astype(np.uint8)
    img_path = tmp_path / "pattern.png"
    Image.fromarray(np.stack([pattern] * 3, axis=-1)).save(img_path)

    start, end = (1.0, 1.05) if anim == "zoom-in" else (1.05, 1.0)
    plan = AnimationPlan(
        animation_type=anim,
        focal_x=0.5,
        focal_y=0.42,
        start_scale=start,
        end_scale=end,
        duration_seconds=1.0,
    )
    tw, th, fps, frames = 216, 384, 30, 30
    _, _, _, fc, _ = build_ffmpeg_zoompan_filter(
        plan=plan, scene_dur=1.0, target_fps=fps, target_width=tw, target_height=th
    )
    proc = subprocess.run(
        [
            "ffmpeg", "-v", "error", "-i", str(img_path),
            "-filter_complex", f"[0:v]{fc},format=gray[v]", "-map", "[v]",
            "-frames:v", str(frames), "-f", "rawvideo", "-",
        ],
        stdout=subprocess.PIPE,
        check=True,
    )
    video = np.frombuffer(proc.stdout, dtype=np.uint8).reshape(frames, th, tw).astype(float)

    # Track horizontal centre-of-mass of edge energy; a smooth centred zoom keeps it steady.
    centres = []
    for fr in video:
        gx = np.abs(np.diff(fr, axis=1)).sum(axis=0)
        cols = np.arange(gx.size)
        centres.append(float((gx * cols).sum() / max(gx.sum(), 1e-6)))
    deltas = np.diff(centres)
    sign_flips = sum(
        1 for a, b in zip(deltas[:-1], deltas[1:]) if abs(a) > 0.25 and abs(b) > 0.25 and a * b < 0
    )
    assert sign_flips == 0, f"Detected {sign_flips} back-and-forth shake reversals: {deltas}"
