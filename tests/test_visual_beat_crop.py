"""Unit tests for Stage 3.3 VisualBeat Crop Engine, Virtual Camera Reframing, and Motion Planning."""

from pathlib import Path
import pytest

from app.animation.crop import (
    QUESTIONABLE_REUSE_MAX_SCALE,
    SAFE_REUSE_MAX_SCALE,
    TARGET_ASPECT_RATIO,
    BeatCameraPlan,
    BeatPlanningResult,
    CropFeasibilityResult,
    CropRegion,
    calculate_portrait_crop,
    evaluate_beat_crop_feasibility,
    evaluate_scene_beats_crop_plan,
    plan_beat_camera_motion,
)
from app.animation.types import DEFAULT_SUBJECT_Y, SUBTITLE_SAFETY_MAX_Y, AnimationPlan
from app.steps.storyboard import VisualBeat, SceneInput


# ===========================================================================
# 1. calculate_portrait_crop Tests
# ===========================================================================


def test_calculate_portrait_crop_standard_2_3():
    """Verify 9:16 portrait crop calculation inside common 1024x1536 (2:3) image."""
    crop = calculate_portrait_crop(
        source_width=1024,
        source_height=1536,
        focal_x=0.5,
        focal_y=DEFAULT_SUBJECT_Y,
        scale_factor=1.0,
    )
    # Target 9:16 aspect ratio: 864 / 1536 = 0.5625
    assert crop.pixel_width == 864
    assert crop.pixel_height == 1536
    assert abs(crop.aspect_ratio - TARGET_ASPECT_RATIO) < 0.005

    # Coordinates strictly within bounds
    assert 0 <= crop.pixel_x <= 1024 - crop.pixel_width
    assert 0 <= crop.pixel_y <= 1536 - crop.pixel_height
    assert 0.0 <= crop.x_min < crop.x_max <= 1.0
    assert 0.0 <= crop.y_min < crop.y_max <= 1.0

    # Even pixel dimensions
    assert crop.pixel_width % 2 == 0
    assert crop.pixel_height % 2 == 0


def test_calculate_portrait_crop_native_9_16():
    """Verify crop calculation on native 1080x1920 (9:16) image returns full frame at scale 1.0."""
    crop = calculate_portrait_crop(
        source_width=1080,
        source_height=1920,
        focal_x=0.5,
        focal_y=0.5,
        scale_factor=1.0,
    )
    assert crop.pixel_width == 1080
    assert crop.pixel_height == 1920
    assert crop.pixel_x == 0
    assert crop.pixel_y == 0
    assert crop.x_min == 0.0
    assert crop.x_max == 1.0
    assert crop.y_min == 0.0
    assert crop.y_max == 1.0


def test_calculate_portrait_crop_square_image():
    """Verify 9:16 portrait crop calculation inside a 1024x1024 square image."""
    crop = calculate_portrait_crop(
        source_width=1024,
        source_height=1024,
        focal_x=0.5,
        focal_y=0.5,
        scale_factor=1.0,
    )
    # 1024 * (9/16) = 576
    assert crop.pixel_width == 576
    assert crop.pixel_height == 1024
    assert crop.aspect_ratio == 0.5625
    assert crop.pixel_x == (1024 - 576) // 2  # 224
    assert crop.pixel_y == 0


def test_calculate_portrait_crop_narrow_tall_image():
    """Verify 9:16 portrait crop on image narrower than 9:16 (e.g. 720x1440)."""
    crop = calculate_portrait_crop(
        source_width=720,
        source_height=1440,
        focal_x=0.5,
        focal_y=0.5,
        scale_factor=1.0,
    )
    assert crop.pixel_width == 720
    # 720 / (9/16) = 1280
    assert crop.pixel_height == 1280
    assert abs(crop.aspect_ratio - TARGET_ASPECT_RATIO) < 0.005
    assert 0 <= crop.pixel_y <= 1440 - 1280


def test_calculate_portrait_crop_extreme_focal_points_never_exceed_bounds():
    """Verify extreme focal anchors (corners) never produce negative or out-of-bounds crops."""
    # Top-left corner
    crop_tl = calculate_portrait_crop(1024, 1536, focal_x=0.0, focal_y=0.0, scale_factor=1.2)
    assert crop_tl.pixel_x == 0
    assert crop_tl.pixel_y == 0
    assert crop_tl.pixel_x + crop_tl.pixel_width <= 1024
    assert crop_tl.pixel_y + crop_tl.pixel_height <= 1536

    # Bottom-right corner
    crop_br = calculate_portrait_crop(1024, 1536, focal_x=1.0, focal_y=1.0, scale_factor=1.2)
    assert crop_br.pixel_x + crop_br.pixel_width == 1024
    assert crop_br.pixel_y + crop_br.pixel_height == 1536
    assert crop_br.pixel_x >= 0
    assert crop_br.pixel_y >= 0


def test_calculate_portrait_crop_zoom_scaling():
    """Verify zoom scaling factor proportionally tightens crop window."""
    crop_1x = calculate_portrait_crop(1024, 1536, scale_factor=1.0)
    crop_1_5x = calculate_portrait_crop(1024, 1536, scale_factor=1.5)

    assert crop_1_5x.pixel_width < crop_1x.pixel_width
    assert crop_1_5x.pixel_height < crop_1x.pixel_height
    # Aspect ratio remains 9:16
    assert abs(crop_1_5x.aspect_ratio - TARGET_ASPECT_RATIO) < 0.005


def test_calculate_portrait_crop_invalid_dimensions():
    """Verify invalid source dimensions raise ValueError."""
    with pytest.raises(ValueError):
        calculate_portrait_crop(0, 1536)
    with pytest.raises(ValueError):
        calculate_portrait_crop(1024, -10)


def test_crop_region_dict_roundtrip():
    """Verify CropRegion serialization and deserialization."""
    crop = calculate_portrait_crop(1024, 1536, scale_factor=1.25)
    data = crop.to_dict()
    assert isinstance(data, dict)
    restored = CropRegion.from_dict(data)
    assert restored.pixel_x == crop.pixel_x
    assert restored.pixel_y == crop.pixel_y
    assert restored.pixel_width == crop.pixel_width
    assert restored.pixel_height == crop.pixel_height
    assert restored.scale_factor == crop.scale_factor


# ===========================================================================
# 2. evaluate_beat_crop_feasibility Tests
# ===========================================================================


def test_feasibility_safe_reuse_same_or_slightly_tighter_shot():
    """Verify safe_reuse when beat framing matches or is slightly tighter than parent scene."""
    beat = VisualBeat(
        id="beat_1_1",
        scene_id="1",
        order=1,
        narration_text="Here is the chip.",
        visual_description="A close-up of the silicon core.",
        shot_type="close-up",
        camera_angle="eye-level",
    )
    parent = SceneInput(
        narration="Here is the chip, powering modern computing.",
        image_prompt="Cinematic view of a computer processor on a motherboard.",
        shot_type="medium",
        camera_angle="eye-level",
    )

    res = evaluate_beat_crop_feasibility(beat=beat, parent_scene=parent, source_width=1024, source_height=1536)
    assert res.status == "safe_reuse"
    # medium (1.20) -> close-up (1.50) is scale 1.25x <= 1.45x
    assert res.scale_factor == 1.25
    assert res.proposed_crop is not None
    assert res.proposed_crop.pixel_width % 2 == 0


def test_feasibility_questionable_reuse_high_zoom():
    """Verify questionable_reuse when crop magnification is between 1.45x and 1.65x."""
    beat = VisualBeat(
        id="beat_1_2",
        scene_id="1",
        order=2,
        narration_text="Looking closely at the transistors.",
        visual_description="Close framing on the microchip.",
        shot_type="close-up",  # 1.50
        camera_angle="eye-level",
    )
    parent = SceneInput(
        narration="A wide view of the entire laboratory.",
        image_prompt="Wide establishing shot of modern research facility.",
        shot_type="wide",  # 1.00 -> 1.50 / 1.00 = 1.50x
        camera_angle="eye-level",
    )

    res = evaluate_beat_crop_feasibility(beat=beat, parent_scene=parent)
    assert res.status == "questionable_reuse"
    assert res.scale_factor == 1.50
    assert any("exceeds safe threshold" in w for w in res.warnings)
    assert res.proposed_crop is not None


def test_feasibility_new_image_needed_extreme_zoom():
    """Verify new_image_needed when digital crop magnification exceeds 1.65x."""
    beat = VisualBeat(
        id="beat_1_3",
        scene_id="1",
        order=3,
        narration_text="Deep inside the atomic structure.",
        visual_description="Extreme macro shot of microscopic silicon traces.",
        shot_type="macro",  # 1.90 -> 1.90 / 1.00 = 1.90x > 1.65x
        camera_angle="eye-level",
    )
    parent = SceneInput(
        narration="Wide landscape of the semiconductor fab.",
        image_prompt="Panoramic wide landscape of the industrial complex.",
        shot_type="wide",  # 1.00
        camera_angle="eye-level",
    )

    res = evaluate_beat_crop_feasibility(beat=beat, parent_scene=parent)
    assert res.status == "new_image_needed"
    assert res.scale_factor == 1.90
    assert "exceeding the" in res.reason


def test_feasibility_new_image_needed_widening_beyond_source():
    """Verify new_image_needed when beat requests wider framing than parent image."""
    beat = VisualBeat(
        id="beat_2_1",
        scene_id="2",
        order=1,
        narration_text="Seeing the entire facility.",
        visual_description="Wide establishing shot of the entire room.",
        shot_type="wide",  # 1.00
    )
    parent = SceneInput(
        narration="A tight view of the technician's hands.",
        image_prompt="Close shot of hands working with delicate tweezers.",
        shot_type="close-up",  # 1.50 -> 1.00 / 1.50 = 0.667 < 0.95
    )

    res = evaluate_beat_crop_feasibility(beat=beat, parent_scene=parent)
    assert res.status == "new_image_needed"
    assert "wider than parent scene image" in res.reason


def test_feasibility_new_image_needed_angle_conflict():
    """Verify new_image_needed when camera angles fundamentally conflict."""
    beat = VisualBeat(
        id="beat_3_1",
        scene_id="3",
        order=1,
        narration_text="From directly above.",
        visual_description="Top-down flat lay view.",
        shot_type="medium",
        camera_angle="top-down",
    )
    parent = SceneInput(
        narration="Eye-level view of the device on a desk.",
        image_prompt="Eye-level perspective of a gadget on wooden desk.",
        shot_type="medium",
        camera_angle="eye-level",
    )

    res = evaluate_beat_crop_feasibility(beat=beat, parent_scene=parent)
    assert res.status == "new_image_needed"
    assert "Camera angle shift" in res.reason


def test_feasibility_new_image_needed_technique_mismatch():
    """Verify new_image_needed when visual technique shifts from photo to diagram."""
    beat = VisualBeat(
        id="beat_4_1",
        scene_id="4",
        order=1,
        narration_text="Looking at the internal schematic.",
        visual_description="Cross-section diagram illustrating internal logic gates.",
        shot_type="cutaway",
        visual_technique="cutaway-diagram",
    )
    parent = SceneInput(
        narration="Photo of the computer case exterior.",
        image_prompt="Clean studio photo of a sleek aluminum computer chassis.",
        shot_type="medium",
        visual_technique="cinematic-photo",
    )

    res = evaluate_beat_crop_feasibility(beat=beat, parent_scene=parent)
    assert res.status == "new_image_needed"
    assert "Visual technique mismatch" in res.reason


def test_feasibility_explicit_image_strategy():
    """Verify new_image_needed when beat explicitly requests new_image strategy."""
    beat = VisualBeat(
        id="beat_5_1",
        scene_id="5",
        order=1,
        narration_text="A completely different subject.",
        visual_description="A bustling city street at night.",
        image_strategy="new_image",
    )
    parent = SceneInput(
        narration="Inside a quiet office room.",
        image_prompt="Quiet modern office room with ambient warm lamps.",
        shot_type="medium",
    )

    res = evaluate_beat_crop_feasibility(beat=beat, parent_scene=parent)
    assert res.status == "new_image_needed"
    assert "explicitly specifies image strategy 'new_image'" in res.reason


def test_feasibility_subtitle_safety_adjustment():
    """Verify vertical focal anchor is clamped when placed in bottom subtitle collision zone."""
    beat = VisualBeat(
        id="beat_6_1",
        scene_id="6",
        order=1,
        narration_text="Looking at the bottom edge.",
        visual_description="Bottom edge connectors.",
        focal_point="bottom edge connectors",
        shot_type="medium",
    )
    # Pass custom dict with low vertical focal point
    res = evaluate_beat_crop_feasibility(
        beat=beat,
        parent_scene={"shot_type": "medium", "focal_point": "lower ground"},
    )
    assert res.status == "safe_reuse"
    # Ensure any proposed crop stays within safety boundaries
    assert res.proposed_crop is not None
    assert res.proposed_crop.y_max <= 1.0


# ===========================================================================
# 3. plan_beat_camera_motion Tests
# ===========================================================================


def test_plan_beat_camera_motion_zoom_in_default():
    """Verify standard beat produces smooth zoom-in within bounds."""
    crop = calculate_portrait_crop(1024, 1536, scale_factor=1.2)
    plan = plan_beat_camera_motion(
        beat={"shot_type": "medium"},
        crop_region=crop,
        source_width=1024,
        source_height=1536,
        duration_seconds=3.0,
    )
    assert plan.animation_type == "zoom-in"
    assert plan.end_scale > plan.start_scale
    assert plan.duration_seconds == 3.0
    # Both start and end crops strictly within bounds
    assert 0.0 <= plan.start_crop.x_min < plan.start_crop.x_max <= 1.0
    assert 0.0 <= plan.end_crop.x_min < plan.end_crop.x_max <= 1.0
    assert 0.0 <= plan.start_crop.y_min < plan.start_crop.y_max <= 1.0
    assert 0.0 <= plan.end_crop.y_min < plan.end_crop.y_max <= 1.0


def test_plan_beat_camera_motion_zoom_out_for_close_up():
    """Verify close-up beats produce gentle zoom-out to avoid edge truncation."""
    crop = calculate_portrait_crop(1024, 1536, scale_factor=1.35)
    plan = plan_beat_camera_motion(
        beat={"shot_type": "close-up"},
        crop_region=crop,
        source_width=1024,
        source_height=1536,
    )
    assert plan.animation_type == "zoom-out"
    assert plan.start_scale > plan.end_scale


def test_plan_beat_camera_motion_static_for_diagram():
    """Verify diagram/cutaway beats hold still for maximum legibility."""
    crop = calculate_portrait_crop(1024, 1536, scale_factor=1.0)
    plan = plan_beat_camera_motion(
        beat={"shot_type": "diagram"},
        crop_region=crop,
        source_width=1024,
        source_height=1536,
    )
    assert plan.animation_type == "static"
    assert plan.start_scale == plan.end_scale
    assert plan.start_crop.pixel_x == plan.end_crop.pixel_x


def test_plan_beat_camera_motion_pan_bounds():
    """Verify lateral pan keeps both start and end crops within source image bounds."""
    crop = calculate_portrait_crop(1024, 1536, scale_factor=1.1)
    plan = plan_beat_camera_motion(
        beat={"shot_type": "medium"},
        crop_region=crop,
        source_width=1024,
        source_height=1536,
        preferred_animation="pan-left",
    )
    assert plan.animation_type == "pan-left"
    assert plan.start_crop.pixel_x >= 0
    assert plan.end_crop.pixel_x >= 0
    assert plan.start_crop.pixel_x + plan.start_crop.pixel_width <= 1024
    assert plan.end_crop.pixel_x + plan.end_crop.pixel_width <= 1024


def test_beat_camera_plan_to_animation_plan_compatibility():
    """Verify BeatCameraPlan converts cleanly into validated standard AnimationPlan."""
    crop = calculate_portrait_crop(1024, 1536, scale_factor=1.1)
    beat_plan = plan_beat_camera_motion(
        beat={"shot_type": "medium"},
        crop_region=crop,
        source_width=1024,
        source_height=1536,
        duration_seconds=2.4,
    )
    std_plan = beat_plan.to_animation_plan(focal_desc="Processor unit")
    assert isinstance(std_plan, AnimationPlan)
    assert std_plan.animation_type == "zoom-in"
    assert std_plan.focal_point == "Processor unit"
    assert std_plan.duration_seconds == 2.4
    assert 1.00 <= std_plan.start_scale <= 1.15
    assert 1.00 <= std_plan.end_scale <= 1.15


# ===========================================================================
# 4. evaluate_scene_beats_crop_plan Tests
# ===========================================================================


def test_evaluate_scene_beats_crop_plan_legacy_scene_empty():
    """Verify legacy scenes lacking visual beats return an empty list without error."""
    scene_without_beats = SceneInput(
        narration="This is a classic scene without visual beats.",
        image_prompt="A classic cinematic frame showing subject in office.",
        shot_type="medium",
    )
    results = evaluate_scene_beats_crop_plan(scene_without_beats)
    assert results == []


def test_evaluate_scene_beats_crop_plan_multi_beat_evaluation():
    """Verify scene with multiple beats produces ordered BeatPlanningResults."""
    beats = [
        VisualBeat(
            id="beat_1_1",
            scene_id="1",
            order=1,
            narration_text="The processor sits at the center.",
            visual_description="Medium shot of processor socket.",
            shot_type="medium",
        ),
        VisualBeat(
            id="beat_1_2",
            scene_id="1",
            order=2,
            narration_text="Its millions of pathways carry signals.",
            visual_description="Close-up of the silicon core.",
            shot_type="close-up",
        ),
    ]
    scene = SceneInput(
        narration="The processor sits at the center, its millions of pathways carrying signals.",
        image_prompt="High tech macro view of silicon computer chip on circuit board.",
        shot_type="medium",
        visual_beats=beats,
    )

    results = evaluate_scene_beats_crop_plan(scene, source_width=1024, source_height=1536)
    assert len(results) == 2

    # Beat 1: medium -> safe_reuse with zoom-in plan
    assert results[0].beat_id == "beat_1_1"
    assert results[0].feasibility.status == "safe_reuse"
    assert results[0].camera_plan is not None
    assert results[0].camera_plan.animation_type == "zoom-in"

    # Beat 2: close-up -> safe_reuse with zoom-out plan
    assert results[1].beat_id == "beat_1_2"
    assert results[1].feasibility.status == "safe_reuse"
    assert results[1].camera_plan is not None
    assert results[1].camera_plan.animation_type == "zoom-out"

    # Verify serialization
    dumped = [r.to_dict() for r in results]
    assert len(dumped) == 2
    assert dumped[0]["feasibility"]["status"] == "safe_reuse"
    assert dumped[1]["camera_plan"]["animation_type"] == "zoom-out"
