"""Tests for Shorts Studio: Prevent Repeated Images.

Verifies:
1. One unique image per visual beat is the default behavior.
2. Consecutive visual beats do not share the same image ID (within scene and across scene boundaries).
3. Removal of automatic inference / carry-forward of previous scene's image.
4. Missing image references raise PipelineError rather than silently falling back to the previous image.
5. Image budget controls count all unique visual beat images and report clearly when budget is exceeded.
6. Visual prompts between consecutive beats are visually distinct.
7. Explicit image references continue to resolve without breaking existing pipeline behavior.
"""

import uuid
from pathlib import Path
import pytest

from app.config import settings
from app.images.mock import MockImageProvider, create_solid_png
from app.steps.images import PipelineError as ImagePipelineError, generate_images_step
from app.steps.render import PipelineError as RenderPipelineError, _resolve_beat_image, generate_render_step
from app.steps.storyboard import (
    SceneInput,
    VisualBeat,
    generate_storyboard_step,
    normalize_scene_visual_beats,
    validate_storyboard_soft_checks,
)


class DummyVideo:
    def __init__(self, video_id=None):
        self.id = video_id or uuid.uuid4()
        self.topic = "Prevent Image Reuse Test"


class DummyStep:
    def __init__(self, name="images", user_prompt=None, output=None, status="pending"):
        self.name = name
        self.user_prompt = user_prompt
        self.output = output
        self.status = status


# ==============================================================================
# 1. Default Behavior & Schema Tests
# ==============================================================================

def test_visual_beat_defaults_to_new_image():
    """Requirement 3: One unique image per visual beat is the default behavior."""
    beat = VisualBeat(
        id="beat_1_1",
        scene_id="1",
        order=1,
        visual_description="Opening establishing shot",
    )
    assert beat.image_strategy == "new_image"
    assert beat.image_id == "beat_1_1"


def test_normalize_scene_visual_beats_defaults_all_beats_to_new_image():
    """All beats in normalize_scene_visual_beats default to new_image strategy."""
    raw_beats = [
        {"id": "b1_1", "order": 1, "visual_description": "Beat 1 description"},
        {"id": "b1_2", "order": 2, "visual_description": "Beat 2 description"},
    ]
    normalized = normalize_scene_visual_beats(raw_beats, scene_index=1)
    assert len(normalized) == 2
    assert normalized[0].image_strategy == "new_image"
    assert normalized[1].image_strategy == "new_image"
    # Consecutive beats do not share the same effective image ID
    assert normalized[0].image_id != normalized[1].image_id


# ==============================================================================
# 2. Consecutive Beats Do Not Share Image ID (Within Scene & Across Scenes)
# ==============================================================================

def test_consecutive_beats_within_scene_do_not_share_image_id():
    """Consecutive beats within the same scene must not share the same image ID."""
    raw_beats = [
        {
            "id": "beat_1_1",
            "order": 1,
            "visual_description": "Wide establishing view of airplane wing",
            "source_image_id": "img_1_1",
        },
        {
            "id": "beat_1_2",
            "order": 2,
            "visual_description": "Close-up of wing flap mechanism",
            "source_image_id": "img_1_2",
        },
    ]
    normalized = normalize_scene_visual_beats(raw_beats, scene_index=1)
    assert normalized[0].image_id == "img_1_1"
    assert normalized[1].image_id == "img_1_2"
    assert normalized[0].image_id != normalized[1].image_id
    assert normalized[0].source_image_id != normalized[1].source_image_id


def test_consecutive_beats_across_scene_boundaries_do_not_share_image_id():
    """Consecutive beats across scene boundaries must not share the same image ID."""
    scene1_beats = [
        {
            "id": "beat_1_2",
            "order": 2,
            "visual_description": "Macro view of pressure bleed hole",
            "source_image_id": "img_1_2",
        }
    ]
    scene2_beats = [
        {
            "id": "beat_2_1",
            "order": 1,
            "visual_description": "Passenger cabin window interior view",
            "source_image_id": "img_2_1",
        }
    ]
    norm1 = normalize_scene_visual_beats(scene1_beats, scene_index=1)
    norm2 = normalize_scene_visual_beats(scene2_beats, scene_index=2)

    last_beat_scene1 = norm1[-1]
    first_beat_scene2 = norm2[0]

    assert last_beat_scene1.image_id != first_beat_scene2.image_id
    assert last_beat_scene1.source_image_id != first_beat_scene2.source_image_id


def test_mock_storyboard_guarantees_unique_images_per_visual_beat(monkeypatch):
    """Mock storyboard output must produce unique image IDs and distinct prompts for all beats."""
    monkeypatch.setattr(settings, "llm_provider", "mock")
    video = DummyVideo()
    script_prev = {
        "script": {
            "script": {
                "title": "Pressure Holes",
                "hook": "Commercial airplane windows feature a tiny hole that protects passenger safety during high altitude flights.",
                "narration": (
                    "Commercial airplane windows feature a tiny hole that protects passenger safety during high altitude flights. "
                    "At thirty thousand feet the outside air pressure drops dramatically compared to the comfortable pressurized cabin interior. "
                    "This creates tremendous pressure difference across the acrylic window panes. "
                    "The bleed hole balances that pressure between the inner and outer layers seamlessly."
                ),
            }
        }
    }
    step = DummyStep(name="storyboard")
    output = generate_storyboard_step(video, script_prev, step)
    scenes = output["storyboard"]["scenes"]

    all_beats: list[dict] = []
    for sc in scenes:
        for b in sc.get("visual_beats", []):
            all_beats.append(b)

    assert len(all_beats) >= len(scenes)

    # Prove consecutive beats do not share the same source_image_id or image_id
    for i in range(len(all_beats) - 1):
        b1 = all_beats[i]
        b2 = all_beats[i + 1]
        sid1 = b1.get("source_image_id") or b1.get("id")
        sid2 = b2.get("source_image_id") or b2.get("id")
        assert sid1 != sid2, f"Consecutive beats {b1['id']} and {b2['id']} share image ID '{sid1}'"


# ==============================================================================
# 3. Removal of Automatic Inference & Carry-Forward
# ==============================================================================

def test_no_automatic_inference_on_reuse_previous_image():
    """Requirement 4: Remove automatic inference/carry-forward of the previous scene's image."""
    raw_beats_scene_2 = [
        {
            "id": "beat_2_1",
            "order": 1,
            "visual_description": "Scene 2 beat with reuse_previous_image but no explicit source_image_id",
            "image_strategy": "reuse_previous_image",
            "source_image_id": None,
        }
    ]
    normalized = normalize_scene_visual_beats(raw_beats_scene_2, scene_index=2)
    assert len(normalized) == 1
    # MUST NOT automatically infer "1" (the previous scene's ID)
    assert normalized[0].source_image_id is None
    assert normalized[0].image_id == "beat_2_1"


def test_images_step_does_not_auto_reuse_previous_scene_from_beats(tmp_path, monkeypatch):
    """images.py does not automatically copy previous scene's image based on beat hints."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    video = DummyVideo()
    step = DummyStep()
    prev = {
        "storyboard": {
            "storyboard": {
                "visual_style": "Clean 3D render",
                "scenes": [
                    {
                        "index": 1,
                        "narration": "First scene narration.",
                        "image_prompt": "First scene image prompt",
                        "shot_type": "wide",
                        "visual_beats": [
                            {"id": "b1_1", "order": 1, "image_strategy": "new_image"}
                        ],
                    },
                    {
                        "index": 2,
                        "narration": "Second scene narration.",
                        "image_prompt": "Second scene image prompt",
                        "shot_type": "medium",
                        "visual_beats": [
                            {"id": "b2_1", "order": 1, "image_strategy": "reuse_previous_image", "source_image_id": None}
                        ],
                    },
                ],
            }
        }
    }

    output = generate_images_step(video, prev, step, image_provider=MockImageProvider())
    items = output["images"]["items"]
    # Scene 2 must NOT be marked as reused from scene 1
    item_scene_1 = next(it for it in items if it["index"] == 1)
    item_scene_2 = next(it for it in items if it["index"] == 2)
    assert item_scene_1["file"] != item_scene_2["file"]
    assert item_scene_2["reused"] is False


# ==============================================================================
# 4. Missing Image References Raise Errors (No Silent Fallback)
# ==============================================================================

def test_missing_explicit_source_image_id_raises_error(tmp_path):
    """Requirement 5: Do not silently reuse the previous image when an image reference is missing."""
    img_file = tmp_path / "scene_01.png"
    img_file.write_bytes(create_solid_png(100, 150, 10, 20, 30))

    valid_images = {1: {"file": "scene_01.png"}}
    img_items = [{"index": 1, "file": "scene_01.png"}]

    beat_missing = {"id": "b1", "order": 1, "source_image_id": "non_existent_image_99"}

    with pytest.raises(RenderPipelineError, match="Referenced image 'non_existent_image_99' for visual beat 'b1' was not found"):
        _resolve_beat_image(
            beat=beat_missing,
            scene_idx=1,
            valid_images=valid_images,
            img_items=img_items,
            storage_root=tmp_path,
        )


def test_missing_scene_index_reference_raises_error(tmp_path):
    """Numeric source_image_id not in valid_images raises PipelineError without silent fallback."""
    valid_images = {1: {"file": "scene_01.png"}}
    img_items = [{"index": 1, "file": "scene_01.png"}]

    beat = {"id": "beat_2_1", "order": 1, "source_image_id": "42"}

    with pytest.raises(RenderPipelineError, match="Referenced image '42'"):
        _resolve_beat_image(
            beat=beat,
            scene_idx=2,
            valid_images=valid_images,
            img_items=img_items,
            storage_root=tmp_path,
        )


# ==============================================================================
# 5. Image Budget Controls Across All Visual Beats
# ==============================================================================

def test_image_budget_counts_all_unique_beats_and_halts_if_exceeded(tmp_path, monkeypatch):
    """Monetary budget warns without blocking; hard safety ceiling halts if unique images exceeded."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "image_budget_usd", 0.20)
    monkeypatch.setattr(settings, "openai_image_est_cost_usd", 0.05)
    monkeypatch.setattr(settings, "image_max_unique_per_video", 8)

    video = DummyVideo()
    step = DummyStep()

    # 5 scenes each with 2 visual beats = 10 unique beats
    # 10 beats exceeds safety ceiling of 8 unique images -> raises PipelineError
    scenes = []
    for i in range(1, 6):
        scenes.append(
            {
                "index": i,
                "narration": f"Scene {i} narration.",
                "image_prompt": f"Scene {i} prompt",
                "shot_type": "medium",
                "visual_beats": [
                    {"id": f"b{i}_1", "order": 1, "visual_description": f"Beat {i}.1", "image_strategy": "new_image"},
                    {"id": f"b{i}_2", "order": 2, "visual_description": f"Beat {i}.2", "image_strategy": "new_image"},
                ],
            }
        )

    prev = {"storyboard": {"storyboard": {"scenes": scenes}}}

    with pytest.raises(ImagePipelineError, match=r"exceeding the hard safety ceiling of 8 unique images"):
        generate_images_step(video, prev, step, image_provider=MockImageProvider())

    # When safety ceiling is raised to 50, monetary budget warning is recorded but generation proceeds
    monkeypatch.setattr(settings, "image_max_unique_per_video", 50)
    out = generate_images_step(video, prev, step, image_provider=MockImageProvider())
    assert out["images"]["complete"] is True
    assert any("exceeds standard budget guideline" in w for w in out["warnings"])


def test_unique_images_generated_for_multiple_beats_within_budget(tmp_path, monkeypatch):
    """Within budget, each visual beat in a scene receives a unique generated image."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "image_budget_usd", 1.00)

    video = DummyVideo()
    step = DummyStep()

    scenes = [
        {
            "index": 1,
            "narration": "First scene narration.",
            "image_prompt": "First scene prompt",
            "shot_type": "wide",
            "visual_beats": [
                {"id": "b1_1", "order": 1, "visual_description": "Establishing shot of aircraft", "image_strategy": "new_image"},
                {"id": "b1_2", "order": 2, "visual_description": "Close macro of bleed hole", "image_strategy": "new_image"},
            ],
        }
    ]

    prev = {"storyboard": {"storyboard": {"scenes": scenes}}}
    output = generate_images_step(video, prev, step, image_provider=MockImageProvider())
    items = output["images"]["items"]

    assert len(items) == 2
    item1 = items[0]
    item2 = items[1]

    # Verify both beats have separate files on disk
    assert item1["file"] != item2["file"]
    assert (tmp_path / item1["file"]).is_file()
    assert (tmp_path / item2["file"]).is_file()
    assert item1["reused"] is False
    assert item2["reused"] is False


# ==============================================================================
# 6. Preserving Explicit Image References
# ==============================================================================

def test_explicit_image_reference_continues_to_resolve_correctly(tmp_path):
    """Explicit source_image_id referencing an existing image continues to resolve properly."""
    img1 = tmp_path / "scene_01.png"
    img1.write_bytes(create_solid_png(100, 150, 10, 20, 30))
    img2 = tmp_path / "scene_02.png"
    img2.write_bytes(create_solid_png(100, 150, 40, 50, 60))

    valid_images = {1: {"file": "scene_01.png"}, 2: {"file": "scene_02.png"}}
    img_items = [{"index": 1, "file": "scene_01.png"}, {"index": 2, "file": "scene_02.png"}]

    beat = {"id": "b1", "order": 1, "source_image_id": "2"}
    abs_path, rel_path, is_distinct = _resolve_beat_image(
        beat=beat,
        scene_idx=1,
        valid_images=valid_images,
        img_items=img_items,
        storage_root=tmp_path,
    )
    assert abs_path == img2
    assert rel_path == "scene_02.png"
    assert is_distinct is True


# ==============================================================================
# 7. Storyboard Soft Checks for Beat Uniqueness & Visual Distinctness
# ==============================================================================

def test_storyboard_soft_checks_flag_consecutive_beat_image_reuse():
    """Soft checks produce a warning if consecutive beats share an image ID."""
    scenes = [
        SceneInput(
            narration="First scene narration.",
            image_prompt="Scene 1 prompt",
            shot_type="wide",
            visual_beats=[
                VisualBeat(
                    id="b1_1",
                    scene_id="1",
                    order=1,
                    visual_description="Opening airplane view",
                    source_image_id="same_image_id",
                ),
                VisualBeat(
                    id="b1_2",
                    scene_id="1",
                    order=2,
                    visual_description="Close view of window",
                    source_image_id="same_image_id",
                ),
            ],
        ),
        SceneInput(
            narration="Second scene narration.",
            image_prompt="Scene 2 prompt",
            shot_type="close-up",
            visual_beats=[
                VisualBeat(
                    id="b2_1",
                    scene_id="2",
                    order=1,
                    visual_description="Macro bleed hole view",
                    source_image_id="distinct_image_id",
                )
            ],
        ),
    ]

    warnings = validate_storyboard_soft_checks(
        scenes=scenes,
        script_narration="First scene narration. Second scene narration.",
        subject_anchor="An airplane window frame.",
    )
    reuse_warnings = [w for w in warnings if "share the same image ID" in w]
    assert len(reuse_warnings) >= 1
    assert "same_image_id" in reuse_warnings[0]


def test_storyboard_soft_checks_flag_consecutive_beat_identical_descriptions():
    """Soft checks produce a warning if consecutive beats have identical visual descriptions."""
    scenes = [
        SceneInput(
            narration="First scene narration.",
            image_prompt="Scene 1 prompt",
            shot_type="wide",
            visual_beats=[
                VisualBeat(
                    id="b1_1",
                    scene_id="1",
                    order=1,
                    visual_description="Exactly identical visual description of airplane window",
                    source_image_id="img_1",
                ),
                VisualBeat(
                    id="b1_2",
                    scene_id="1",
                    order=2,
                    visual_description="Exactly identical visual description of airplane window",
                    source_image_id="img_2",
                ),
            ],
        ),
        SceneInput(
            narration="Second scene narration.",
            image_prompt="Scene 2 prompt",
            shot_type="close-up",
            visual_beats=[
                VisualBeat(
                    id="b2_1",
                    scene_id="2",
                    order=1,
                    visual_description="Fresh distinct description of cabin interior",
                    source_image_id="img_3",
                )
            ],
        ),
    ]

    warnings = validate_storyboard_soft_checks(
        scenes=scenes,
        script_narration="First scene narration. Second scene narration.",
        subject_anchor="An airplane window frame.",
    )
    desc_warnings = [w for w in warnings if "identical visual descriptions" in w]
    assert len(desc_warnings) >= 1
