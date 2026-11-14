"""Tests for Phase 3.1: VisualBeat schema, storyboard planning, and backward compatibility."""

import json
import pytest
from pydantic import ValidationError

from app.steps.storyboard import (
    ALLOWED_IMAGE_STRATEGIES,
    ALLOWED_SHOT_TYPES,
    SceneInput,
    StoryboardLLMOutput,
    VisualBeat,
    normalize_scene_visual_beats,
    validate_storyboard_soft_checks,
)


def test_visual_beat_model_creation_and_validation():
    """1. VisualBeat model creation and validation."""
    beat = VisualBeat(
        id="beat_1_1",
        scene_id="1",
        order=1,
        visual_description="Wide establishing aerial view of an aircraft over sunlit clouds",
        visual_purpose="Establish subject and setting",
        shot_type="wide",
        camera_angle="high-angle aerial",
        focal_point="aircraft in cloud deck",
        visual_technique="cinematic-photo",
        image_strategy="new_image",
        source_image_id="1",
    )
    assert beat.id == "beat_1_1"
    assert beat.scene_id == "1"
    assert beat.order == 1
    assert beat.shot_type == "wide"
    assert beat.image_strategy == "new_image"
    assert beat.source_image_id == "1"

    # Validation failure: missing required visual_description
    with pytest.raises(ValidationError):
        VisualBeat(id="b1", scene_id="1", order=1)  # type: ignore

    # Validation failure: invalid shot_type
    with pytest.raises(ValidationError):
        VisualBeat(visual_description="Something", shot_type="invalid_shot")  # type: ignore

    # Validation failure: invalid image_strategy
    with pytest.raises(ValidationError):
        VisualBeat(visual_description="Something", image_strategy="unknown_strategy")  # type: ignore


def test_scene_containing_multiple_ordered_visual_beats():
    """2. A scene containing multiple ordered VisualBeats."""
    beats = [
        VisualBeat(
            id="beat_2_1",
            scene_id="2",
            order=1,
            visual_description="Server rack with blinking status lights",
            visual_purpose="Establish server room environment",
            shot_type="medium",
            camera_angle="eye-level",
            focal_point="server chassis",
            image_strategy="new_image",
            source_image_id="2",
        ),
        VisualBeat(
            id="beat_2_2",
            scene_id="2",
            order=2,
            visual_description="Close-up crop on primary processor chip",
            visual_purpose="Direct attention to high-power silicon die",
            shot_type="close-up",
            camera_angle="top-down",
            focal_point="microprocessor die",
            image_strategy="reuse_scene_image",
            source_image_id="2",
        ),
        VisualBeat(
            id="beat_2_3",
            scene_id="2",
            order=3,
            visual_description="Macro cross-section of glowing copper heat pipes",
            visual_purpose="Show intense heat and power dissipation",
            shot_type="macro",
            camera_angle="45-degree isometric",
            focal_point="copper cooling traces",
            image_strategy="reuse_scene_image",
            source_image_id="2",
        ),
    ]

    scene = SceneInput(
        narration="Because the answer takes powerful computer chips, and those chips use electricity.",
        image_prompt="Modern server motherboard with central processor",
        shot_type="medium",
        visual_beats=beats,
    )

    assert len(scene.visual_beats) == 3
    assert [b.order for b in scene.visual_beats] == [1, 2, 3]
    assert [b.image_strategy for b in scene.visual_beats] == [
        "new_image",
        "reuse_scene_image",
        "reuse_scene_image",
    ]


def test_missing_optional_beat_fields_apply_defaults():
    """3. Missing optional beat fields apply sensible defaults."""
    beat = VisualBeat(
        visual_description="A standalone engine rotor on a test mount",
    )
    assert beat.id == ""
    assert beat.scene_id == ""
    assert beat.order == 1
    assert beat.shot_type == "medium"
    assert beat.camera_angle == "eye-level"
    assert beat.focal_point == ""
    assert beat.visual_technique == "cinematic-photo"
    assert beat.image_strategy == "new_image"
    assert beat.source_image_id is None


def test_empty_or_missing_beat_lists_in_scene():
    """4. Empty or missing beat lists in SceneInput."""
    # When visual_beats is omitted completely:
    scene_omitted = SceneInput(
        narration="First sentence of the narration.",
        image_prompt="Airplane exterior",
        shot_type="aerial",
    )
    assert scene_omitted.visual_beats == []

    # When visual_beats is explicitly empty:
    scene_empty = SceneInput(
        narration="Second sentence of the narration.",
        image_prompt="Airplane interior",
        shot_type="interior",
        visual_beats=[],
    )
    assert scene_empty.visual_beats == []


def test_malformed_beat_data_and_safe_fallback_behavior():
    """5. Malformed beat data and safe fallback behavior."""
    fallback_scene = {
        "visual_purpose": "Explain turbine airflow",
        "image_prompt": "Jet turbine intake diagram",
        "shot_type": "diagram",
        "camera_angle": "cross-section",
        "focal_point": "compressor fan blades",
        "visual_technique": "technical-illustration",
    }

    # Case A: Empty list falls back to 1 synthesized beat from parent scene
    beats_empty = normalize_scene_visual_beats([], scene_index=3, fallback_scene=fallback_scene)
    assert len(beats_empty) == 1
    assert beats_empty[0].id == "beat_3_1"
    assert beats_empty[0].scene_id == "3"
    assert beats_empty[0].order == 1
    assert beats_empty[0].shot_type == "diagram"
    assert beats_empty[0].focal_point == "compressor fan blades"
    assert beats_empty[0].image_strategy == "new_image"
    assert beats_empty[0].source_image_id == "3"

    # Case B: Malformed elements (invalid shot_type, invalid strategy, empty descriptions)
    malformed_input = [
        {
            "id": "",
            "order": 99,
            "visual_description": "",
            "shot_type": "invalid_super_shot",
            "image_strategy": "invalid_strategy",
        },
        {
            "id": "beat_custom",
            "order": 1,
            "visual_description": "Custom valid description",
            "shot_type": "close-up",
            "image_strategy": "reuse_scene_image",
            "source_image_id": None,
        },
    ]

    sanitized = normalize_scene_visual_beats(malformed_input, scene_index=4, fallback_scene=fallback_scene)
    assert len(sanitized) == 2
    # First beat was sanitized
    assert sanitized[0].id == "beat_4_1"
    assert sanitized[0].order == 1
    assert sanitized[0].shot_type == "diagram"  # fell back to scene's shot_type
    assert sanitized[0].image_strategy == "new_image"  # defaulted for order 1
    assert sanitized[0].visual_description == "Explain turbine airflow"  # fallback from purpose

    # Second beat preserved custom fields with sequential order
    assert sanitized[1].id == "beat_custom"
    assert sanitized[1].order == 2
    assert sanitized[1].shot_type == "close-up"
    assert sanitized[1].image_strategy == "reuse_scene_image"
    assert sanitized[1].source_image_id == "4"  # auto-assigned scene_index for reuse_scene_image


def test_loading_and_serializing_legacy_storyboards_without_visual_beats():
    """6. Loading and serializing legacy storyboards without VisualBeats."""
    legacy_json = """
    {
        "narration": "Commercial jets rely on these tiny ports for passenger safety.",
        "visual_purpose": "Explain fuselage bleed port",
        "shot_type": "macro",
        "camera_angle": "extreme-macro",
        "focal_point": "bleed port aperture",
        "visual_technique": "macro-photo",
        "environment": "exterior aircraft hull",
        "image_prompt": "Macro view of tiny bleed hole in acrylic window",
        "uses_anchor": true
    }
    """
    raw_data = json.loads(legacy_json)
    # Must parse without validation errors
    scene = SceneInput(**raw_data)
    assert scene.narration == "Commercial jets rely on these tiny ports for passenger safety."
    assert scene.visual_beats == []

    # Normalization synthesizes default beat
    normalized = normalize_scene_visual_beats(scene.visual_beats, scene_index=1, fallback_scene=scene)
    assert len(normalized) == 1
    assert normalized[0].order == 1
    assert normalized[0].shot_type == "macro"
    assert normalized[0].image_strategy == "new_image"

    # Serialization roundtrip preserves original fields
    dumped = scene.model_dump()
    assert dumped["narration"] == raw_data["narration"]
    assert dumped["image_prompt"] == raw_data["image_prompt"]
    assert "visual_beats" in dumped


def test_preserving_existing_scene_narration_and_metadata():
    """7. Preserving existing scene narration and metadata alongside visual beats."""
    scene = SceneInput(
        narration="Aircraft windows consist of three separate acrylic panes.",
        visual_purpose="Reveal layered pane construction",
        shot_type="cutaway",
        camera_angle="isometric-cutaway",
        focal_point="separated 3-pane assembly",
        visual_technique="technical-illustration",
        environment="technical schematic space",
        image_prompt="Exploded technical cutaway showing outer, middle, and inner acrylic panes",
        uses_anchor=False,
        visual_beats=[
            VisualBeat(
                id="beat_1_1",
                scene_id="1",
                order=1,
                visual_description="Assembled window unit in fuselage frame",
                shot_type="medium",
                image_strategy="new_image",
            ),
            VisualBeat(
                id="beat_1_2",
                scene_id="1",
                order=2,
                visual_description="Exploded view separating all 3 panes",
                shot_type="cutaway",
                image_strategy="reuse_scene_image",
            ),
        ],
    )

    data = scene.model_dump()
    assert data["narration"] == "Aircraft windows consist of three separate acrylic panes."
    assert data["visual_purpose"] == "Reveal layered pane construction"
    assert data["shot_type"] == "cutaway"
    assert data["focal_point"] == "separated 3-pane assembly"
    assert data["environment"] == "technical schematic space"
    assert len(data["visual_beats"]) == 2


def test_storyboard_response_parsing_with_new_structure():
    """8. Storyboard response parsing with the new structure (StoryboardLLMOutput)."""
    llm_payload = {
        "subject_anchor": "A commercial jet passenger window with acrylic panes.",
        "environment_anchor": "Cruising altitude above sunny clouds.",
        "scenes": [
            {
                "narration": "Have a look at airplane windows.",
                "visual_purpose": "Establish airplane window",
                "shot_type": "medium",
                "image_prompt": "Airplane passenger window from inside cabin",
                "visual_beats": [
                    {
                        "id": "beat_1_1",
                        "scene_id": "1",
                        "order": 1,
                        "visual_description": "Cabin interior view looking towards window",
                        "visual_purpose": "Establish cabin setting",
                        "shot_type": "medium",
                        "image_strategy": "new_image",
                    },
                    {
                        "id": "beat_1_2",
                        "scene_id": "1",
                        "order": 2,
                        "visual_description": "Closer framing on the window aperture",
                        "visual_purpose": "Direct viewer attention to window",
                        "shot_type": "close-up",
                        "image_strategy": "reuse_scene_image",
                    },
                ],
            }
        ],
    }

    parsed = StoryboardLLMOutput(**llm_payload)
    assert parsed.subject_anchor == "A commercial jet passenger window with acrylic panes."
    assert len(parsed.scenes) == 1
    sc = parsed.scenes[0]
    assert len(sc.visual_beats) == 2
    assert sc.visual_beats[0].id == "beat_1_1"
    assert sc.visual_beats[1].id == "beat_1_2"
    assert sc.visual_beats[1].image_strategy == "reuse_scene_image"


def test_ensuring_beat_ordering_remains_stable():
    """9. Ensuring beat ordering remains stable and sequentially ordered."""
    raw_unordered = [
        {
            "id": "beat_b",
            "order": 3,
            "visual_description": "Detail B",
            "shot_type": "close-up",
            "image_strategy": "reuse_scene_image",
        },
        {
            "id": "beat_a",
            "order": 10,
            "visual_description": "Detail A",
            "shot_type": "wide",
            "image_strategy": "new_image",
        },
        {
            "id": "beat_c",
            "order": 1,
            "visual_description": "Detail C",
            "shot_type": "macro",
            "image_strategy": "reuse_scene_image",
        },
    ]

    normalized = normalize_scene_visual_beats(raw_unordered, scene_index=5)
    assert [b.order for b in normalized] == [1, 2, 3]
    assert [b.id for b in normalized] == ["beat_b", "beat_a", "beat_c"]
    assert all(b.scene_id == "5" for b in normalized)


def test_end_to_end_storyboard_step_produces_visual_beats(monkeypatch):
    """10. End-to-end generate_storyboard_step execution produces valid visual beats."""
    import uuid
    from app.config import settings
    from app.steps.storyboard import generate_storyboard_step

    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_provider_storyboard", "mock")

    class DummyVideo:
        id = uuid.uuid4()
        topic = "AI Server Chips"
        channel = None

    class DummyStep:
        name = "storyboard"
        user_prompt = None
        output = None

    script_data = {
        "title": "Why AI Questions Cost So Much",
        "hook": "Why does answering one AI question cost so much?",
        "narration": (
            "Why does answering one AI question cost so much? "
            "Because the answer takes powerful computer chips, "
            "and those chips use electricity every time they work. "
            "Inside modern data centers, rows of high-density server racks operate continuously. "
            "Each processor die channels electrical current through billions of nanoscale transistors. "
            "This massive computational workload generates intense thermal energy. "
            "Powerful industrial cooling loops must constantly pump chilled fluid across the silicon. "
            "Every single prompt triggers a complex chain reaction of electrical power and cooling infrastructure."
        ),
        "angle": "hardware economics",
    }
    prev = {"script": {"script": script_data}}

    result = generate_storyboard_step(DummyVideo(), prev, DummyStep())

    assert "storyboard" in result
    sb = result["storyboard"]
    assert "scenes" in sb
    assert "total_visual_beats" in sb
    assert sb["total_visual_beats"] >= len(sb["scenes"])

    for sc in sb["scenes"]:
        assert "visual_beats" in sc
        beats = sc["visual_beats"]
        assert len(beats) >= 1
        for b in beats:
            assert b["order"] >= 1
            assert b["visual_description"]
            assert b["shot_type"] in ALLOWED_SHOT_TYPES
            assert b["image_strategy"] in ALLOWED_IMAGE_STRATEGIES
            assert b["scene_id"] == str(sc["index"])


# ===========================================================================
# Stage 4.2 Tests: Cross-Scene reuse_previous_image Normalization
# ===========================================================================

def test_normalize_scene_visual_beats_reuse_previous_image_later_scene():
    """First beat of a later scene using reuse_previous_image does NOT infer preceding scene reference."""
    raw_beats_scene_2 = [
        {
            "id": "beat_2_1",
            "order": 1,
            "visual_description": "Continue viewing server rack from scene 1",
            "image_strategy": "reuse_previous_image",
            "source_image_id": None,
        }
    ]
    normalized_2 = normalize_scene_visual_beats(raw_beats_scene_2, scene_index=2)
    assert len(normalized_2) == 1
    assert normalized_2[0].order == 1
    assert normalized_2[0].image_strategy == "reuse_previous_image"
    assert normalized_2[0].source_image_id is None

    # Also test scene 5 does not infer scene 4
    raw_beats_scene_5 = [
        {
            "id": "beat_5_1",
            "order": 1,
            "visual_description": "Linger on cooling tower from scene 4",
            "image_strategy": "reuse_previous_image",
        }
    ]
    normalized_5 = normalize_scene_visual_beats(raw_beats_scene_5, scene_index=5)
    assert normalized_5[0].source_image_id is None


def test_normalize_scene_visual_beats_reuse_previous_image_first_scene_safe_fallback():
    """First scene using reuse_previous_image safely handles absence of previous scene (source_image_id=None)."""
    raw_beats_scene_1 = [
        {
            "id": "beat_1_1",
            "order": 1,
            "visual_description": "Opening shot trying to reuse previous",
            "image_strategy": "reuse_previous_image",
            "source_image_id": None,
        }
    ]
    normalized = normalize_scene_visual_beats(raw_beats_scene_1, scene_index=1)
    assert len(normalized) == 1
    assert normalized[0].order == 1
    assert normalized[0].image_strategy == "reuse_previous_image"
    # Preserves existing safe fallback: None allows renderer to use parent scene image
    assert normalized[0].source_image_id is None


def test_normalize_scene_visual_beats_explicit_source_image_id_unchanged():
    """Explicit source_image_id is never overwritten across any image strategy."""
    raw_beats = [
        {
            "id": "b1",
            "order": 1,
            "visual_description": "Explicit ref for reuse_previous_image",
            "image_strategy": "reuse_previous_image",
            "source_image_id": "42",
        },
        {
            "id": "b2",
            "order": 2,
            "visual_description": "Explicit ref for reuse_scene_image",
            "image_strategy": "reuse_scene_image",
            "source_image_id": "custom_image_ref",
        },
        {
            "id": "b3",
            "order": 3,
            "visual_description": "Explicit ref for new_image",
            "image_strategy": "new_image",
            "source_image_id": "special_asset_id",
        },
    ]
    normalized = normalize_scene_visual_beats(raw_beats, scene_index=3)
    assert len(normalized) == 3
    assert normalized[0].source_image_id == "42"
    assert normalized[1].source_image_id == "custom_image_ref"
    assert normalized[2].source_image_id == "special_asset_id"


def test_normalize_scene_visual_beats_reuse_scene_image_unchanged():
    """reuse_scene_image continues to default to parent scene index."""
    raw_beats = [
        {
            "id": "b1",
            "order": 1,
            "visual_description": "Crop into current scene",
            "image_strategy": "reuse_scene_image",
            "source_image_id": None,
        }
    ]
    normalized = normalize_scene_visual_beats(raw_beats, scene_index=4)
    assert len(normalized) == 1
    assert normalized[0].image_strategy == "reuse_scene_image"
    assert normalized[0].source_image_id == "4"


def test_normalize_scene_visual_beats_new_image_unchanged():
    """new_image strategy continues to leave source_image_id as None if unspecified."""
    raw_beats = [
        {
            "id": "b1",
            "order": 1,
            "visual_description": "Brand new subject establishing shot",
            "image_strategy": "new_image",
            "source_image_id": None,
        }
    ]
    normalized = normalize_scene_visual_beats(raw_beats, scene_index=2)
    assert len(normalized) == 1
    assert normalized[0].image_strategy == "new_image"
    assert normalized[0].source_image_id is None


def test_normalize_scene_visual_beats_subsequent_beat_reuse_previous_image():
    """Subsequent beat in a scene does not automatically infer previous beat reference without explicit ID."""
    # Case A: Beat 1 and Beat 2 use reuse_previous_image with None source_image_id
    raw_beats_a = [
        {
            "id": "b2_1",
            "visual_description": "Linger on scene 1",
            "image_strategy": "reuse_previous_image",
            "source_image_id": None,
        },
        {
            "id": "b2_2",
            "visual_description": "Continue lingering on scene 1",
            "image_strategy": "reuse_previous_image",
            "source_image_id": None,
        },
    ]
    normalized_a = normalize_scene_visual_beats(raw_beats_a, scene_index=2)
    assert normalized_a[0].source_image_id is None
    assert normalized_a[1].source_image_id is None

    # Case B: Beat 1 is new_image (scene 3 image), Beat 2 reuses previous image with None -> remains None
    raw_beats_b = [
        {
            "id": "b3_1",
            "visual_description": "New image for scene 3",
            "image_strategy": "new_image",
        },
        {
            "id": "b3_2",
            "visual_description": "Reuse previous beat image in scene 3",
            "image_strategy": "reuse_previous_image",
        },
    ]
    normalized_b = normalize_scene_visual_beats(raw_beats_b, scene_index=3)
    assert normalized_b[0].source_image_id is None
    assert normalized_b[1].source_image_id is None


def test_inferred_reference_matches_renderer_asset_resolution(tmp_path):
    """Explicit cross-scene reference seamlessly resolves via renderer's _resolve_beat_image contract."""
    from app.images.mock import create_solid_png
    from app.steps.render import _resolve_beat_image

    img1 = tmp_path / "scene_01.png"
    img1.write_bytes(create_solid_png(100, 150, 10, 20, 30))
    img2 = tmp_path / "scene_02.png"
    img2.write_bytes(create_solid_png(100, 150, 40, 50, 60))

    valid_images = {1: {"file": "scene_01.png"}, 2: {"file": "scene_02.png"}}
    img_items = [{"index": 1, "file": "scene_01.png"}, {"index": 2, "file": "scene_02.png"}]

    # 1. Scene 2 with explicit reuse_previous_image -> resolves to scene_01.png as distinct
    normalized_sc2 = normalize_scene_visual_beats(
        [{"visual_description": "Cross-scene lingering", "image_strategy": "reuse_previous_image", "source_image_id": "1"}],
        scene_index=2,
    )
    beat_dict = normalized_sc2[0].model_dump()
    assert beat_dict["source_image_id"] == "1"

    abs_path, rel_path, is_distinct = _resolve_beat_image(
        beat=beat_dict,
        scene_idx=2,
        valid_images=valid_images,
        img_items=img_items,
        storage_root=tmp_path,
    )
    assert abs_path == img1
    assert rel_path == "scene_01.png"
    assert is_distinct is True

    # 2. Scene 1 with reuse_previous_image (no prior scene) -> safely resolves to parent scene_01.png
    normalized_sc1 = normalize_scene_visual_beats(
        [{"visual_description": "Opening beat", "image_strategy": "reuse_previous_image"}],
        scene_index=1,
    )
    beat_dict_1 = normalized_sc1[0].model_dump()
    assert beat_dict_1["source_image_id"] is None

    abs_path_1, rel_path_1, is_distinct_1 = _resolve_beat_image(
        beat=beat_dict_1,
        scene_idx=1,
        valid_images=valid_images,
        img_items=img_items,
        storage_root=tmp_path,
    )
    assert abs_path_1 == img1
    assert rel_path_1 == "scene_01.png"
    assert is_distinct_1 is False


