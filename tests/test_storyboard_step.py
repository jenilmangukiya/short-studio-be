"""Unit tests for the real storyboard step (network-free)."""

import json
import uuid
import pytest

from app.config import settings
from app.pipeline import h_images, h_voiceover
from app.steps.prompt_compose import (
    COMPOSITION_RULES,
    MAX_FINAL_PROMPT_LENGTH,
    compose_final_prompt,
    recompute_final_prompts,
)
from app.steps.storyboard import (
    ALLOWED_SHOT_TYPES,
    DEFAULT_MOCK_SUBJECT_ANCHOR,
    FORBIDDEN_PROMPT_WORDS,
    FORBIDDEN_WORDS_REGEX,
    PipelineError,
    SceneInput,
    StoryboardLLMOutput,
    WORDS_PER_SECOND,
    build_storyboard_prompt,
    find_first_difference,
    generate_storyboard_step,
    load_default_visual_style,
    load_prompt_template,
    normalize_for_comparison,
    split_script_for_mock,
    validate_storyboard_soft_checks,
)


def _make_sample_script_dict(
    hook="Have a look at airplane windows.",
    narration=None,
) -> dict:
    """Helper to build a valid script output dictionary."""
    if narration is None:
        narration = (
            f"{hook} Air travelers often notice a tiny hole in the passenger window. "
            "This small bleed hole balances the immense pressure difference between the pressurized cabin "
            "and the thin outside atmosphere. Aircraft windows consist of three separate acrylic panes. "
            "The outer pane bears the primary pressure load, while the inner pane prevents passengers "
            "from scratching the crucial middle layer. The bleed hole allows moisture to escape, preventing "
            "frost from obstructing the view. Without this simple engineering detail, windows could fail "
            "or fog completely. Aircraft designers solved a massive cabin pressure challenge with a miniature opening. "
            "Every commercial jet flying at high altitude relies on these tiny ports for passenger safety and "
            "clear visibility throughout the entire flight."
        )

    return {
        "title": "Why Airplane Windows Have Tiny Holes",
        "hook": hook,
        "narration": narration,
        "angle": "why it's designed this way",
        "claims_to_verify": [
            "Airplane windows have three acrylic panes.",
            "The bleed hole balances cabin pressure.",
        ],
        "hashtags": ["#aviation", "#engineering", "#facts"],
    }


def _make_valid_scenes(script_narration: str, count: int = 8) -> list[SceneInput]:
    """Helper to generate `count` valid scenes splitting the script narration."""
    words = script_narration.split()
    chunk_size = len(words) // count
    scenes: list[SceneInput] = []

    for i in range(count):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < count - 1 else len(words)
        chunk_words = words[start:end]
        shot_type = ALLOWED_SHOT_TYPES[i % len(ALLOWED_SHOT_TYPES)]
        scenes.append(
            SceneInput(
                narration=" ".join(chunk_words),
                image_prompt=f"A mechanical component for scene {i+1} illustrating operation",
                shot_type=shot_type,
                uses_anchor=(i + 1) not in (2, 5),
            )
        )
    return scenes


# ==============================================================================
# 1. Soft-Check Validator Tests
# ==============================================================================

def test_validator_passes_on_compliant_storyboard():
    script = _make_sample_script_dict()
    scenes = _make_valid_scenes(script["narration"], count=9)
    warnings = validate_storyboard_soft_checks(
        scenes,
        script["narration"],
        subject_anchor="A small glossy blue cube with rounded edges.",
        visual_style="Clean 3D render, soft studio lighting",
    )
    assert warnings == []


def test_validator_detects_too_few_scenes():
    script = _make_sample_script_dict()
    scenes = _make_valid_scenes(script["narration"], count=5)
    warnings = validate_storyboard_soft_checks(scenes, script["narration"])
    assert any("scene count is 5" in w.lower() for w in warnings)


def test_validator_detects_too_many_scenes():
    script = _make_sample_script_dict()
    scenes = _make_valid_scenes(script["narration"], count=14)
    warnings = validate_storyboard_soft_checks(scenes, script["narration"])
    assert any("scene count is 14" in w.lower() for w in warnings)


def test_validator_detects_scene_duration_out_of_bounds():
    script = _make_sample_script_dict()
    scenes = _make_valid_scenes(script["narration"], count=8)
    # Scene with 2 words -> 0.8s (too short)
    scenes[0].narration = "Short text."
    warnings = validate_storyboard_soft_checks(scenes, script["narration"])
    assert any("duration is 0.8s" in w.lower() for w in warnings)


def test_validator_detects_empty_image_prompt():
    script = _make_sample_script_dict()
    scenes = _make_valid_scenes(script["narration"], count=8)
    scenes[1].image_prompt = "   "
    warnings = validate_storyboard_soft_checks(scenes, script["narration"])
    assert any("scene 2 image prompt is empty" in w.lower() for w in warnings)


def test_validator_detects_image_prompt_too_long():
    script = _make_sample_script_dict()
    scenes = _make_valid_scenes(script["narration"], count=8)
    scenes[2].image_prompt = "a" * 305
    warnings = validate_storyboard_soft_checks(scenes, script["narration"])
    assert any("at most 300 chars" in w.lower() for w in warnings)


def test_validator_detects_forbidden_words_whole_word_only():
    # 1. Exact forbidden word triggers warning
    script = _make_sample_script_dict()
    scenes = _make_valid_scenes(script["narration"], count=8)
    scenes[0].image_prompt = "Close-up of window with text visible on surface"
    warnings = validate_storyboard_soft_checks(scenes, script["narration"])
    assert any("contains forbidden whole words" in w.lower() and "text" in w.lower() for w in warnings)

    # 2. Similar words that contain forbidden substring ("textured", "context", "texture") MUST PASS
    scenes[0].image_prompt = "Close-up of textured metal surface in realistic context with smooth texture"
    warnings_clean = validate_storyboard_soft_checks(scenes, script["narration"])
    assert not any("contains forbidden whole words" in w.lower() for w in warnings_clean)


def test_validator_detects_consecutive_identical_shot_types():
    script = _make_sample_script_dict()
    scenes = _make_valid_scenes(script["narration"], count=8)
    scenes[1].shot_type = scenes[0].shot_type  # Consecutive duplicates
    warnings = validate_storyboard_soft_checks(scenes, script["narration"])
    assert any("identical shot type" in w.lower() for w in warnings)


def test_validator_normalizes_quotes_and_whitespace_for_narration_match():
    # Script narration has straight quotes and hyphen
    narration_straight = "Don't ignore the window - it balances cabin pressure."
    script = {"narration": narration_straight}

    # Scenes use curly quotes and em-dash
    scenes = [
        SceneInput(
            narration="Don’t ignore the window—",
            image_prompt="A passenger window",
            shot_type="close-up",
            uses_anchor=True,
        ),
        SceneInput(
            narration="it balances cabin pressure.",
            image_prompt="Pressure bleed hole",
            shot_type="macro",
            uses_anchor=False,
        ),
    ]
    warnings = validate_storyboard_soft_checks(scenes, narration_straight)
    # Should warn about scene count (2 < 8), but NOT about narration mismatch
    assert not any("do not reproduce the script narration" in w.lower() for w in warnings)


def test_validator_reports_first_difference_on_narration_mismatch():
    script_narration = "The outer pane bears the primary pressure load."
    scenes = [
        SceneInput(
            narration="The outer pane carries the primary pressure load.",
            image_prompt="Acrylic window pane",
            shot_type="close-up",
            uses_anchor=True,
        )
    ]
    warnings = validate_storyboard_soft_checks(scenes, script_narration)
    mismatch_warning = next(w for w in warnings if "do not reproduce" in w.lower())
    assert "Divergence at position" in mismatch_warning
    assert "bears" in mismatch_warning or "carries" in mismatch_warning


# ==============================================================================
# 2. Timing and Duration Computation Tests
# ==============================================================================

def test_est_seconds_computation():
    words_5 = "One two three four five."
    assert round(len(words_5.split()) / WORDS_PER_SECOND, 1) == 2.0

    words_11 = "One two three four five six seven eight nine ten eleven."
    assert round(len(words_11.split()) / WORDS_PER_SECOND, 1) == 4.4


# ==============================================================================
# 3. Visual Style Fallback, Source, and Framing Independence Tests
# ==============================================================================

def test_visual_style_channel_setting(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_provider_storyboard", "mock")

    class DummyChannel:
        settings = {"visual_style": "Vintage claymation, warm tungsten lighting"}

    class DummyVideo:
        id = uuid.uuid4()
        topic = "Airplane windows"
        channel = DummyChannel()

    class DummyStep:
        name = "storyboard"
        user_prompt = None
        output = None

    prev = {"script": {"script": _make_sample_script_dict()}}
    result = generate_storyboard_step(DummyVideo(), prev, DummyStep())

    assert result["storyboard"]["visual_style"] == "Vintage claymation, warm tungsten lighting"
    assert result["storyboard"]["style_source"] == "channel"


def test_visual_style_default_fallback(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_provider_storyboard", "mock")

    class DummyVideo:
        id = uuid.uuid4()
        topic = "Airplane windows"
        channel = None

    class DummyStep:
        name = "storyboard"
        user_prompt = None
        output = None

    prev = {"script": {"script": _make_sample_script_dict()}}
    result = generate_storyboard_step(DummyVideo(), prev, DummyStep())

    default_style = load_default_visual_style()
    assert result["storyboard"]["visual_style"] == default_style
    assert result["storyboard"]["style_source"] == "default"


def test_default_visual_style_does_not_contain_framing_or_aspect_ratio():
    """Verify that default_visual_style owns only the aesthetic and no 9:16 or hero object framing."""
    style_text = load_default_visual_style()
    assert "9:16" not in style_text
    assert "single hero object" not in style_text


# ==============================================================================
# 4. Prompt Assembly Tests (Fresh vs Revision Mode)
# ==============================================================================

def test_build_storyboard_prompt_fresh():
    rules_text, prompt_version = load_prompt_template()
    script = _make_sample_script_dict()
    prompt = build_storyboard_prompt(
        rules_text=rules_text,
        topic="Airplane windows",
        script_data=script,
        visual_style="Clean 3D render",
    )
    assert "## Video Script to Storyboard" in prompt
    assert script["narration"] in prompt
    assert "Target Visual Style" in prompt
    assert "REVISION MODE" not in prompt


def test_build_storyboard_prompt_revision_mode():
    rules_text, prompt_version = load_prompt_template()
    script = _make_sample_script_dict()
    prev_sb = {
        "visual_style": "Clean 3D render",
        "subject_anchor": "A titanium window frame.",
        "scenes": [
            {
                "index": 1,
                "image_prompt": "Old prompt",
                "narration": "Some narration",
                "uses_anchor": True,
            }
        ],
    }
    instruction = "Change scene 1 framing to top-down"

    prompt = build_storyboard_prompt(
        rules_text=rules_text,
        topic="Airplane windows",
        script_data=script,
        visual_style="Clean 3D render",
        previous_storyboard=prev_sb,
        revision_instruction=instruction,
    )
    assert "## REVISION MODE" in prompt
    assert instruction in prompt
    assert "Never alter the narration text" in prompt
    assert "Keep subject_anchor and every uses_anchor flag unchanged" in prompt
    assert "Old prompt" in prompt


# ==============================================================================
# 5. Deterministic Mock Splitter Tests
# ==============================================================================

def test_mock_splitter_standard_script():
    script = _make_sample_script_dict()
    scenes, anchor = split_script_for_mock(script["narration"])

    # 1. 8 to 12 scenes produced
    assert 8 <= len(scenes) <= 12

    # 2. Anchor is deterministic placeholder
    assert anchor == DEFAULT_MOCK_SUBJECT_ANCHOR

    # 3. uses_anchor by scene index membership (false for 2 and 5, true for others)
    assert scenes[0].uses_anchor is True   # index 1
    assert scenes[1].uses_anchor is False  # index 2
    assert scenes[2].uses_anchor is True   # index 3
    assert scenes[3].uses_anchor is True   # index 4
    assert scenes[4].uses_anchor is False  # index 5
    if len(scenes) > 5:
        assert scenes[5].uses_anchor is True

    # 4. Re-joining scenes reproduces script narration exactly
    joined = " ".join(s.narration for s in scenes)
    assert normalize_for_comparison(joined) == normalize_for_comparison(script["narration"])

    # 5. Rotating shot types (no neighbouring duplicates)
    for i in range(1, len(scenes)):
        assert scenes[i].shot_type != scenes[i - 1].shot_type

    # 6. Image prompts are non-empty and have no forbidden whole words
    for s in scenes:
        assert s.image_prompt
        assert not FORBIDDEN_WORDS_REGEX.findall(s.image_prompt)


def test_mock_splitter_short_script_does_not_crash():
    # Very short script (~10 words) yielding 2 scenes
    short_text = "Airplane windows have a hole. It balances cabin pressure."
    scenes, anchor = split_script_for_mock(short_text)

    assert len(scenes) == 2
    assert anchor == DEFAULT_MOCK_SUBJECT_ANCHOR
    assert scenes[0].uses_anchor is True   # index 1
    assert scenes[1].uses_anchor is False  # index 2
    joined = " ".join(s.narration for s in scenes)
    assert normalize_for_comparison(joined) == normalize_for_comparison(short_text)


def test_mock_splitter_empty_input():
    assert split_script_for_mock("") == ([], "")
    assert split_script_for_mock("   \n\t  ") == ([], "")


# ==============================================================================
# 6. Full Pipeline Step Execution with Mock Provider
# ==============================================================================

def test_full_storyboard_step_mock(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_provider_storyboard", "mock")

    class DummyVideo:
        id = uuid.uuid4()
        topic = "Airplane windows"
        channel = None

    class DummyStep:
        name = "storyboard"
        user_prompt = None
        output = None

    script_data = _make_sample_script_dict()
    prev = {"script": {"script": script_data}}

    result = generate_storyboard_step(DummyVideo(), prev, DummyStep())

    assert "storyboard" in result
    assert "warnings" in result
    assert "meta" in result

    sb = result["storyboard"]
    assert "scenes" in sb
    assert "total_seconds" in sb
    assert sb["style_source"] == "default"
    assert sb["subject_anchor"] == DEFAULT_MOCK_SUBJECT_ANCHOR
    assert len(sb["scenes"]) >= 8

    # Each scene has index, est_seconds, uses_anchor, and final_prompt
    for idx, s in enumerate(sb["scenes"], start=1):
        assert s["index"] == idx
        assert s["est_seconds"] > 0
        assert s["shot_type"] in ALLOWED_SHOT_TYPES
        assert "uses_anchor" in s
        assert "final_prompt" in s
        assert s["final_prompt"].startswith(s["image_prompt"])
        assert COMPOSITION_RULES in s["final_prompt"]

        # Check anchor presence aligns with uses_anchor flag
        if s["uses_anchor"]:
            assert DEFAULT_MOCK_SUBJECT_ANCHOR in s["final_prompt"]
        else:
            assert DEFAULT_MOCK_SUBJECT_ANCHOR not in s["final_prompt"]

    meta = result["meta"]
    assert meta["provider"] == "mock"
    assert meta["task"] == "storyboard"
    assert meta["revision"] is False
    assert "storyboard-v2:" in meta["prompt_version"]


def test_full_storyboard_step_revision_mode(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_provider_storyboard", "mock")

    class DummyVideo:
        id = uuid.uuid4()
        topic = "Airplane windows"
        channel = None

    class DummyStep:
        name = "storyboard"
        user_prompt = "Change scene 1 shot type to macro"
        output = {
            "storyboard": {
                "visual_style": "Clean 3D render",
                "subject_anchor": "A custom titanium frame.",
                "scenes": [
                    {
                        "index": 1,
                        "narration": "Have a look at airplane windows.",
                        "image_prompt": "Airplane window",
                        "shot_type": "close-up",
                        "uses_anchor": True,
                        "final_prompt": "Old final prompt",
                        "est_seconds": 2.4,
                    }
                ],
                "total_seconds": 2.4,
            }
        }

    script_data = _make_sample_script_dict()
    prev = {"script": {"script": script_data}}

    result = generate_storyboard_step(DummyVideo(), prev, DummyStep())
    assert result["meta"]["revision"] is True
    # Final prompts recomputed cleanly
    for s in result["storyboard"]["scenes"]:
        assert COMPOSITION_RULES in s["final_prompt"]


def test_missing_script_error():
    class DummyVideo:
        id = uuid.uuid4()
        topic = "Airplane windows"
        channel = None

    class DummyStep:
        name = "storyboard"
        user_prompt = None
        output = None

    with pytest.raises(PipelineError, match="Script step must be completed"):
        generate_storyboard_step(DummyVideo(), {}, DummyStep())

    with pytest.raises(PipelineError, match="Script step must be completed"):
        generate_storyboard_step(DummyVideo(), {"script": {}}, DummyStep())

    with pytest.raises(PipelineError, match="Script step must be completed"):
        generate_storyboard_step(DummyVideo(), {"script": {"script": {}}}, DummyStep())


# ==============================================================================
# 7. Pipeline Downstream Mock Handlers Integration Test
# ==============================================================================

def test_pipeline_downstream_handlers_support_nested_storyboard(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    class DummyVideo:
        id = uuid.uuid4()
        topic = "Test video"

    prev = {
        "storyboard": {
            "storyboard": {
                "visual_style": "Clean 3D render",
                "style_source": "default",
                "subject_anchor": "A small glossy blue cube.",
                "scenes": [
                    {
                        "index": 1,
                        "narration": "First scene narration.",
                        "image_prompt": "First scene prompt",
                        "shot_type": "close-up",
                        "uses_anchor": True,
                        "final_prompt": "First scene final prompt",
                        "est_seconds": 2.0,
                    },
                    {
                        "index": 2,
                        "narration": "Second scene narration.",
                        "image_prompt": "Second scene prompt",
                        "shot_type": "macro",
                        "uses_anchor": False,
                        "final_prompt": "Second scene final prompt",
                        "est_seconds": 3.0,
                    },
                ],
                "total_seconds": 5.0,
            }
        }
    }

    # h_images extracts scenes and uses final_prompt
    img_result = h_images(DummyVideo(), prev, None)
    assert img_result["mock"] is True
    assert len(img_result["images"]) == 2
    assert img_result["images"][0]["scene"] == 1

    # Check written file contents
    img_file = tmp_path / str(DummyVideo.id) / "images" / "scene_1.txt"
    assert "First scene final prompt" in img_file.read_text()

    # h_voiceover extracts total_seconds and writes voice.txt
    vo_result = h_voiceover(DummyVideo(), prev, None)
    assert vo_result["mock"] is True
    assert vo_result["duration_seconds"] == 5.0


def test_backward_compatibility_old_shape_storyboard(tmp_path, monkeypatch):
    """Ensure older storyboards without subject_anchor, uses_anchor, or final_prompt don't crash downstream."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))

    class DummyVideo:
        id = uuid.uuid4()
        topic = "Legacy video"

    old_sb_output = {
        "storyboard": {
            "visual_style": "Clean 3D render",
            "style_source": "default",
            # No subject_anchor
            "scenes": [
                {
                    "index": 1,
                    "narration": "Legacy scene narration.",
                    "image_prompt": "Legacy image prompt",
                    "shot_type": "close-up",
                    "est_seconds": 2.0,
                    # No uses_anchor, no final_prompt
                }
            ],
            "total_seconds": 2.0,
        }
    }

    # Downstream images mock should fall back to image_prompt without crashing
    img_result = h_images(DummyVideo(), prev={"storyboard": old_sb_output}, step=None)
    assert img_result["mock"] is True
    img_file = tmp_path / str(DummyVideo.id) / "images" / "scene_1.txt"
    assert "Legacy image prompt" in img_file.read_text()

    # Recomputing final prompts on legacy storyboard works gracefully
    updated = recompute_final_prompts(old_sb_output)
    scene_1 = updated["storyboard"]["scenes"][0]
    assert "final_prompt" in scene_1
    assert "Legacy image prompt" in scene_1["final_prompt"]
    assert COMPOSITION_RULES in scene_1["final_prompt"]


# ==============================================================================
# 8. Prompt Composition Unit Tests
# ==============================================================================

def test_compose_final_prompt_with_and_without_anchor():
    anchor = "A sleek titanium window frame with three panes."
    style = "Clean 3D render, soft studio lighting"

    # 1. uses_anchor is True
    scene_with = {
        "image_prompt": "Close-up of airplane passenger window bleed hole",
        "uses_anchor": True,
    }
    p_with = compose_final_prompt(scene_with, subject_anchor=anchor, visual_style=style)
    assert "Close-up of airplane passenger window bleed hole." in p_with
    assert anchor in p_with
    assert style in p_with
    assert COMPOSITION_RULES in p_with

    # 2. uses_anchor is False
    scene_without = {
        "image_prompt": "High altitude clouds floating in the sky",
        "uses_anchor": False,
    }
    p_without = compose_final_prompt(scene_without, subject_anchor=anchor, visual_style=style)
    assert "High altitude clouds floating in the sky." in p_without
    assert anchor not in p_without
    assert style in p_without
    assert COMPOSITION_RULES in p_without


def test_compose_final_prompt_clean_punctuation_joining():
    # Prompt with messy punctuation: trailing dots, trailing comma in style
    scene = {
        "image_prompt": "A macro shot of acrylic pane..",
        "uses_anchor": True,
    }
    anchor = "A small blue cube. "
    style = "Matte 3D materials, studio light, "

    prompt = compose_final_prompt(scene, subject_anchor=anchor, visual_style=style)
    # No duplicate periods
    assert ".." not in prompt
    assert ",." not in prompt
    assert "  " not in prompt
    assert prompt.endswith(".")


def test_recompute_final_prompts_updates_in_place():
    sb = {
        "visual_style": "Minimalist clay",
        "subject_anchor": "A blue rubber ball.",
        "scenes": [
            {
                "index": 1,
                "image_prompt": "Ball bouncing on wooden floor",
                "uses_anchor": True,
                "final_prompt": "Stale prompt",
            },
            {
                "index": 2,
                "image_prompt": "Empty wooden room",
                "uses_anchor": False,
                "final_prompt": "Stale prompt",
            },
        ],
    }

    recompute_final_prompts(sb)
    assert "A blue rubber ball" in sb["scenes"][0]["final_prompt"]
    assert "A blue rubber ball" not in sb["scenes"][1]["final_prompt"]
    assert "Stale prompt" not in sb["scenes"][0]["final_prompt"]


# ==============================================================================
# 9. New Soft Check Validation Tests (Subject Anchor, Uses Anchor, Final Prompt)
# ==============================================================================

def test_validator_detects_empty_subject_anchor():
    script = _make_sample_script_dict()
    scenes = _make_valid_scenes(script["narration"], count=8)
    warnings = validate_storyboard_soft_checks(scenes, script["narration"], subject_anchor="")
    assert any("subject anchor is empty" in w.lower() for w in warnings)


def test_validator_detects_subject_anchor_too_long():
    script = _make_sample_script_dict()
    scenes = _make_valid_scenes(script["narration"], count=8)
    long_anchor = "A" * 255
    warnings = validate_storyboard_soft_checks(scenes, script["narration"], subject_anchor=long_anchor)
    assert any("subject anchor length is 255 chars" in w.lower() for w in warnings)


def test_validator_lenient_sentence_count():
    script = _make_sample_script_dict()
    scenes = _make_valid_scenes(script["narration"], count=8)

    # 1. Decimals (2.5) and abbreviations (approx.) should NOT count as extra sentences
    lenient_anchor = "A 2.5cm cube with approx. 3.0mm rounded corners. It has a glossy blue tint."
    warnings = validate_storyboard_soft_checks(scenes, script["narration"], subject_anchor=lenient_anchor)
    assert not any("sentences (target: at most 2 sentences)" in w.lower() for w in warnings)

    # 2. Three real sentences SHOULD trigger a warning
    three_sentences = "This is sentence one. This is sentence two. This is sentence three."
    warnings_3 = validate_storyboard_soft_checks(scenes, script["narration"], subject_anchor=three_sentences)
    assert any("has 3 sentences (target: at most 2 sentences)" in w.lower() for w in warnings_3)


def test_validator_detects_subject_anchor_forbidden_words():
    script = _make_sample_script_dict()
    scenes = _make_valid_scenes(script["narration"], count=8)
    anchor_with_logo = "A small metallic cube with a company logo embossed on top."
    warnings = validate_storyboard_soft_checks(scenes, script["narration"], subject_anchor=anchor_with_logo)
    assert any("subject anchor contains forbidden whole words" in w.lower() and "logo" in w.lower() for w in warnings)


def test_validator_detects_no_scene_uses_anchor():
    script = _make_sample_script_dict()
    scenes = _make_valid_scenes(script["narration"], count=8)
    for s in scenes:
        s.uses_anchor = False
    warnings = validate_storyboard_soft_checks(scenes, script["narration"], subject_anchor="A small blue cube.")
    assert any("no scenes have uses_anchor set to true" in w.lower() for w in warnings)


def test_validator_composed_prompt_with_rules_does_not_trigger_forbidden_words():
    """Ensure that composition rules ('no text, no letters, no logos') do NOT trigger forbidden-word warning on final_prompt."""
    script = _make_sample_script_dict()
    scenes = _make_valid_scenes(script["narration"], count=8)
    anchor = "A small glossy blue cube with rounded edges."
    style = "Clean 3D render, soft studio lighting"

    warnings = validate_storyboard_soft_checks(
        scenes,
        script["narration"],
        subject_anchor=anchor,
        visual_style=style,
    )
    # Must NOT warn about forbidden words from COMPOSITION_RULES
    assert not any("forbidden whole words" in w.lower() for w in warnings)


def test_validator_detects_final_prompt_too_long():
    script = _make_sample_script_dict()
    scenes = _make_valid_scenes(script["narration"], count=8)
    # Construct a scene whose composed final_prompt exceeds MAX_FINAL_PROMPT_LENGTH (1000)
    scenes[0].image_prompt = "A" * 600
    long_anchor = "B" * 240
    long_style = "C" * 200

    warnings = validate_storyboard_soft_checks(
        scenes,
        script["narration"],
        subject_anchor=long_anchor,
        visual_style=long_style,
    )
    assert any("final prompt length is" in w.lower() and "target: at most 1000 chars" in w.lower() for w in warnings)


# ==============================================================================
# 10. Edge Cases and Retry Flow Tests
# ==============================================================================

def test_validator_detects_zero_scenes():
    warnings = validate_storyboard_soft_checks([], "Some narration")
    assert warnings == ["Storyboard contains zero scenes."]


def test_find_first_difference_edge_branches():
    assert find_first_difference("same text", "same text") == "No difference detected."
    early = find_first_difference("abc", "abcdefgh")
    assert "ends early" in early
    extra = find_first_difference("abcdefgh", "abc")
    assert "extra text" in extra


def test_storyboard_step_retry_on_warning(monkeypatch):
    """Test that a non-mock LLM triggers retry once when checks fail, updating attempts to 2."""
    from app.llm.base import BaseLLM

    call_count = 0

    class FakeLLM(BaseLLM):
        @property
        def provider_name(self) -> str:
            return "fake-provider"

        def _execute_text(self, prompt, system_prompt):
            return "dummy", 10, 10, None, "stop"

        def _execute_structured(self, prompt, response_model, system_prompt):
            nonlocal call_count
            call_count += 1
            script = _make_sample_script_dict()
            if call_count == 1:
                # Attempt 1: 5 scenes (will fail scene count check)
                scenes = _make_valid_scenes(script["narration"], count=5)
            else:
                # Attempt 2: 8 scenes (compliant)
                scenes = _make_valid_scenes(script["narration"], count=8)

            return (
                StoryboardLLMOutput(
                    subject_anchor="A small glossy blue cube with rounded edges.",
                    scenes=scenes,
                ),
                100,
                100,
                10,
                "stop",
            )

    import app.steps.storyboard as sb_module
    monkeypatch.setattr(sb_module, "get_llm", lambda task: FakeLLM(model="fake-model"))

    class DummyVideo:
        id = uuid.uuid4()
        topic = "Test retry"
        channel = None

    class DummyStep:
        name = "storyboard"
        user_prompt = None
        output = None

    script_data = _make_sample_script_dict()
    prev = {"script": {"script": script_data}}

    result = generate_storyboard_step(DummyVideo(), prev, DummyStep())

    assert call_count == 2
    assert result["meta"]["attempts"] == 2
    assert result["warnings"] == []
    assert len(result["storyboard"]["scenes"]) == 8


def test_revision_mode_runs_narration_checks_and_retries(monkeypatch):
    """Test that revision mode also executes narration validation and the retry path."""
    from app.llm.base import BaseLLM

    call_count = 0

    class FakeLLM(BaseLLM):
        @property
        def provider_name(self) -> str:
            return "fake-provider"

        def _execute_text(self, prompt, system_prompt):
            return "dummy", 10, 10, None, "stop"

        def _execute_structured(self, prompt, response_model, system_prompt):
            nonlocal call_count
            call_count += 1
            script = _make_sample_script_dict()
            if call_count == 1:
                # Attempt 1: revised scenes accidentally altered narration
                scenes = _make_valid_scenes(script["narration"], count=8)
                scenes[0].narration = "Altered narration text here."
            else:
                # Attempt 2: fixed to match original narration
                scenes = _make_valid_scenes(script["narration"], count=8)

            return (
                StoryboardLLMOutput(
                    subject_anchor="A small glossy blue cube with rounded edges.",
                    scenes=scenes,
                ),
                100,
                100,
                None,
                "stop",
            )

    import app.steps.storyboard as sb_module
    monkeypatch.setattr(sb_module, "get_llm", lambda task: FakeLLM(model="fake-model"))

    class DummyVideo:
        id = uuid.uuid4()
        topic = "Test revision retry"
        channel = None

    class DummyStep:
        name = "storyboard"
        user_prompt = "Change visual style to macro"
        output = {
            "storyboard": {
                "visual_style": "Clean 3D render",
                "subject_anchor": "A small glossy blue cube.",
                "scenes": [],
                "total_seconds": 10.0,
            }
        }

    script_data = _make_sample_script_dict()
    prev = {"script": {"script": script_data}}

    result = generate_storyboard_step(DummyVideo(), prev, DummyStep())

    assert call_count == 2
    assert result["meta"]["attempts"] == 2
    assert result["meta"]["revision"] is True
    assert result["warnings"] == []
