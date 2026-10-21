"""Unit tests for the real storyboard step (network-free)."""

import json
import uuid
import pytest

from app.config import settings
from app.pipeline import h_images, h_voiceover
from app.steps.storyboard import (
    ALLOWED_SHOT_TYPES,
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
            )
        )
    return scenes


# ==============================================================================
# 1. Soft-Check Validator Tests
# ==============================================================================

def test_validator_passes_on_compliant_storyboard():
    script = _make_sample_script_dict()
    scenes = _make_valid_scenes(script["narration"], count=9)
    warnings = validate_storyboard_soft_checks(scenes, script["narration"])
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
        ),
        SceneInput(
            narration="it balances cabin pressure.",
            image_prompt="Pressure bleed hole",
            shot_type="macro",
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
    # WORDS_PER_SECOND = 2.5
    # 5 words -> 5 / 2.5 = 2.0s
    # 10 words -> 10 / 2.5 = 4.0s
    # 11 words -> 11 / 2.5 = 4.4s
    words_5 = "One two three four five."
    assert round(len(words_5.split()) / WORDS_PER_SECOND, 1) == 2.0

    words_11 = "One two three four five six seven eight nine ten eleven."
    assert round(len(words_11.split()) / WORDS_PER_SECOND, 1) == 4.4


# ==============================================================================
# 3. Visual Style Fallback and Source Tests
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
        channel = None  # No channel settings

    class DummyStep:
        name = "storyboard"
        user_prompt = None
        output = None

    prev = {"script": {"script": _make_sample_script_dict()}}
    result = generate_storyboard_step(DummyVideo(), prev, DummyStep())

    default_style = load_default_visual_style()
    assert result["storyboard"]["visual_style"] == default_style
    assert result["storyboard"]["style_source"] == "default"


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
        "scenes": [{"index": 1, "image_prompt": "Old prompt", "narration": "Some narration"}],
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
    assert "Old prompt" in prompt


# ==============================================================================
# 5. Deterministic Mock Splitter Tests
# ==============================================================================

def test_mock_splitter_standard_script():
    script = _make_sample_script_dict()
    scenes = split_script_for_mock(script["narration"])

    # 1. 8 to 12 scenes produced
    assert 8 <= len(scenes) <= 12

    # 2. Re-joining scenes reproduces script narration exactly
    joined = " ".join(s.narration for s in scenes)
    assert normalize_for_comparison(joined) == normalize_for_comparison(script["narration"])

    # 3. Rotating shot types (no neighbouring duplicates)
    for i in range(1, len(scenes)):
        assert scenes[i].shot_type != scenes[i - 1].shot_type

    # 4. Image prompts are non-empty and have no forbidden whole words
    for s in scenes:
        assert s.image_prompt
        assert not FORBIDDEN_WORDS_REGEX.findall(s.image_prompt)


def test_mock_splitter_short_script_does_not_crash():
    # Very short script (~10 words)
    short_text = "Airplane windows have a hole. It balances cabin pressure."
    scenes = split_script_for_mock(short_text)

    # Must produce as many scenes as it can without crashing
    assert len(scenes) == 2
    joined = " ".join(s.narration for s in scenes)
    assert normalize_for_comparison(joined) == normalize_for_comparison(short_text)


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
    assert len(sb["scenes"]) >= 8

    # Each scene has index and est_seconds
    for idx, s in enumerate(sb["scenes"], start=1):
        assert s["index"] == idx
        assert s["est_seconds"] > 0
        assert s["shot_type"] in ALLOWED_SHOT_TYPES

    meta = result["meta"]
    assert meta["provider"] == "mock"
    assert meta["task"] == "storyboard"
    assert meta["revision"] is False
    assert "storyboard-v1:" in meta["prompt_version"]


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
                "scenes": [
                    {
                        "index": 1,
                        "narration": "Have a look at airplane windows.",
                        "image_prompt": "Airplane window",
                        "shot_type": "close-up",
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


def test_missing_script_error():
    class DummyVideo:
        id = uuid.uuid4()
        topic = "Airplane windows"
        channel = None

    class DummyStep:
        name = "storyboard"
        user_prompt = None
        output = None

    # Missing "script" key
    with pytest.raises(PipelineError, match="Script step must be completed"):
        generate_storyboard_step(DummyVideo(), {}, DummyStep())

    # "script" key exists but empty
    with pytest.raises(PipelineError, match="Script step must be completed"):
        generate_storyboard_step(DummyVideo(), {"script": {}}, DummyStep())

    # "script" has empty dict
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
                "scenes": [
                    {
                        "index": 1,
                        "narration": "First scene narration.",
                        "image_prompt": "First scene prompt",
                        "shot_type": "close-up",
                        "est_seconds": 2.0,
                    },
                    {
                        "index": 2,
                        "narration": "Second scene narration.",
                        "image_prompt": "Second scene prompt",
                        "shot_type": "macro",
                        "est_seconds": 3.0,
                    },
                ],
                "total_seconds": 5.0,
            }
        }
    }

    # h_images should extract scenes and write placeholder files
    img_result = h_images(DummyVideo(), prev, None)
    assert img_result["mock"] is True
    assert len(img_result["images"]) == 2
    assert img_result["images"][0]["scene"] == 1

    # h_voiceover should extract total_seconds and write voice.txt
    vo_result = h_voiceover(DummyVideo(), prev, None)
    assert vo_result["mock"] is True
    assert vo_result["duration_seconds"] == 5.0


# ==============================================================================
# 8. Edge Cases and Retry Flow Tests
# ==============================================================================

def test_validator_detects_zero_scenes():
    warnings = validate_storyboard_soft_checks([], "Some narration")
    assert warnings == ["Storyboard contains zero scenes."]


def test_find_first_difference_edge_branches():
    # Exact match
    assert find_first_difference("same text", "same text") == "No difference detected."
    # Str1 shorter (ends early)
    early = find_first_difference("abc", "abcdefgh")
    assert "ends early" in early
    # Str1 longer (extra text)
    extra = find_first_difference("abcdefgh", "abc")
    assert "extra text" in extra


def test_mock_splitter_empty_input():
    assert split_script_for_mock("") == []
    assert split_script_for_mock("   \n\t  ") == []


def test_storyboard_step_retry_on_warning(monkeypatch):
    """Test that a non-mock LLM triggers retry once when checks fail, updating attempts to 2."""
    from app.llm.base import BaseLLM, LLMResult

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

            return StoryboardLLMOutput(scenes=scenes), 100, 100, 10, "stop"

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

            return StoryboardLLMOutput(scenes=scenes), 100, 100, None, "stop"

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

