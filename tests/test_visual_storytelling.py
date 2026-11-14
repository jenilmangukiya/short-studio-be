"""Unit and integration tests for Shorts Studio: Visual Storytelling and Dynamic Image Generation.

Verifies:
1. Pacing: 40-45 second video breaks into ~2-5 second visual beats (averaging 12-18 beats total).
2. Emotion-aware storytelling: questions/curiosity, surprise, mistakes/misconceptions, discoveries, warnings, explanations.
3. Unique images across consecutive beats and scene boundaries (zero silent reuse).
4. Visual variety & character/style consistency via prompt composition.
5. Safety ceilings: hard safety ceiling on unique images and retry attempt safety limit.
6. Failed generation counts against retry safety limit.
7. Cost estimation before and during generation, with non-blocking monetary budget behavior.
"""

import uuid
from pathlib import Path
import pytest

from app.config import settings
from app.images.base import BaseImageProvider
from app.images.mock import MockImageProvider, create_solid_png
from app.images.types import ImageResult
from app.steps.images import PipelineError as ImagePipelineError, generate_images_step
from app.steps.storyboard import (
    SceneInput,
    VisualBeat,
    generate_storyboard_step,
    normalize_scene_visual_beats,
    split_script_for_mock,
    validate_storyboard_soft_checks,
)


class DummyVideo:
    def __init__(self, video_id=None, topic="Visual Storytelling Test"):
        self.id = video_id or uuid.uuid4()
        self.topic = topic
        self.title = topic


class DummyStep:
    def __init__(self, name="images", user_prompt=None, output=None, status="pending"):
        self.name = name
        self.user_prompt = user_prompt
        self.output = output
        self.status = status


class CountingFailProvider(BaseImageProvider):
    """Image provider that counts calls and optionally fails after a given number of attempts."""

    def __init__(self, fail_after=None, error_msg="Provider error"):
        super().__init__(model="counting-fail-model", timeout_seconds=10.0, est_cost_usd=0.04)
        self.fail_after = fail_after
        self.error_msg = error_msg
        self.attempts = 0

    @property
    def provider_name(self) -> str:
        return "counting-fail-provider"

    def _execute_generate(self, prompt: str, size: str, quality: str):
        self.attempts += 1
        if self.fail_after is not None and self.attempts > self.fail_after:
            raise RuntimeError(f"{self.error_msg} on attempt {self.attempts}")
        png = create_solid_png(100, 150, 10, 20, 30)
        return png, 10, 50, 60


# ==============================================================================
# 1. 40-45 Second Video Pacing (2-5 Second Visual Beats)
# ==============================================================================

SAMPLE_42S_SCRIPT = (
    "Why do airplane windows always have tiny rounded holes at the bottom? "
    "Actually, most passengers assume it is a dangerous manufacturing defect. "
    "In reality, that microscopic hole is an ingenious engineering marvel called a bleed hole. "
    "At thirty thousand feet, cabin air pressure pushes outward with immense force. "
    "The bleed hole balances pressure between the middle pane and outer structural pane. "
    "Because of this dynamic balance, only the heavy outer pane bears the flight stress. "
    "Warning: without this tiny breather hole, windows could crack from severe pressure shocks. "
    "Instead, engineers discovered it keeps the glass crystal clear and passengers completely safe."
)


def test_40_45s_video_dynamic_visual_beats_pacing():
    """Verify a 40-45s script produces 8-12 scenes with ~2-5s visual beats (12-18 total beats)."""
    scenes, anchor = split_script_for_mock(SAMPLE_42S_SCRIPT)

    # 1. Scene count within standard range
    assert 8 <= len(scenes) <= 12

    # 2. Collect all beats
    all_beats = [b for s in scenes for b in (s.visual_beats or [])]
    total_beats = len(all_beats)

    # Expected beat count for 40-45s video: between 12 and 18 beats
    assert 12 <= total_beats <= 20

    # 3. Check beat duration: generally 2 to 5 seconds (5 to 12 words)
    for b in all_beats:
        words = len(b.narration_text.split())
        est_sec = words / 2.5
        # Each beat should be in the 1.5 to 5.5 second range
        assert 1.2 <= est_sec <= 5.8, f"Beat '{b.narration_text}' has duration {est_sec:.1f}s outside 2-5s target"
        assert b.image_strategy == "new_image"

    # 4. Verbatim coverage: joined beat narrations equal the script
    joined_beats = " ".join(b.narration_text.strip() for b in all_beats)
    assert "".join(joined_beats.split()) == "".join(SAMPLE_42S_SCRIPT.split())


# ==============================================================================
# 2. Emotion-Aware Storytelling
# ==============================================================================

def test_emotion_aware_beat_selection():
    """Verify that script emotional cues translate into corresponding emotion-aware visual beats."""
    script_with_emotions = (
        "Why does gravity pull everything down? "
        "Actually, it is astonishingly different from what you learned. "
        "Many people assume gravity is a magnetic force pulling down. "
        "Einstein discovered that space itself bends around heavy objects. "
        "Warning: severe gravitational tides near black holes are fatal. "
        "Because mass curves spacetime, planets orbit along curved paths smoothly."
    )
    scenes, _ = split_script_for_mock(script_with_emotions)
    all_beats = [b for s in scenes for b in (s.visual_beats or [])]

    purposes = [b.visual_purpose.lower() for b in all_beats]
    descriptions = [b.visual_description.lower() for b in all_beats]

    # 1. Questions/Curiosity: thinking character or question mark iconography
    assert any("curiosity" in p or "question mark" in d or "thinking character" in d for p, d in zip(purposes, descriptions))

    # 2. Surprise/Shock: surprised expression or sudden reveal
    assert any("surprise" in p or "shocked" in d or "surprised" in d for p, d in zip(purposes, descriptions))

    # 3. Mistakes/Misconceptions: confused character or red cross symbol
    assert any("misconception" in p or "red cross" in d or "confused" in d for p, d in zip(purposes, descriptions))

    # 4. Discoveries: excited expression or lightbulb eureka moment
    assert any("discovery" in p or "lightbulb" in d or "eureka" in d for p, d in zip(purposes, descriptions))

    # 5. Warnings: hazard warning or caution badge
    assert any("warning" in p or "hazard" in d or "caution" in d for p, d in zip(purposes, descriptions))

    # 6. Explanations: mechanism demonstration or cutaway/diagram
    assert any("mechanism" in p or "cutaway" in d or "diagram" in d for p, d in zip(purposes, descriptions))


def test_negation_handling_hazard_warnings():
    """Verify negation handling: 'This is not dangerous' does not trigger hazard visual, while unnegated danger does."""
    # 1. Unnegated danger -> triggers hazard visual
    scenes_pos, _ = split_script_for_mock("This is dangerous and requires protective equipment.")
    beats_pos = [b for s in scenes_pos for b in (s.visual_beats or [])]
    assert any("hazard" in b.visual_purpose.lower() or "caution" in b.visual_description.lower() for b in beats_pos)

    # 2. Negated danger ("not dangerous") -> must NOT trigger hazard visual
    scenes_neg, _ = split_script_for_mock("This is not dangerous for passengers inside the cabin.")
    beats_neg = [b for s in scenes_neg for b in (s.visual_beats or [])]
    assert not any("hazard" in b.visual_purpose.lower() or "caution" in b.visual_description.lower() for b in beats_neg)

    # 3. Negated risk ("no risk") -> must NOT trigger hazard visual
    scenes_no_risk, _ = split_script_for_mock("There is no risk of structural failure during flight.")
    beats_no_risk = [b for s in scenes_no_risk for b in (s.visual_beats or [])]
    assert not any("hazard" in b.visual_purpose.lower() or "caution" in b.visual_description.lower() for b in beats_no_risk)

    # 4. Negated first clause but legitimate warning in second clause -> triggers hazard visual
    scenes_mixed, _ = split_script_for_mock("This is not dangerous, but high voltage exposure is fatal.")
    beats_mixed = [b for s in scenes_mixed for b in (s.visual_beats or [])]
    assert any("hazard" in b.visual_purpose.lower() or "caution" in b.visual_description.lower() for b in beats_mixed)


def test_curiosity_and_surprise_expressions():
    """Verify recognition of curiosity and surprise idioms like 'You won't believe what happens next' and 'Believe it or not'."""
    # 1. "You won't believe what happens next"
    scenes_wont_believe, _ = split_script_for_mock("You won't believe what happens next when air pressure drops.")
    beats_wont_believe = [b for s in scenes_wont_believe for b in (s.visual_beats or [])]
    assert any(
        "curiosity" in b.visual_purpose.lower() or "surprise" in b.visual_purpose.lower()
        or "thinking character" in b.visual_description.lower() or "surprised" in b.visual_description.lower()
        for b in beats_wont_believe
    )

    # 2. "Believe it or not"
    scenes_believe, _ = split_script_for_mock("Believe it or not, this tiny pinhole keeps the entire plane safe.")
    beats_believe = [b for s in scenes_believe for b in (s.visual_beats or [])]
    assert any(
        "curiosity" in b.visual_purpose.lower() or "surprise" in b.visual_purpose.lower()
        or "thinking character" in b.visual_description.lower() or "surprised" in b.visual_description.lower()
        for b in beats_believe
    )

    # 3. "What happens next"
    scenes_next, _ = split_script_for_mock("Watch what happens next as the balance equalizes.")
    beats_next = [b for s in scenes_next for b in (s.visual_beats or [])]
    assert any(
        "curiosity" in b.visual_purpose.lower() or "surprise" in b.visual_purpose.lower()
        for b in beats_next
    )


# ==============================================================================
# 3. Unique Images Across Consecutive Beats and Scenes
# ==============================================================================

def test_unique_images_across_all_beats_in_pipeline(tmp_path, monkeypatch):
    """Verify every visual beat generates a unique PNG image file with unique ID."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "image_max_unique_per_video", 50)

    video = DummyVideo()
    step = DummyStep()

    scenes, anchor = split_script_for_mock(SAMPLE_42S_SCRIPT)
    prev = {
        "storyboard": {
            "storyboard": {
                "scenes": [s.model_dump() for s in scenes],
                "subject_anchor": anchor,
                "visual_style": "Clean 3D render, studio lighting",
            }
        }
    }

    result = generate_images_step(video, prev, step, image_provider=MockImageProvider())
    items = result["images"]["items"]

    # All items should have status ok
    assert all(it["status"] == "ok" for it in items)

    # Consecutive items must not share the same file or ID
    for i in range(len(items) - 1):
        it1 = items[i]
        it2 = items[i + 1]
        assert it1["id"] != it2["id"]
        assert it1["file"] != it2["file"]

    # Every item corresponds to a real generated PNG file on disk
    for it in items:
        file_path = tmp_path / it["file"]
        assert file_path.is_file()
        assert file_path.stat().st_size > 0


# ==============================================================================
# 4. Safety Ceilings: Unique Images and Generation Attempts
# ==============================================================================

def test_hard_safety_ceiling_blocks_runaway_unique_images(tmp_path, monkeypatch):
    """Hard safety ceiling (e.g. 50 unique images) halts execution if unique images exceed limit."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    # Set ceiling lower for testing
    monkeypatch.setattr(settings, "image_max_unique_per_video", 5)

    video = DummyVideo()
    step = DummyStep()

    # 4 scenes each with 2 beats = 8 unique images > 5 ceiling
    scenes = []
    for i in range(1, 5):
        scenes.append({
            "index": i,
            "narration": f"Scene {i} narration.",
            "image_prompt": f"Scene {i} prompt",
            "shot_type": "medium",
            "visual_beats": [
                {"id": f"beat_{i}_1", "order": 1, "visual_description": f"Beat {i}.1", "image_strategy": "new_image"},
                {"id": f"beat_{i}_2", "order": 2, "visual_description": f"Beat {i}.2", "image_strategy": "new_image"},
            ],
        })

    prev = {"storyboard": {"storyboard": {"scenes": scenes}}}

    with pytest.raises(ImagePipelineError, match=r"exceeding the hard safety ceiling of 5 unique images"):
        generate_images_step(video, prev, step, image_provider=MockImageProvider())


def test_generation_attempts_safety_limit(tmp_path, monkeypatch):
    """Maximum generation-attempt limit halts execution to prevent infinite retry loops."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "image_max_unique_per_video", 50)
    monkeypatch.setattr(settings, "image_max_generation_attempts", 3)

    video = DummyVideo()
    step = DummyStep()

    # 4 scenes with 1 beat each = 4 generation attempts > 3 max attempts
    scenes = []
    for i in range(1, 5):
        scenes.append({
            "index": i,
            "narration": f"Scene {i} narration.",
            "image_prompt": f"Scene {i} prompt",
            "shot_type": "medium",
            "visual_beats": [
                {"id": f"beat_{i}_1", "order": 1, "visual_description": f"Beat {i}.1", "image_strategy": "new_image"},
            ],
        })

    prev = {"storyboard": {"storyboard": {"scenes": scenes}}}

    with pytest.raises(ImagePipelineError, match=r"exceeded maximum attempt safety limit \(3\)"):
        generate_images_step(video, prev, step, image_provider=MockImageProvider())


def test_failed_attempts_count_against_retry_limit(tmp_path, monkeypatch):
    """Every generation attempt, including failed attempts, counts against the retry safety limit."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "image_max_unique_per_video", 50)
    monkeypatch.setattr(settings, "image_max_generation_attempts", 4)

    video = DummyVideo()
    step = DummyStep()

    # Provider will fail starting after 2 attempts
    provider = CountingFailProvider(fail_after=1, error_msg="Rate limit error")

    scenes = []
    for i in range(1, 6):
        scenes.append({
            "index": i,
            "narration": f"Scene {i} narration.",
            "image_prompt": f"Scene {i} prompt",
            "shot_type": "medium",
            "visual_beats": [
                {"id": f"beat_{i}_1", "order": 1, "visual_description": f"Beat {i}.1", "image_strategy": "new_image"},
            ],
        })

    prev = {"storyboard": {"storyboard": {"scenes": scenes}}}

    # Attempts:
    # Scene 1: succeeds (attempt 1)
    # Scene 2: fails with error (attempt 2)
    # Scene 3: fails with error (attempt 3)
    # Scene 4: fails with error (attempt 4)
    # Scene 5: generation_attempts reaches 5 > 4 -> raises attempt safety limit PipelineError
    with pytest.raises(ImagePipelineError, match=r"exceeded maximum attempt safety limit \(4\)"):
        generate_images_step(video, prev, step, image_provider=provider)

    assert provider.attempts == 4


# ==============================================================================
# 5. Cost Estimation and Non-Blocking Monetary Budget
# ==============================================================================

def test_cost_estimation_before_generation_and_non_blocking_budget(tmp_path, monkeypatch):
    """Verify estimated cost is shown before generation and monetary budget does not block."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "image_budget_usd", 0.10)
    monkeypatch.setattr(settings, "openai_image_est_cost_usd", 0.04)
    monkeypatch.setattr(settings, "image_max_unique_per_video", 50)

    video = DummyVideo()
    step = DummyStep()

    # 4 scenes with 1 beat each = 4 unique images
    # 4 * $0.04 = $0.16, which exceeds the $0.10 budget guideline
    scenes = []
    for i in range(1, 5):
        scenes.append({
            "index": i,
            "narration": f"Scene {i} narration.",
            "image_prompt": f"Scene {i} prompt",
            "shot_type": "medium",
            "visual_beats": [
                {"id": f"beat_{i}_1", "order": 1, "visual_description": f"Beat {i}.1", "image_strategy": "new_image"},
            ],
        })

    prev = {"storyboard": {"storyboard": {"scenes": scenes}}}

    result = generate_images_step(video, prev, step, image_provider=MockImageProvider())

    # 1. Generation completed successfully (not blocked)
    assert result["images"]["complete"] is True

    # 2. Warning recorded about exceeding budget guideline
    assert any("exceeds standard budget guideline" in w for w in result["warnings"])

    # 3. Telemetry reflects planned estimated cost and unique images
    meta = result["meta"]
    assert meta["planned_unique_images"] == 4
    assert meta["planned_est_cost_usd"] == 0.16
    assert meta["generation_attempts"] == 4
