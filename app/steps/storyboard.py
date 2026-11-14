"""Storyboard step implementation for Shorts Studio.

Divides a structured video script into consecutive visual scenes using task-routed LLMs
(Mock, Gemini, OpenAI). Validates scene counts, timing, verbatim narration coverage,
forbidden prompt words, shot type diversity, subject anchors, and code-composed final prompts.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy.orm import Session, object_session

from app.animation import plan_scene_animation
from app.llm import LLMError, get_llm
from app.models import Video, VideoStep
from app.schemas import (
    CameraDirection,
    CharacterState,
    EntityBindings,
    FoleyCue,
    StoryboardBeat,
)
from app.steps.prompt_compose import (
    COMPOSITION_RULES,
    MAX_FINAL_PROMPT_LENGTH,
    compose_final_prompt,
    recompute_final_prompts,
)
from app.steps.storyboard_validation import validate_connected_storyboard

# Single source of truth for allowed shot framing types
ALLOWED_SHOT_TYPES: tuple[str, ...] = (
    "close-up",
    "macro",
    "extreme-close-up",
    "wide",
    "medium",
    "aerial",
    "interior",
    "exterior",
    "top-down",
    "cutaway",
    "cross-section",
    "diagram",
    "in-context",
)

ShotType = Literal[
    "close-up",
    "macro",
    "extreme-close-up",
    "wide",
    "medium",
    "aerial",
    "interior",
    "exterior",
    "top-down",
    "cutaway",
    "cross-section",
    "diagram",
    "in-context",
]

# Single source of truth for visual beat image sourcing strategies
ALLOWED_IMAGE_STRATEGIES: tuple[str, ...] = (
    "reuse_scene_image",
    "reuse_previous_image",
    "new_image",
)

ImageStrategy = Literal[
    "reuse_scene_image",
    "reuse_previous_image",
    "new_image",
]

# Words per second speech constant for calculating estimated scene duration
WORDS_PER_SECOND: float = 2.5

# Words excluded when evaluating consecutive scene semantic keyword repetition
DIVERSITY_STOPWORDS: frozenset[str] = frozenset({
    "this", "that", "with", "from", "into", "onto", "over", "under", "about",
    "scene", "view", "shows", "showing", "look", "looking", "light", "lighting",
    "soft", "clean", "upper", "lower", "frame", "middle", "background", "foreground",
    "through", "across", "along", "against", "around", "behind", "between", "while",
    "where", "small", "large", "their", "other", "another", "mechanical", "component",
    "element", "illustrating", "operation", "concept", "object", "placeholder", "details",
    "shot", "angle", "high", "deep", "dramatic", "subtle", "distinct", "style", "render",
})

# Single source of truth for words forbidden in scene image prompts (matched as whole words)
FORBIDDEN_PROMPT_WORDS: tuple[str, ...] = (
    "text",
    "letters",
    "lettering",
    "words",
    "word",
    "logo",
    "logos",
    "caption",
    "captions",
    "watermark",
    "subtitle",
    "subtitles",
)

FORBIDDEN_WORDS_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in FORBIDDEN_PROMPT_WORDS) + r")\b",
    flags=re.IGNORECASE,
)

# Words that should not dangle at scene cut boundaries without terminal punctuation
DANGLING_BOUNDARY_WORDS: frozenset[str] = frozenset({
    "and", "or", "but", "so", "because", "of", "to", "in", "for",
    "with", "from", "at", "by", "that", "which", "a", "an", "the",
})

# Default deterministic subject anchor for mock LLM runs
DEFAULT_MOCK_SUBJECT_ANCHOR: str = "A small glossy blue cube with rounded edges."
DEFAULT_MOCK_ENVIRONMENT_ANCHOR: str = ""

# Maximum allowed character length for raw scene image_prompt
MAX_IMAGE_PROMPT_LENGTH: int = 500

PROMPT_FILE_PATH = Path(__file__).resolve().parent.parent / "prompts" / "storyboard.md"
DEFAULT_STYLE_PATH = Path(__file__).resolve().parent.parent / "prompts" / "default_visual_style.txt"


class PipelineError(Exception):
    """Raised when pipeline preconditions or unrecoverable step errors occur."""


class AnimationPlanInput(BaseModel):
    """Camera animation metadata for a scene."""

    animation_type: Literal[
        "zoom-in",
        "zoom-out",
        "pan-left",
        "pan-right",
        "pan-up",
        "pan-down",
        "subtle-drift",
        "static",
    ] = Field(
        default="zoom-in",
        description="Type of camera movement: zoom-in, zoom-out, pan-left, pan-right, pan-up, pan-down, subtle-drift, static",
    )
    focal_point: str = Field(
        default="",
        description="Description of the focal point",
    )
    focal_x: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Normalized horizontal focal coordinate (0.0=left, 1.0=right)",
    )
    focal_y: float = Field(
        default=0.42,
        ge=0.0,
        le=1.0,
        description="Normalized vertical focal coordinate (0.0=top, 1.0=bottom)",
    )
    start_scale: float = Field(
        default=1.00,
        ge=1.0,
        le=1.15,
        description="Starting zoom scale factor (1.00 to 1.15)",
    )
    end_scale: float = Field(
        default=1.05,
        ge=1.0,
        le=1.15,
        description="Ending zoom scale factor (1.00 to 1.15)",
    )
    start_x: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Starting camera viewport center X (0.0 to 1.0)",
    )
    start_y: float = Field(
        default=0.42,
        ge=0.0,
        le=1.0,
        description="Starting camera viewport center Y (0.0 to 1.0)",
    )
    end_x: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Ending camera viewport center X (0.0 to 1.0)",
    )
    end_y: float = Field(
        default=0.42,
        ge=0.0,
        le=1.0,
        description="Ending camera viewport center Y (0.0 to 1.0)",
    )
    intensity: str = Field(
        default="subtle",
        description="Movement intensity: 'subtle', 'moderate', or 'restrained'",
    )
    reasoning: str = Field(
        default="",
        description="Brief explanation of why this camera motion suits the scene composition and narration",
    )


class VisualBeat(BaseModel):
    """An individual visual moment or shot within a parent scene."""

    id: str = Field(
        default="",
        description="Unique beat identifier, e.g. 'beat_1_1'",
    )
    scene_id: str = Field(
        default="",
        description="Reference identifier of parent scene, e.g. '1' or scene index",
    )
    order: int = Field(
        default=1,
        ge=1,
        description="1-based position or sequence of this beat within the parent scene",
    )
    narration_text: str = Field(
        default="",
        description="The specific narration phrase or sentence slice spoken during this visual beat",
    )
    start_time: float | None = Field(
        default=None,
        description="Beat start timestamp in seconds on the continuous narration timeline",
    )
    end_time: float | None = Field(
        default=None,
        description="Beat end timestamp in seconds on the continuous narration timeline",
    )
    duration_seconds: float | None = Field(
        default=None,
        description="Duration of this visual beat in seconds (end_time - start_time)",
    )
    start_word_index: int | None = Field(
        default=None,
        ge=0,
        description="0-based index of the first aligned narration word assigned to this beat within parent scene (inclusive)",
    )
    end_word_index: int | None = Field(
        default=None,
        ge=0,
        description="0-based index of the last aligned narration word assigned to this beat within parent scene (inclusive)",
    )
    timing_type: str = Field(
        default="unassigned",
        description="Source of beat timing: 'aligned', 'fallback', or 'unassigned'",
    )
    visual_description: str = Field(
        description="Clear description of what should be visually visible during this moment",
    )
    visual_purpose: str = Field(
        default="",
        description="Why this visual moment exists in the story (e.g. establish subject, direct focus, reveal detail, show reaction)",
    )
    shot_type: ShotType = Field(
        default="medium",
        description="Camera framing / shot type for this visual beat",
    )
    camera_angle: str = Field(
        default="eye-level",
        description="Camera angle or perspective (e.g. eye-level, low-angle, isometric, top-down)",
    )
    focal_point: str = Field(
        default="",
        description="The primary subject or detail viewers should notice in this moment",
    )
    visual_technique: str = Field(
        default="cinematic-photo",
        description="Visual technique (e.g. cinematic-photo, macro-photo, technical-illustration, cutaway-diagram)",
    )
    image_strategy: ImageStrategy = Field(
        default="new_image",
        description="Image strategy planning hint: 'new_image', 'reuse_scene_image', or 'reuse_previous_image'",
    )
    source_image_id: str | None = Field(
        default=None,
        description="Optional reference to an existing image if reusing a specific image asset",
    )
    entity_bindings: EntityBindings | None = Field(
        default=None,
        description="Bound entity IDs from Video Bible",
    )
    camera: CameraDirection | None = Field(
        default=None,
        description="Camera shot type, angle, and movement",
    )
    character_state: CharacterState | None = Field(
        default=None,
        description="Character pose, expression, and action in progress",
    )
    environment_state_override: str | None = Field(
        default=None,
        description="Environment lighting or state override for this moment",
    )
    transition_from_previous: str = Field(
        default="same_scene_reframe",
        description="Transition type from previous beat",
    )
    continuity_intent: str = Field(
        default="",
        description="Directorial intent for visual continuity",
    )
    foley_cues: list[FoleyCue] = Field(
        default_factory=list,
        description="Foley audio cues for this beat",
    )

    @property
    def image_id(self) -> str:
        """Effective image identifier for this visual beat."""
        if self.source_image_id and str(self.source_image_id).strip():
            return str(self.source_image_id).strip()
        return self.id or f"beat_{self.scene_id}_{self.order}"


class SceneInput(BaseModel):
    """Raw scene produced by the LLM."""

    narration: str = Field(
        description="The exact words spoken in this scene, reproducing a slice of the script verbatim"
    )
    visual_purpose: str = Field(
        default="",
        description="The visual story beat or viewer takeaway (e.g. establish subject, direct attention, reveal detail, interior perspective, structural cutaway, mechanism diagram, proof, closure)",
    )
    shot_type: ShotType = Field(
        description="The camera framing / shot type chosen for this scene from the allowed list"
    )
    camera_angle: str = Field(
        default="eye-level",
        description="Camera angle or perspective (e.g. wide aerial, passenger-perspective, extreme macro, 45-degree isometric, side cross-section)",
    )
    focal_point: str = Field(
        default="",
        description="The primary focal point of the image (e.g. passenger jet in clouds, highlighted window, tiny bleed hole, exploded 3-pane assembly)",
    )
    visual_technique: str = Field(
        default="cinematic-photo",
        description="Visual technique (e.g. cinematic-photo, macro-photo, passenger-pov, technical-illustration, cutaway-diagram, schematic)",
    )
    environment: str = Field(
        default="",
        description="The specific environment or background setting for this shot (e.g. data center server hall, passenger cabin, outer space, minimal technical void)",
    )
    image_prompt: str = Field(
        default="",
        description="Rich visual description. Optional when Video Bible is active, as application code composes the final prompt.",
    )
    uses_anchor: bool = Field(
        default=False,
        description="True ONLY if the recurring anchor object is the central physical subject of this specific shot. Set to false for establishing shots, interior POV, diagrams, cutaways, etc.",
    )
    entity_bindings: EntityBindings | None = Field(
        default=None,
        description="Bound entity IDs from Video Bible (character, environment, objects, concept)",
    )
    camera: CameraDirection | None = Field(
        default=None,
        description="Camera shot type, angle, focal point, and movement",
    )
    character_state: CharacterState | None = Field(
        default=None,
        description="Character pose, expression, and action in progress if character appears",
    )
    environment_state_override: str | None = Field(
        default=None,
        description="Environment lighting or state override for this moment",
    )
    transition_from_previous: str = Field(
        default="establishing",
        description="Transition type from previous scene",
    )
    continuity_intent: str = Field(
        default="",
        description="Directorial intent for visual continuity",
    )
    foley_cues: list[FoleyCue] = Field(
        default_factory=list,
        description="Foley sound design cues for this scene",
    )
    animation: AnimationPlanInput | None = Field(
        default=None,
        description="Structured camera animation instructions for this scene",
    )
    visual_beats: list[VisualBeat] = Field(
        default_factory=list,
        description="Ordered list of visual beats (visual moments) within this scene",
    )


def normalize_scene_visual_beats(
    raw_beats: list[VisualBeat | dict[str, Any]] | None,
    scene_index: int,
    fallback_scene: SceneInput | dict[str, Any] | None = None,
) -> list[VisualBeat]:
    """Ensure a scene always has a valid, non-empty, sequentially ordered list of VisualBeats.

    If raw_beats is empty or invalid, synthesizes a 1-to-1 default VisualBeat matching the parent scene.
    If raw_beats has items, sanitizes and clamps fields, guaranteeing stable sequential order (1..N).
    """

    def _extract_sc(field: str, default: Any = "") -> Any:
        if fallback_scene is None:
            return default
        if isinstance(fallback_scene, dict):
            return fallback_scene.get(field, default)
        return getattr(fallback_scene, field, default)

    sc_shot = _extract_sc("shot_type", "medium")
    fallback_shot: ShotType = sc_shot if sc_shot in ALLOWED_SHOT_TYPES else "medium"
    fallback_angle = str(_extract_sc("camera_angle", "eye-level") or "eye-level").strip()
    fallback_focal = str(_extract_sc("focal_point", "") or "").strip()
    fallback_tech = str(_extract_sc("visual_technique", "cinematic-photo") or "cinematic-photo").strip()
    fallback_purpose = str(_extract_sc("visual_purpose", "") or "").strip()
    fallback_prompt = str(_extract_sc("image_prompt", "") or "").strip()
    fallback_narration = str(_extract_sc("narration", "") or "").strip()
    fallback_desc = fallback_purpose or fallback_prompt or f"Scene {scene_index} visual moment"

    fallback_bindings = _extract_sc("entity_bindings", None)
    fallback_camera = _extract_sc("camera", None)
    fallback_char_state = _extract_sc("character_state", None)
    fallback_env_override = _extract_sc("environment_state_override", None)
    fallback_transition = _extract_sc("transition_from_previous", "same_scene_reframe")
    fallback_continuity = _extract_sc("continuity_intent", "")
    fallback_foley = _extract_sc("foley_cues", [])

    if not raw_beats:
        return [
            VisualBeat(
                id=f"beat_{scene_index}_1",
                scene_id=str(scene_index),
                order=1,
                narration_text=fallback_narration,
                visual_description=fallback_desc,
                visual_purpose=fallback_purpose or "Establish scene visual beat",
                shot_type=fallback_shot,
                camera_angle=fallback_angle or "eye-level",
                focal_point=fallback_focal,
                visual_technique=fallback_tech or "cinematic-photo",
                image_strategy="new_image",
                source_image_id=str(scene_index),
                entity_bindings=fallback_bindings,
                camera=fallback_camera,
                character_state=fallback_char_state,
                environment_state_override=fallback_env_override,
                transition_from_previous=fallback_transition or "same_scene_reframe",
                continuity_intent=fallback_continuity or "",
                foley_cues=fallback_foley or [],
            )
        ]

    normalized: list[VisualBeat] = []
    for item in raw_beats:
        if isinstance(item, VisualBeat):
            b_dict = item.model_dump()
        elif isinstance(item, dict):
            b_dict = dict(item)
        else:
            continue

        seq_order = len(normalized) + 1
        b_id = str(b_dict.get("id") or "").strip() or f"beat_{scene_index}_{seq_order}"
        b_desc = str(b_dict.get("visual_description") or "").strip() or fallback_desc
        b_purpose = str(b_dict.get("visual_purpose") or "").strip() or fallback_purpose
        b_narr = str(b_dict.get("narration_text") or "").strip()
        if not b_narr and len(raw_beats) == 1:
            b_narr = fallback_narration

        b_start_time = float(b_dict["start_time"]) if b_dict.get("start_time") is not None else None
        b_end_time = float(b_dict["end_time"]) if b_dict.get("end_time") is not None else None
        b_duration = float(b_dict["duration_seconds"]) if b_dict.get("duration_seconds") is not None else None
        b_start_word = int(b_dict["start_word_index"]) if b_dict.get("start_word_index") is not None else None
        b_end_word = int(b_dict["end_word_index"]) if b_dict.get("end_word_index") is not None else None
        b_timing_type = str(b_dict.get("timing_type") or "unassigned")

        b_shot_raw = b_dict.get("shot_type")
        b_shot: ShotType = b_shot_raw if b_shot_raw in ALLOWED_SHOT_TYPES else fallback_shot

        b_angle = str(b_dict.get("camera_angle") or "").strip() or fallback_angle or "eye-level"
        b_focal = str(b_dict.get("focal_point") or "").strip() or fallback_focal
        b_tech = str(b_dict.get("visual_technique") or "").strip() or fallback_tech or "cinematic-photo"

        b_strat_raw = b_dict.get("image_strategy")
        if b_strat_raw in ALLOWED_IMAGE_STRATEGIES:
            b_strat: ImageStrategy = b_strat_raw
        else:
            b_strat = "new_image"

        b_source_raw = b_dict.get("source_image_id")
        b_source_id = str(b_source_raw).strip() if b_source_raw is not None and str(b_source_raw).strip() else None
        if b_strat == "reuse_scene_image" and not b_source_id:
            b_source_id = str(scene_index)
        elif b_strat == "reuse_previous_image" and not b_source_id:
            # Do not automatically infer or carry-forward previous scene or beat's image
            b_source_id = None

        normalized.append(
            VisualBeat(
                id=b_id,
                scene_id=str(scene_index),
                order=seq_order,
                narration_text=b_narr,
                start_time=b_start_time,
                end_time=b_end_time,
                duration_seconds=b_duration,
                start_word_index=b_start_word,
                end_word_index=b_end_word,
                timing_type=b_timing_type,
                visual_description=b_desc,
                visual_purpose=b_purpose,
                shot_type=b_shot,
                camera_angle=b_angle,
                focal_point=b_focal,
                visual_technique=b_tech,
                image_strategy=b_strat,
                source_image_id=b_source_id,
                entity_bindings=b_dict.get("entity_bindings") or fallback_bindings,
                camera=b_dict.get("camera") or fallback_camera,
                character_state=b_dict.get("character_state") or fallback_char_state,
                environment_state_override=b_dict.get("environment_state_override") or fallback_env_override,
                transition_from_previous=b_dict.get("transition_from_previous") or fallback_transition or "same_scene_reframe",
                continuity_intent=b_dict.get("continuity_intent") or fallback_continuity or "",
                foley_cues=b_dict.get("foley_cues") or fallback_foley or [],
            )
        )

    if not normalized:
        return [
            VisualBeat(
                id=f"beat_{scene_index}_1",
                scene_id=str(scene_index),
                order=1,
                narration_text=fallback_narration,
                visual_description=fallback_desc,
                visual_purpose=fallback_purpose or "Establish scene visual beat",
                shot_type=fallback_shot,
                camera_angle=fallback_angle or "eye-level",
                focal_point=fallback_focal,
                visual_technique=fallback_tech or "cinematic-photo",
                image_strategy="new_image",
                source_image_id=str(scene_index),
            )
        ]

    return normalized


class StoryboardLLMOutput(BaseModel):
    """Structured storyboard output returned by the LLM."""

    subject_anchor: str = Field(
        default="",
        description=(
            "One or two plain sentences describing the main recurring object(s) of the video: "
            "shape, colors, materials, size cues. No style words, no text, letters, numbers, logos, brands or people."
        ),
    )
    environment_anchor: str = Field(
        default="",
        description=(
            "One or two plain sentences describing the primary environment or setting across scenes "
            "(e.g. room architecture, materials, ambient colors, environmental lighting). "
            "Leave empty if scenes take place across varied environments."
        ),
    )
    scenes: list[SceneInput] = Field(
        description="List of 8 to 12 consecutive scenes covering the entire narration"
    )


def load_prompt_template(path: Path = PROMPT_FILE_PATH) -> tuple[str, str]:
    """Load the storyboard prompt rules from disk and derive a version string with a content hash.

    Returns:
        tuple of (prompt_text, prompt_version_identifier)
    """
    if not path.exists():
        raise FileNotFoundError(f"Storyboard prompt template not found at {path}")

    content = path.read_text(encoding="utf-8")
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]

    version_label = "storyboard-v1"
    first_line = content.splitlines()[0].strip() if content.splitlines() else ""
    if first_line.lower().startswith("version:"):
        version_label = first_line.split(":", 1)[1].strip()
    elif first_line.startswith("#"):
        version_label = first_line.lstrip("#").strip().split()[0]

    prompt_version = f"{version_label}:{content_hash}"
    return content, prompt_version


def load_default_visual_style(path: Path = DEFAULT_STYLE_PATH) -> str:
    """Load the fallback visual style description from disk."""
    if path.exists():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return (
        "Clean 3D render, soft studio lighting, smooth matte materials, "
        "simple pastel gradient background, shallow depth of field"
    )


def normalize_for_comparison(text: str) -> str:
    """Normalize whitespace, dashes, and smart quotes for reliable matching."""
    t = text.strip()
    # Replace curly/smart quotes with standard quotes
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    # Normalize dashes and hyphens with surrounding spacing
    t = t.replace("—", "-").replace("–", "-")
    t = re.sub(r"\s*-\s*", " - ", t)
    # Collapse multiple whitespace
    t = re.sub(r"\s+", " ", t)
    return t


def find_first_difference(str1: str, str2: str) -> str:
    """Return a descriptive message pointing to the first point of divergence between two strings."""
    min_len = min(len(str1), len(str2))
    for i in range(min_len):
        if str1[i] != str2[i]:
            ctx_start = max(0, i - 15)
            ctx_end = min(i + 25, max(len(str1), len(str2)))
            return (
                f"Divergence at position {i}: expected '...{str2[ctx_start:ctx_end]}...' "
                f"but got '...{str1[ctx_start:ctx_end]}...'"
            )
    if len(str1) < len(str2):
        return f"Joined narration ends early at character {min_len}: missing '{str2[min_len:min_len+30]}...'"
    elif len(str1) > len(str2):
        return f"Joined narration has extra text after character {min_len}: '{str1[min_len:min_len+30]}...'"
    return "No difference detected."


def validate_storyboard_soft_checks(
    scenes: list[SceneInput | dict[str, Any]],
    script_narration: str,
    subject_anchor: str = "",
    visual_style: str = "",
    environment_anchor: str = "",
) -> list[str]:
    """Execute code-side soft checks on the generated storyboard scenes.

    Returns a list of warning messages for any failed checks.
    """
    warnings: list[str] = []

    if not scenes:
        warnings.append("Storyboard contains zero scenes.")
        return warnings

    # Helper to extract scene fields uniformly
    def get_field(scene: SceneInput | dict[str, Any], field: str) -> Any:
        if isinstance(scene, dict):
            return scene.get(field, "")
        return getattr(scene, field, "")

    # 1. Scene count check (target: 8 - 12 scenes)
    scene_count = len(scenes)
    if scene_count < 8 or scene_count > 12:
        warnings.append(
            f"Scene count is {scene_count} (target: 8-12 scenes)."
        )

    # 2. Narration equality check: joined scene narrations must equal script narration
    scene_narrations = [get_field(s, "narration") for s in scenes]
    joined_narration = " ".join(n.strip() for n in scene_narrations if n.strip())
    norm_joined = normalize_for_comparison(joined_narration)
    norm_script = normalize_for_comparison(script_narration)

    if norm_joined != norm_script:
        diff_info = find_first_difference(norm_joined, norm_script)
        warnings.append(
            f"Scene narrations joined do not reproduce the script narration exactly. {diff_info}"
        )

    # 3. Subject anchor checks
    if not subject_anchor or not subject_anchor.strip():
        warnings.append("Subject anchor is empty.")
    else:
        anchor_str = subject_anchor.strip()
        if len(anchor_str) > 250:
            warnings.append(
                f"Subject anchor length is {len(anchor_str)} chars (target: at most 250 chars)."
            )

        # Lenient sentence count (split on terminal punctuation followed by space + capital letter)
        anchor_sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", anchor_str)
            if s.strip()
        ]
        if len(anchor_sentences) > 2:
            warnings.append(
                f"Subject anchor has {len(anchor_sentences)} sentences (target: at most 2 sentences)."
            )

        # Forbidden words in subject_anchor (whole words only)
        forbidden_anchor = sorted(list({m.lower() for m in FORBIDDEN_WORDS_REGEX.findall(anchor_str)}))
        if forbidden_anchor:
            warnings.append(
                f"Subject anchor contains forbidden whole words: {forbidden_anchor}."
            )

    # 3b. Environment anchor checks (optional)
    if environment_anchor and environment_anchor.strip():
        env_str = environment_anchor.strip()
        if len(env_str) > 250:
            warnings.append(
                f"Environment anchor length is {len(env_str)} chars (target: at most 250 chars)."
            )
        forbidden_env = sorted(list({m.lower() for m in FORBIDDEN_WORDS_REGEX.findall(env_str)}))
        if forbidden_env:
            warnings.append(
                f"Environment anchor contains forbidden whole words: {forbidden_env}."
            )

    # 4. Check that at least one scene uses the subject anchor
    has_anchor = any(bool(get_field(s, "uses_anchor")) for s in scenes)
    if not has_anchor:
        warnings.append("No scenes have uses_anchor set to true (target: at least one scene).")

    # 5. Overall storyboard shot diversity check
    if len(scenes) >= 8:
        unique_shots = {get_field(s, "shot_type") for s in scenes}
        if len(unique_shots) < 4:
            warnings.append(
                f"Storyboard uses only {len(unique_shots)} distinct shot types across {len(scenes)} scenes "
                f"(target: at least 4 distinct shot types for visual diversity)."
            )

    # 6. Per-scene duration, image_prompt, shot_type, diversity, and final_prompt checks
    prev_shot_type: str | None = None
    prev_scene: SceneInput | dict[str, Any] | None = None

    for idx, s in enumerate(scenes, start=1):
        narr = get_field(s, "narration")
        img_prompt = get_field(s, "image_prompt")
        shot_type = get_field(s, "shot_type")
        final_prompt = get_field(s, "final_prompt")
        camera_angle = get_field(s, "camera_angle")
        focal_point = get_field(s, "focal_point")
        visual_technique = get_field(s, "visual_technique")
        uses_anchor_flag = bool(get_field(s, "uses_anchor"))

        # Duration check (2.0 - 7.0 seconds)
        word_count = len(narr.split())
        est_sec = round(word_count / WORDS_PER_SECOND, 1)
        if est_sec < 2.0 or est_sec > 7.0:
            warnings.append(
                f"Scene {idx} estimated duration is {est_sec}s ({word_count} words; target: 2.0-7.0s)."
            )

        # Natural grammatical boundary checks
        narr_str = str(narr).strip()
        if narr_str:
            if word_count < 3 and len(scenes) > 1:
                warnings.append(
                    f"Scene {idx} narration has only {word_count} words ({narr_str!r}), creating an awkward fragment. Keep clauses complete."
                )

            # Check for dangling words at the end of the scene (e.g. "and", "or", "of", "to")
            if idx < len(scenes):
                last_m = re.search(r"\b([a-zA-Z]+)[^\w\s]*$", narr_str)
                if last_m:
                    last_w = last_m.group(1).lower()
                    if last_w in DANGLING_BOUNDARY_WORDS:
                        warnings.append(
                            f"Scene {idx} narration ends on dangling word '{last_w}'. Split narration at natural grammatical boundaries."
                        )

        # Image prompt checks: non-empty, max MAX_IMAGE_PROMPT_LENGTH chars, no forbidden words
        if not img_prompt or not str(img_prompt).strip():
            warnings.append(f"Scene {idx} image prompt is empty.")
        else:
            if len(img_prompt) > MAX_IMAGE_PROMPT_LENGTH:
                warnings.append(
                    f"Scene {idx} image prompt length is {len(img_prompt)} chars (target: at most {MAX_IMAGE_PROMPT_LENGTH} chars)."
                )

            forbidden_matches = sorted(list({m.lower() for m in FORBIDDEN_WORDS_REGEX.findall(img_prompt)}))
            if forbidden_matches:
                warnings.append(
                    f"Scene {idx} image prompt contains forbidden whole words: {forbidden_matches}."
                )

        # Shot type variety check: neighbouring scenes must not share identical shot_type
        if prev_shot_type is not None and shot_type == prev_shot_type:
            warnings.append(
                f"Scenes {idx-1} and {idx} have identical shot type '{shot_type}'."
            )
        prev_shot_type = shot_type

        # Visual diversity check: neighbouring scenes must not be suspiciously repetitive
        if prev_scene is not None:
            prev_prompt = str(get_field(prev_scene, "image_prompt") or "")
            curr_prompt = str(img_prompt or "")

            prev_words = {
                w.lower()
                for w in re.findall(r"\b[A-Za-z]{4,}\b", prev_prompt)
                if w.lower() not in DIVERSITY_STOPWORDS
            }
            curr_words = {
                w.lower()
                for w in re.findall(r"\b[A-Za-z]{4,}\b", curr_prompt)
                if w.lower() not in DIVERSITY_STOPWORDS
            }

            if prev_words and curr_words:
                common_words = prev_words & curr_words
                union_words = prev_words | curr_words
                jaccard = len(common_words) / len(union_words) if union_words else 0.0

                if len(common_words) >= 4 and jaccard >= 0.5:
                    sorted_common = sorted(list(common_words))
                    warnings.append(
                        f"Scenes {idx-1} and {idx} have suspiciously repetitive visual subjects "
                        f"({', '.join(sorted_common[:6])}). Vary the visual subject, perspective, or technique."
                    )

            prev_fp = str(get_field(prev_scene, "focal_point") or "").strip().lower()
            curr_fp = str(focal_point or "").strip().lower()
            if prev_fp and curr_fp and prev_fp == curr_fp:
                warnings.append(
                    f"Scenes {idx-1} and {idx} have identical visual focal point '{curr_fp}'. "
                    f"Advance the visual beat to a new subject or detail."
                )

        prev_scene = s

        # Final prompt check: non-empty and sensible length limit (compose if missing)
        resolved_final_prompt = final_prompt
        if not resolved_final_prompt:
            resolved_final_prompt = compose_final_prompt(
                scene=s,
                subject_anchor=subject_anchor,
                visual_style=visual_style,
                environment_anchor=environment_anchor,
            )

        if not resolved_final_prompt or not str(resolved_final_prompt).strip():
            warnings.append(f"Scene {idx} final prompt is empty.")
        elif len(str(resolved_final_prompt)) > MAX_FINAL_PROMPT_LENGTH:
            warnings.append(
                f"Scene {idx} final prompt length is {len(str(resolved_final_prompt))} chars (target: at most {MAX_FINAL_PROMPT_LENGTH} chars)."
            )

        # Visual beats checks (if present)
        beats = get_field(s, "visual_beats")
        if isinstance(beats, list) and len(beats) > 0:
            if len(beats) > 4:
                warnings.append(
                    f"Scene {idx} contains {len(beats)} visual beats (target: 1-3 beats per scene)."
                )
            for b_idx, b in enumerate(beats, start=1):
                b_order = b.get("order") if isinstance(b, dict) else getattr(b, "order", b_idx)
                if b_order != b_idx:
                    warnings.append(
                        f"Scene {idx} visual beat {b_idx} has non-sequential order {b_order} (expected {b_idx})."
                    )
                b_desc = b.get("visual_description") if isinstance(b, dict) else getattr(b, "visual_description", "")
                if not b_desc or not str(b_desc).strip():
                    warnings.append(f"Scene {idx} visual beat {b_idx} has an empty visual_description.")
                b_strat = b.get("image_strategy") if isinstance(b, dict) else getattr(b, "image_strategy", "")
                if b_strat and b_strat not in ALLOWED_IMAGE_STRATEGIES:
                    warnings.append(
                        f"Scene {idx} visual beat {b_idx} has invalid image_strategy '{b_strat}'."
                    )

    # Cross-beat distinctness & uniqueness checks across consecutive visual beats
    all_storyboard_beats: list[tuple[int, int, Any]] = []
    for s_idx, s in enumerate(scenes, start=1):
        s_beats = get_field(s, "visual_beats")
        if isinstance(s_beats, list):
            for b_idx, b in enumerate(s_beats, start=1):
                all_storyboard_beats.append((s_idx, b_idx, b))

    for i in range(len(all_storyboard_beats) - 1):
        s_idx_1, b_idx_1, b1 = all_storyboard_beats[i]
        s_idx_2, b_idx_2, b2 = all_storyboard_beats[i + 1]

        def _b_get(beat: Any, key: str, default: Any = "") -> Any:
            if isinstance(beat, dict):
                return beat.get(key, default)
            return getattr(beat, key, default)

        sid1 = _b_get(b1, "source_image_id")
        sid2 = _b_get(b2, "source_image_id")

        # Flag repeated image IDs between consecutive visual beats
        if sid1 and sid2 and str(sid1).strip() == str(sid2).strip():
            warnings.append(
                f"Consecutive visual beats (scene {s_idx_1} beat {b_idx_1} and scene {s_idx_2} beat {b_idx_2}) "
                f"share the same image ID '{sid1}'. Visual beats must not reuse the same image."
            )

        # Check prompt distinctness between consecutive beats
        desc1 = str(_b_get(b1, "visual_description", "")).strip().lower()
        desc2 = str(_b_get(b2, "visual_description", "")).strip().lower()
        if desc1 and desc2 and desc1 == desc2:
            warnings.append(
                f"Consecutive visual beats (scene {s_idx_1} beat {b_idx_1} and scene {s_idx_2} beat {b_idx_2}) "
                "have identical visual descriptions. Vary perspective, camera angle, composition, and action."
            )

    return warnings


# Narrative visual beats used for deterministic mock storyboard generation
MOCK_VISUAL_BEATS: tuple[dict[str, Any], ...] = (
    {
        "purpose": "Establish subject and scale",
        "shot_type": "aerial",
        "camera_angle": "high-angle aerial",
        "technique": "cinematic-photo",
        "focal_point": "subject soaring above cloudscape",
        "prompt": "Wide aerial establishing shot capturing the primary subject high above rolling sunset clouds",
        "uses_anchor": False,
        "animation": {
            "animation_type": "zoom-in",
            "focal_point": "subject soaring above cloudscape",
            "focal_x": 0.50,
            "focal_y": 0.40,
            "start_scale": 1.00,
            "end_scale": 1.05,
            "start_x": 0.50,
            "start_y": 0.40,
            "end_x": 0.50,
            "end_y": 0.40,
            "intensity": "subtle",
            "reasoning": "Slow push-in toward primary subject draws viewer attention naturally.",
        },
    },
    {
        "purpose": "Direct attention to specific focus",
        "shot_type": "medium",
        "camera_angle": "exterior 45-degree angle",
        "technique": "cinematic-photo",
        "focal_point": "highlighted passenger window array",
        "prompt": "Exterior medium framing directing viewer focus towards the key component with focused illumination",
        "uses_anchor": True,
        "animation": {
            "animation_type": "zoom-in",
            "focal_point": "highlighted passenger window array",
            "focal_x": 0.50,
            "focal_y": 0.42,
            "start_scale": 1.00,
            "end_scale": 1.05,
            "start_x": 0.50,
            "start_y": 0.42,
            "end_x": 0.50,
            "end_y": 0.42,
            "intensity": "subtle",
            "reasoning": "Slow push-in toward passenger window directs viewer attention.",
        },
    },
    {
        "purpose": "Reveal tiny detail and texture",
        "shot_type": "macro",
        "camera_angle": "extreme close-up 90-degree",
        "technique": "macro-photo",
        "focal_point": "miniature bleed opening and texture",
        "prompt": "Macro extreme close-up revealing fine physical detail and miniature circular opening in clear material",
        "uses_anchor": True,
        "animation": {
            "animation_type": "zoom-out",
            "focal_point": "miniature bleed opening and texture",
            "focal_x": 0.50,
            "focal_y": 0.45,
            "start_scale": 1.05,
            "end_scale": 1.00,
            "start_x": 0.50,
            "start_y": 0.45,
            "end_x": 0.50,
            "end_y": 0.45,
            "intensity": "moderate",
            "reasoning": "Gentle pull-out from tiny bleed hole reveals acrylic frame without cropping edges.",
        },
    },
    {
        "purpose": "Interior passenger perspective",
        "shot_type": "interior",
        "camera_angle": "passenger-perspective eye-level",
        "technique": "passenger-pov",
        "focal_point": "cabin frame looking outward",
        "prompt": "First-person passenger perspective from inside cabin looking past contoured bezel to distant sky",
        "uses_anchor": False,
        "animation": {
            "animation_type": "zoom-in",
            "focal_point": "cabin frame looking outward",
            "focal_x": 0.50,
            "focal_y": 0.42,
            "start_scale": 1.00,
            "end_scale": 1.05,
            "start_x": 0.50,
            "start_y": 0.42,
            "end_x": 0.50,
            "end_y": 0.42,
            "intensity": "subtle",
            "reasoning": "Slow push-in toward cabin window view conveys calm passenger perspective.",
        },
    },
    {
        "purpose": "Explain construction and layers",
        "shot_type": "cutaway",
        "camera_angle": "isometric technical cutaway",
        "technique": "technical-illustration",
        "focal_point": "three distinct separated layer assembly",
        "prompt": "Exploded technical cutaway diagram illustrating the three distinct structural layers in separation",
        "uses_anchor": False,
        "animation": {
            "animation_type": "static",
            "focal_point": "three distinct separated layer assembly",
            "focal_x": 0.50,
            "focal_y": 0.42,
            "start_scale": 1.00,
            "end_scale": 1.00,
            "start_x": 0.50,
            "start_y": 0.42,
            "end_x": 0.50,
            "end_y": 0.42,
            "intensity": "restrained",
            "reasoning": "Static hold maintains crisp legibility of technical cutaway layers.",
        },
    },
    {
        "purpose": "Schematic diagram of functional forces",
        "shot_type": "diagram",
        "camera_angle": "side cross-section schematic",
        "technique": "schematic",
        "focal_point": "directional flow arrows balancing pressure",
        "prompt": "Cross-section schematic diagram illustrating dynamic pressure balance with directional flow arrows",
        "uses_anchor": False,
        "animation": {
            "animation_type": "static",
            "focal_point": "directional flow arrows balancing pressure",
            "focal_x": 0.50,
            "focal_y": 0.42,
            "start_scale": 1.00,
            "end_scale": 1.00,
            "start_x": 0.50,
            "start_y": 0.42,
            "end_x": 0.50,
            "end_y": 0.42,
            "intensity": "restrained",
            "reasoning": "Static hold ensures clear readability of schematic diagram labels and flow arrows.",
        },
    },
    {
        "purpose": "Proof of real-world operation",
        "shot_type": "close-up",
        "camera_angle": "low-angle close-up",
        "technique": "cinematic-photo",
        "focal_point": "crystal clear surface resisting frost",
        "prompt": "Detailed close-up demonstrating clear unblemished surface remaining free of frost in subzero air",
        "uses_anchor": True,
        "animation": {
            "animation_type": "zoom-in",
            "focal_point": "crystal clear surface resisting frost",
            "focal_x": 0.50,
            "focal_y": 0.42,
            "start_scale": 1.00,
            "end_scale": 1.05,
            "start_x": 0.50,
            "start_y": 0.42,
            "end_x": 0.50,
            "end_y": 0.42,
            "intensity": "subtle",
            "reasoning": "Gentle zoom-in showcases crystal clarity of the window pane.",
        },
    },
    {
        "purpose": "Visual closure and resolution",
        "shot_type": "wide",
        "camera_angle": "wide cinematic horizon",
        "technique": "cinematic-photo",
        "focal_point": "peaceful horizon in twilight sunset",
        "prompt": "Cinematic wide composition with warm twilight rim lighting bringing elegant visual closure",
        "uses_anchor": False,
        "animation": {
            "animation_type": "zoom-in",
            "focal_point": "peaceful horizon in twilight sunset",
            "focal_x": 0.50,
            "focal_y": 0.40,
            "start_scale": 1.00,
            "end_scale": 1.05,
            "start_x": 0.50,
            "start_y": 0.40,
            "end_x": 0.50,
            "end_y": 0.40,
            "intensity": "subtle",
            "reasoning": "Gentle push-in toward twilight sunset brings warm, elegant closure.",
        },
    },
)


def _has_unnegated_hazard(text: str) -> bool:
    """Check whether text contains a hazard/danger term that is not negated by 'not', 'never', 'no', etc."""
    hazard_terms = (
        "danger",
        "dangerous",
        "warning",
        "fatal",
        "risk",
        "risky",
        "caution",
        "hazard",
        "hazardous",
        "disaster",
        "critical",
    )
    for match in re.finditer(r"\b(" + "|".join(hazard_terms) + r")\b", text):
        start = match.start()
        preceding = text[max(0, start - 30) : start].strip()
        # If preceded by negation (e.g. "not", "never", "no", "is not", "was not", "isn't", "aren't")
        if re.search(r"\b(not|never|no|isn't|isnt|aren't|arent|wasn't|weren't)\b(?:\s+\w+){0,2}\s*$", preceding):
            continue
        return True
    return False


def _generate_mock_visual_beats_for_scene(
    chunk: str, scene_idx: int, beat: dict[str, Any]
) -> list[VisualBeat]:
    """Break scene narration into meaningful, emotion-aware visual beats (2–5 seconds pacing)."""
    words = chunk.split()
    total_words = len(words)

    # 1. Determine slices
    slices: list[str] = []
    if total_words <= 6:
        slices = [chunk]
    elif total_words <= 13:
        # Split into 2 beats around mid
        mid = total_words // 2
        punct_match = None
        for m in re.finditer(r"([,;:—–?!])\s+", chunk):
            idx = len(chunk[: m.end()].split())
            if 3 <= idx <= total_words - 3:
                punct_match = m
                break
        if punct_match:
            split_pos = punct_match.end()
            p1 = chunk[:split_pos].strip()
            p2 = chunk[split_pos:].strip()
            if p1 and p2:
                slices = [p1, p2]
        if not slices:
            p1 = " ".join(words[:mid])
            p2 = " ".join(words[mid:])
            slices = [p1, p2]
    else:
        # 14+ words: split into 2 or 3 beats
        punct_matches = list(re.finditer(r"([,;:—–?!])\s+", chunk))
        valid_puncts = [
            m for m in punct_matches
            if 4 <= len(chunk[: m.end()].split()) <= total_words - 4
        ]
        if len(valid_puncts) >= 2:
            m1 = valid_puncts[0]
            m2 = valid_puncts[-1]
            p1 = chunk[: m1.end()].strip()
            p2 = chunk[m1.end() : m2.end()].strip()
            p3 = chunk[m2.end() :].strip()
            if p1 and p2 and p3:
                slices = [p1, p2, p3]
        elif len(valid_puncts) == 1:
            m = valid_puncts[0]
            p1 = chunk[: m.end()].strip()
            p2 = chunk[m.end() :].strip()
            slices = [p1, p2]
        else:
            third = total_words // 3
            if total_words >= 16:
                p1 = " ".join(words[:third])
                p2 = " ".join(words[third : 2 * third])
                p3 = " ".join(words[2 * third :])
                slices = [p1, p2, p3]
            else:
                mid = total_words // 2
                slices = [" ".join(words[:mid]), " ".join(words[mid:])]

    # 2. Assign emotion-aware and varied visual attributes to each beat
    beats_out: list[VisualBeat] = []
    focal_base = beat.get("focal_point", "key subject")
    technique_base = beat.get("technique", "cinematic-photo")

    for b_idx, p_text in enumerate(slices, start=1):
        p_lower = p_text.lower()

        # Emotion-aware storytelling detection
        if "?" in p_text or any(k in p_lower for k in ("why", "how", "what if", "ever wonder", "mystery", "puzzling", "what happens next")):
            purpose = "Question curiosity: explore the puzzle"
            if b_idx == 1:
                shot = "medium"
                angle = "eye-level"
                focal = "curious thinking character with glowing question mark"
                prompt = f"Stylized thinking character with curious expression looking upward toward glowing question mark iconography in atmospheric lighting, exploring {focal_base}"
            elif b_idx == 2:
                shot = "close-up"
                angle = "low-angle hero"
                focal = "curious character tilting head examining puzzle"
                prompt = f"Close-up of curious character tilting head with glowing question mark reflection, intently examining {focal_base}"
            else:
                shot = "top-down"
                angle = "top-down flat lay"
                focal = "thinking silhouette surrounded by question cues"
                prompt = f"Top-down isometric view of thinking character silhouette surrounded by glowing curiosity question marks around {focal_base}"
            tech = "cinematic-photo"
        elif any(k in p_lower for k in ("actually", "shocking", "surprising", "unexpected", "suddenly", "unbelievable", "twist", "jaw-drop", "won't believe", "wont believe", "believe it or not")):
            purpose = "Surprise reaction: sudden dramatic reveal"
            if b_idx == 1:
                shot = "close-up"
                angle = "low-angle hero"
                focal = "surprised character with wide eyes in dramatic lighting"
                prompt = f"Close-up of stylized character with wide eyes and shocked expression, dramatic reveal with high contrast rim lighting highlighting {focal_base}"
            elif b_idx == 2:
                shot = "extreme-close-up"
                angle = "eye-level"
                focal = "astonished reaction revealing unexpected truth"
                prompt = f"Extreme close-up of astonished character reaction with dramatic spotlight reveal uncovering {focal_base}"
            else:
                shot = "wide"
                angle = "low-angle"
                focal = "dramatic wide perspective of sudden reveal"
                prompt = f"Wide dramatic perspective of astonished character stepping back in surprise from {focal_base}"
            tech = "cinematic-photo"
        elif any(k in p_lower for k in ("wrong", "myth", "mistake", "assume", "assumed", "lie", "false", "misconception", "fail")):
            purpose = "Misconception contrast: highlight the mistake"
            if b_idx == 1:
                shot = "medium"
                angle = "top-down flat lay"
                focal = "confused character with prominent red cross symbol"
                prompt = f"Confused stylized character scratching head beside a prominent red cross wrong-answer symbol, contrasting assumption with reality of {focal_base}"
            elif b_idx == 2:
                shot = "close-up"
                angle = "eye-level"
                focal = "confused expression pointing at false belief indicator"
                prompt = f"Close-up of confused character pointing toward glowing red X indicator questioning flawed assumption of {focal_base}"
            else:
                shot = "in-context"
                angle = "isometric 45-degree"
                focal = "split comparison showing mistaken belief crossed out"
                prompt = f"Split-screen comparison layout showing the common mistaken belief crossed out in red next to {focal_base}"
            tech = "technical-illustration"
        elif any(k in p_lower for k in ("discovered", "turns out", "secret", "reveal", "reveals", "found", "eureka", "breakthrough", "solution")):
            purpose = "Discovery reveal: breakthrough eureka moment"
            if b_idx == 1:
                shot = "close-up"
                angle = "isometric 45-degree"
                focal = "excited character with glowing lightbulb moment"
                prompt = f"Excited stylized character with a glowing lightbulb reveal and triumphant breakthrough expression in warm luminous lighting on {focal_base}"
            elif b_idx == 2:
                shot = "medium"
                angle = "low-angle hero"
                focal = "triumphant character pointing at breakthrough lightbulb"
                prompt = f"Medium hero shot of character enthusiastically pointing up at brilliant illuminated breakthrough lightbulb over {focal_base}"
            else:
                shot = "macro"
                angle = "eye-level"
                focal = "glowing filament radiating golden eureka glow"
                prompt = f"Macro detail of glowing filament within lightbulb radiating warm golden eureka glow onto {focal_base}"
            tech = "cinematic-photo"
        elif _has_unnegated_hazard(p_lower):
            purpose = "Hazard warning: emphasize risk and consequences"
            if b_idx == 1:
                shot = "close-up"
                angle = "low-angle dutch"
                focal = "bold caution hazard badge with alert illumination"
                prompt = f"Bold caution hazard warning badge with glowing amber alert lighting and dramatic shadows indicating impending consequence for {focal_base}"
            elif b_idx == 2:
                shot = "medium"
                angle = "eye-level"
                focal = "concerned character reacting to flashing hazard symbol"
                prompt = f"Medium shot of concerned character reacting to flashing hazard triangle symbols warning against {focal_base}"
            else:
                shot = "macro"
                angle = "low-angle"
                focal = "caution barrier stripes and pulsating warning beacon"
                prompt = f"Macro view of caution barrier stripes and pulsating warning beacon illuminating {focal_base}"
            tech = "technical-illustration"
        elif any(k in p_lower for k in ("works", "because", "mechanism", "process", "system", "structure", "airflow", "pressure")):
            purpose = "Mechanism demonstration: clear technical explanation"
            if b_idx == 1:
                shot = "cutaway"
                angle = "isometric technical cutaway"
                focal = "detailed cross-section with directional flow arrows"
                prompt = f"Clean technical cutaway diagram revealing internal structure and dynamic directional flow vectors demonstrating the physical mechanism of {focal_base}"
            elif b_idx == 2:
                shot = "cross-section"
                angle = "side cross-section"
                focal = "pressure balance channels and internal components"
                prompt = f"Detailed cross-section view showing pressure balance channels and internal components of {focal_base}"
            else:
                shot = "diagram"
                angle = "isometric 45-degree"
                focal = "schematic diagram tracking airflow pathways"
                prompt = f"Schematic diagram tracking energy transfer vectors and operational airflow pathways through {focal_base}"
            tech = "cutaway-diagram"
        else:
            # Fallback contextual progression ensuring variety across beats
            if b_idx == 1:
                purpose = beat.get("purpose", "Establish visual context")
                shot = beat.get("shot_type", "medium")
                angle = beat.get("camera_angle", "eye-level")
                focal = focal_base
                prompt = beat.get("prompt", f"Clear cinematic composition showing {focal_base}")
                tech = technique_base
            elif b_idx == 2:
                purpose = f"Advance visual narrative on {focal_base}"
                shot = "close-up" if beat.get("shot_type") != "close-up" else "macro"
                angle = "top-down" if beat.get("camera_angle") != "top-down" else "isometric 45-degree"
                focal = f"{focal_base} dynamic detail"
                prompt = f"Distinct angle showing {focal_base} in active motion with dynamic framing and high contrast"
                tech = technique_base
            else:
                purpose = f"In-depth perspective of {focal_base}"
                shot = "macro" if beat.get("shot_type") != "macro" else "wide"
                angle = "low-angle hero" if beat.get("camera_angle") != "low-angle hero" else "eye-level"
                focal = f"{focal_base} intricate texture"
                prompt = f"Detailed macro composition revealing intricate texture and physical depth of {focal_base}"
                tech = "macro-photo" if technique_base != "macro-photo" else "cinematic-photo"

        beats_out.append(
            VisualBeat(
                id=f"beat_{scene_idx}_{b_idx}",
                scene_id=str(scene_idx),
                order=b_idx,
                narration_text=p_text,
                visual_description=prompt,
                visual_purpose=purpose,
                shot_type=shot if shot in ALLOWED_SHOT_TYPES else "medium",
                camera_angle=angle,
                focal_point=focal,
                visual_technique=tech,
                image_strategy="new_image",
                source_image_id=f"img_{scene_idx}_{b_idx}",
            )
        )

    return beats_out


def split_script_for_mock(
    narration: str,
    video_bible_data: dict[str, Any] | None = None,
    creative_direction_data: dict[str, Any] | None = None,
) -> tuple[list[SceneInput], str]:
    """Deterministically split script narration into scenes along sentence/clause boundaries.

    Aims for 8-12 scenes. If fewer than 8 sentences, splits long sentences at commas.
    If the text is too short (<8), produces as many valid scenes as possible without crashing.
    Assigns rotating shot types, clean placeholder image prompts, and sets uses_anchor by scene index
    membership (false for scenes 2 and 5, true for all others).

    When video_bible_data is provided, binds entities, camera directions, and character states.

    Returns:
        tuple of (scenes, subject_anchor)
    """
    raw_text = narration.strip()
    if not raw_text:
        return [], ""

    # Step 1: Split into initial sentences along sentence-ending punctuation
    sentence_matches = [s.strip() for s in re.split(r"(?<=[.!?])\s+", raw_text) if s.strip()]
    if not sentence_matches:
        sentence_matches = [raw_text]

    chunks = list(sentence_matches)

    # Step 2: If fewer than 8 chunks (or any chunk exceeds 17 words while < 12 chunks),
    # iteratively split long chunks at natural grammatical clause/phrase boundaries
    while len(chunks) < 8 or (len(chunks) < 12 and any(len(c.split()) > 17 for c in chunks)):
        best_idx = -1
        best_score = -1.0
        best_split_parts: tuple[str, str] | None = None
        need_more_scenes = len(chunks) < 8

        for idx, chunk in enumerate(chunks):
            words = chunk.split()
            min_words_to_split = 8 if need_more_scenes else 18
            if len(words) < min_words_to_split:
                continue

            candidates: list[tuple[float, str, str]] = []
            # Primary candidates: explicit clause punctuation (, ; : — –)
            for match in re.finditer(r"([,;:—–])\s+", chunk):
                split_pos = match.end()
                p1 = chunk[:split_pos].strip()
                p2 = chunk[split_pos:].strip()
                w1 = p1.split()
                w2 = p2.split()
                if len(w1) < 4 or len(w2) < 4:
                    continue

                last_m = re.search(r"\b([a-zA-Z]+)[^\w\s]*$", p1)
                last_w = last_m.group(1).lower() if last_m else ""
                if last_w in DANGLING_BOUNDARY_WORDS:
                    continue

                punct = match.group(1)
                bonus = 10.0 if punct in (";", ":", "—", "–") else 0.0
                if p2.lower().startswith(("and ", "but ", "while ", "because ", "so ", "though ", "which ", "where ")):
                    bonus += 8.0

                balance_penalty = abs(len(w1) - len(w2))
                length_bonus = max(0, len(words) - 16) * 2.0
                score = 50.0 + bonus + length_bonus - balance_penalty
                candidates.append((score, p1, p2))

            # Secondary candidates: unpunctuated clause transitions (e.g. before "while", "because", "until")
            if not candidates:
                for match in re.finditer(
                    r"\s+(?=(?:while|because|until|before|after|where|which|allowing|preventing)\b)",
                    chunk,
                    flags=re.IGNORECASE,
                ):
                    p1 = chunk[: match.start()].strip()
                    p2 = chunk[match.end() :].strip()
                    w1 = p1.split()
                    w2 = p2.split()
                    if len(w1) < 5 or len(w2) < 5:
                        continue
                    last_m = re.search(r"\b([a-zA-Z]+)[^\w\s]*$", p1)
                    last_w = last_m.group(1).lower() if last_m else ""
                    if last_w in DANGLING_BOUNDARY_WORDS:
                        continue
                    balance_penalty = abs(len(w1) - len(w2))
                    score = 35.0 + max(0, len(words) - 16) * 2.0 - balance_penalty
                    candidates.append((score, p1, p2))

            if candidates:
                candidates.sort(key=lambda c: c[0], reverse=True)
                top_score, top_p1, top_p2 = candidates[0]
                if top_score > best_score:
                    best_score = top_score
                    best_idx = idx
                    best_split_parts = (top_p1, top_p2)

        if best_idx != -1 and best_split_parts:
            chunks = (
                chunks[:best_idx]
                + [best_split_parts[0], best_split_parts[1]]
                + chunks[best_idx + 1:]
            )
        else:
            # Cannot find any more natural grammatical splits with >= 4 words on each side
            break

    # Step 3: If more than 12 chunks, merge adjacent shortest chunks
    while len(chunks) > 12:
        best_idx = 0
        min_combined_wc = float("inf")
        for idx in range(len(chunks) - 1):
            combined_wc = len(chunks[idx].split()) + len(chunks[idx + 1].split())
            if combined_wc < min_combined_wc:
                min_combined_wc = combined_wc
                best_idx = idx
        merged = f"{chunks[best_idx]} {chunks[best_idx + 1]}"
        chunks = chunks[:best_idx] + [merged] + chunks[best_idx + 2:]

    # Resolve Video Bible entities if available
    char_id = None
    env_id = None
    obj_id = None
    concept_id = None
    if video_bible_data:
        entities = video_bible_data.get("entities") or {}
        if hasattr(entities, "model_dump"):
            entities = entities.model_dump()
        chars = entities.get("characters") or []
        envs = entities.get("environments") or []
        objs = entities.get("objects_and_props") or []
        concepts = entities.get("concepts_and_diagrams") or []

        allow_char = True
        if creative_direction_data:
            p_cfg = creative_direction_data.get("protagonist_config") or {}
            if hasattr(p_cfg, "model_dump"):
                p_cfg = p_cfg.model_dump()
            if p_cfg.get("character_count") == "none":
                allow_char = False

        if allow_char and chars:
            char_id = str(chars[0].get("id"))
        if envs:
            env_id = str(envs[0].get("id"))
        if objs:
            obj_id = str(objs[0].get("id"))
        if concepts:
            concept_id = str(concepts[0].get("id"))

    # Step 4: Construct scenes with rotating story beats, shot types, and distinct visual purposes
    scenes: list[SceneInput] = []
    for i, chunk in enumerate(chunks):
        scene_idx = i + 1
        beat = MOCK_VISUAL_BEATS[i % len(MOCK_VISUAL_BEATS)]
        uses_anchor = scene_idx not in (2, 5)
        anim_data = beat.get("animation")
        anim_obj = AnimationPlanInput(**anim_data) if anim_data else None

        mock_beats = _generate_mock_visual_beats_for_scene(chunk, scene_idx, beat)

        # Entity and Camera binding
        scene_bindings: EntityBindings | None = None
        scene_cam: CameraDirection | None = None
        scene_char_state: CharacterState | None = None
        scene_trans: str = "establishing" if scene_idx == 1 else "same_scene_reframe"

        if video_bible_data:
            if concept_id and scene_idx == 4:
                # Scene 4 represents a standalone conceptual diagram beat
                scene_bindings = EntityBindings(
                    character_id=None,
                    environment_id=None,
                    object_ids=[],
                    concept_id=concept_id,
                )
                scene_cam = CameraDirection(
                    shot_type="medium",
                    camera_angle="eye_level",
                    focal_point_description="Central schematic diagram flow",
                    movement_type="static",
                )
                scene_char_state = None
                scene_trans = "explanatory_cutaway"
            else:
                bound_objs = [obj_id] if (obj_id and scene_idx % 2 == 1) else []
                scene_bindings = EntityBindings(
                    character_id=char_id,
                    environment_id=env_id,
                    object_ids=bound_objs,
                    concept_id=None,
                )
                valid_shots = ("extreme_close_up", "close_up", "medium_close_up", "medium", "wide", "extreme_wide")
                raw_shot = beat["shot_type"].replace("-", "_")
                shot_for_cam = raw_shot if raw_shot in valid_shots else "medium"

                valid_angles = ("eye_level", "low_angle", "high_angle", "top_down", "side_profile", "dutch_angle")
                raw_angle = beat["camera_angle"].replace("-", "_")
                angle_for_cam = raw_angle if raw_angle in valid_angles else "eye_level"

                scene_cam = CameraDirection(
                    shot_type=shot_for_cam,
                    camera_angle=angle_for_cam,
                    focal_point_description=beat["focal_point"],
                    movement_type="push_in" if scene_idx % 2 == 1 else "static",
                )
                scene_char_state = (
                    CharacterState(
                        pose_description="Resting in bed with eyes half-closed",
                        expression_description="Drowsy and relaxed",
                        action_in_progress="Drifting off to sleep",
                    )
                    if char_id
                    else None
                )
                if scene_idx == 1:
                    scene_trans = "establishing"
                elif scene_idx == 5 and concept_id:
                    scene_trans = "resolution_return"
                elif scene_idx % 2 == 0:
                    scene_trans = "action_reaction"
                else:
                    scene_trans = "same_scene_reframe"

            for mb in mock_beats:
                mb.entity_bindings = scene_bindings
                mb.camera = scene_cam
                mb.character_state = scene_char_state
                mb.transition_from_previous = scene_trans

        scenes.append(
            SceneInput(
                narration=chunk,
                visual_purpose=mock_beats[0].visual_purpose or beat["purpose"],
                shot_type=beat["shot_type"],
                camera_angle=beat["camera_angle"],
                focal_point=beat["focal_point"],
                visual_technique=beat["technique"],
                environment=beat.get("environment", ""),
                image_prompt=mock_beats[0].visual_description or beat["prompt"],
                uses_anchor=uses_anchor,
                animation=anim_obj,
                visual_beats=mock_beats,
                entity_bindings=scene_bindings,
                camera=scene_cam,
                character_state=scene_char_state,
                transition_from_previous=scene_trans,
            )
        )

    return scenes, DEFAULT_MOCK_SUBJECT_ANCHOR


def build_storyboard_prompt(
    rules_text: str,
    topic: str,
    script_data: dict[str, Any],
    visual_style: str,
    channel_name: str | None = None,
    channel_niche: str | None = None,
    previous_storyboard: dict[str, Any] | None = None,
    revision_instruction: str | None = None,
    creative_direction_data: dict[str, Any] | None = None,
    video_bible_data: dict[str, Any] | None = None,
) -> str:
    """Assemble the comprehensive prompt including script context, visual style, Video Bible, and revision details."""
    sections: list[str] = [rules_text.strip(), "\n## Video Script to Storyboard"]

    sections.append(f"- **Video Topic**: {topic}")
    title = script_data.get("title")
    if title:
        sections.append(f"- **Title**: {title}")
    hook = script_data.get("hook")
    if hook:
        sections.append(f"- **Opening Hook**: {hook}")
    angle = script_data.get("angle")
    if angle:
        sections.append(f"- **Storytelling Angle**: {angle}")

    narration = script_data.get("narration", "")
    sections.append(f"\n### Spoken Script Narration to Split (Reproduce Verbatim):\n{narration}")

    if creative_direction_data:
        sections.append("\n### Approved Creative Direction")
        sections.append(f"- **Format Archetype**: {creative_direction_data.get('format_archetype', '')}")
        sections.append(f"- **Narrative Angle**: {creative_direction_data.get('storytelling_angle', '')}")
        sections.append(f"- **Central Thesis**: {creative_direction_data.get('central_thesis', '')}")
        sections.append(f"- **Tone**: {creative_direction_data.get('tone', '')}")
        protagonist = creative_direction_data.get("protagonist_config", {})
        if isinstance(protagonist, dict):
            sections.append(f"- **Protagonist Configuration**: {protagonist.get('character_count', 'none')}")
            if protagonist.get("archetype_summary"):
                sections.append(f"- **Character Archetype**: {protagonist.get('archetype_summary')}")
        constraints = creative_direction_data.get("creative_constraints", [])
        if constraints:
            sections.append("- **Creative Constraints**:\n  " + "\n  ".join(f"- {c}" for c in constraints))

    if video_bible_data:
        sections.append("\n### Approved Video Bible (Single Source of Truth)")
        entities = video_bible_data.get("entities") or {}
        if hasattr(entities, "model_dump"):
            entities = entities.model_dump()
        art_dir = video_bible_data.get("art_direction") or {}
        if hasattr(art_dir, "model_dump"):
            art_dir = art_dir.model_dump()

        chars = entities.get("characters") or []
        envs = entities.get("environments") or []
        objs = entities.get("objects_and_props") or []
        concepts = entities.get("concepts_and_diagrams") or []

        sections.append("#### Available Entity Catalog:")
        if chars:
            sections.append("- **Characters**:")
            for c in chars:
                sections.append(f"  - `{c.get('id')}`: {c.get('name')} ({c.get('visual_summary')})")
        else:
            sections.append("- **Characters**: None (character_count is none)")

        if envs:
            sections.append("- **Environments**:")
            for e in envs:
                sections.append(f"  - `{e.get('id')}`: {e.get('name')} - {e.get('description')}")

        if objs:
            sections.append("- **Objects & Props**:")
            for o in objs:
                sections.append(f"  - `{o.get('id')}`: {o.get('name')} ({o.get('visual_details')})")

        if concepts:
            sections.append("- **Concepts & Diagrams**:")
            for cp in concepts:
                sections.append(f"  - `{cp.get('id')}`: {cp.get('name')} ({cp.get('diagram_style')})")

        sections.append("\n#### Art Direction:")
        sections.append(f"- **Visual Genre**: {art_dir.get('visual_genre')}")
        sections.append(f"- **Lighting Philosophy**: {art_dir.get('lighting_philosophy')}")
        sections.append(f"- **Rendering Texture**: {art_dir.get('rendering_texture')}")
        sections.append(f"- **Color Palette**: {', '.join(str(p) for p in (art_dir.get('primary_palette_hex') or []))}")
        sections.append(f"- **Negative Anchors**: {', '.join(str(n) for n in (art_dir.get('negative_anchors') or []))}")

        sections.append(
            "\n#### Directorial Intent Instructions:\n"
            "- Bind entities using exact catalog IDs in `entity_bindings`.\n"
            "- Set `camera` framing, angle, and focal point.\n"
            "- If characters exist, specify `character_state` (pose, expression, action).\n"
            "- Motivate transitions using `transition_from_previous`.\n"
            "- For standalone conceptual diagrams, set `concept_id` and leave `character_id` and `environment_id` as null.\n"
            "- You do NOT need to write an `image_prompt`. Application code composes the final prompt from the Video Bible."
        )

        cd_char_count = (
            creative_direction_data.get("protagonist_config", {}).get("character_count")
            if creative_direction_data and isinstance(creative_direction_data.get("protagonist_config"), dict)
            else None
        )
        if (not chars) or (str(cd_char_count).lower() == "none"):
            sections.append(
                "\n> **CRITICAL CHARACTER RULE:** Character count is NONE. No characters exist in the Video Bible.\n"
                "> You MUST NOT introduce, invent, or name any protagonist, person, or character.\n"
                "> `character_id` and `character_state` must be `null` across all beats.\n"
                "> Use other visual approaches instead: first-person POV, environment details, object close-ups, macro photography, physical phenomena, or conceptual diagrams."
            )

        sections.append(
            "\n> **SCIENTIFIC ACCURACY & UNCERTAINTY RULE:**\n"
            "> Preserve scientific uncertainty from the approved script.\n"
            "> If the script frames an explanation as a hypothesis, leading theory, or possibility (e.g. 'one leading theory', 'scientists believe', 'might be', 'could be', 'we don't fully know yet'), "
            "> DO NOT turn it into an established, absolute fact. Keep visual metaphors and diagram descriptions aligned with the script's epistemic status (e.g. theoretical model, illustrative hypothesis)."
        )

    if channel_name or channel_niche:
        sections.append("\n### Channel Context")
        if channel_name:
            sections.append(f"- **Channel Name**: {channel_name}")
        if channel_niche:
            sections.append(f"- **Niche**: {channel_niche}")

    if not video_bible_data:
        sections.append("\n### Target Visual Style (Context Only)")
        sections.append(
            f"The visual style applied during rendering will be:\n\"{visual_style}\"\n"
            "IMPORTANT: Do NOT include these style descriptors in your image prompts. "
            "Describe ONLY the physical subject, mechanism, or action."
        )

    if previous_storyboard and revision_instruction:
        sections.append("\n## REVISION MODE")
        sections.append(
            "A previous draft of the storyboard exists. Modify ONLY what the revision instruction requests. "
            "Never alter the narration text or scene wording. Keep subject_anchor and every uses_anchor flag "
            "unchanged unless the revision instruction explicitly asks to change them. Preserve scene continuity, "
            "timing, and shot types unless the revision explicitly requests a change."
        )
        sections.append(f"- **Revision Instruction**: {revision_instruction}")
        sections.append(f"- **Previous Storyboard**: {json.dumps(previous_storyboard, indent=2)}")

    sections.append(
        "\n## Instructions\n"
        "Generate a structured storyboard that strictly conforms to the StoryboardLLMOutput schema."
    )

    return "\n".join(sections)


def generate_storyboard_step(
    video: Video,
    prev: dict[str, Any],
    step: VideoStep,
) -> dict[str, Any]:
    """Pipeline step handler for generating structured video storyboards.

    Signature conforms to pipeline handler interface: (video, prev, step) -> dict.
    """
    # 1. Acquire DB session if bound to model (clean fallback for CLI)
    db = None
    try:
        db = object_session(video)
    except Exception:
        db = None

    # 2. Extract previous script output (nested shape: prev["script"]["script"])
    script_step = prev.get("script")
    if not isinstance(script_step, dict):
        raise PipelineError("Script step must be completed before generating a storyboard")

    script_data = script_step.get("script")
    if not isinstance(script_data, dict) or not script_data.get("narration"):
        raise PipelineError("Script step must be completed before generating a storyboard")

    script_narration: str = script_data["narration"]

    # Extract upstream creative direction and video bible (present in 8-step pipeline)
    cd_step = prev.get("creative_direction")
    cd_data = cd_step.get("creative_direction") if isinstance(cd_step, dict) else None

    vb_step = prev.get("video_bible")
    vb_data = vb_step.get("video_bible") if isinstance(vb_step, dict) else None

    # 3. Resolve Visual Style & Style Source
    channel = getattr(video, "channel", None)
    channel_settings = getattr(channel, "settings", None) if channel else None
    channel_name = getattr(channel, "name", None) if channel else None
    channel_niche = getattr(channel, "niche", None) if channel else None

    visual_style: str
    style_source: Literal["channel", "default"]
    if (
        isinstance(channel_settings, dict)
        and isinstance(channel_settings.get("visual_style"), str)
        and channel_settings["visual_style"].strip()
    ):
        visual_style = channel_settings["visual_style"].strip()
        style_source = "channel"
    else:
        visual_style = load_default_visual_style()
        style_source = "default"

    # 4. Check Revision Mode (previous storyboard comes ONLY from step.output["storyboard"])
    is_revision = False
    previous_storyboard = None
    if step.user_prompt and isinstance(step.output, dict) and step.output.get("storyboard"):
        is_revision = True
        previous_storyboard = step.output["storyboard"]

    # 5. Load prompt rules template and version hash
    rules_text, prompt_version = load_prompt_template()

    # 6. Build prompt
    prompt = build_storyboard_prompt(
        rules_text=rules_text,
        topic=getattr(video, "topic", "Video Topic"),
        script_data=script_data,
        visual_style=visual_style,
        channel_name=channel_name,
        channel_niche=channel_niche,
        previous_storyboard=previous_storyboard,
        revision_instruction=step.user_prompt,
        creative_direction_data=cd_data,
        video_bible_data=vb_data,
    )

    # 7. Route LLM
    llm = get_llm(task="storyboard")

    total_in_tokens = 0
    total_out_tokens = 0
    total_think_tokens: int | None = None
    total_latency = 0.0
    attempts = 1
    raw_scenes: list[SceneInput]
    subject_anchor: str = ""
    environment_anchor: str = ""

    if llm.provider_name == "mock":
        # Deterministic local splitter for mock provider with zero AI cost
        raw_scenes, subject_anchor = split_script_for_mock(
            script_narration,
            video_bible_data=vb_data,
            creative_direction_data=cd_data,
        )
        environment_anchor = DEFAULT_MOCK_ENVIRONMENT_ANCHOR
        total_in_tokens = max(1, len(prompt.split()))
        total_out_tokens = max(25, 20 * len(raw_scenes))
        total_latency = 0.001
        warnings = validate_storyboard_soft_checks(
            raw_scenes,
            script_narration,
            subject_anchor=subject_anchor,
            visual_style=visual_style,
            environment_anchor=environment_anchor,
        )
    else:
        try:
            result = llm.generate_structured(
                prompt=prompt,
                response_model=StoryboardLLMOutput,
                task="storyboard",
            )
        except Exception as exc:
            raise PipelineError(f"Storyboard generation failed: {exc}") from exc

        output_obj: StoryboardLLMOutput = result.output
        raw_scenes = output_obj.scenes
        subject_anchor = output_obj.subject_anchor or ""
        environment_anchor = getattr(output_obj, "environment_anchor", "") or ""
        total_in_tokens += result.input_tokens or 0
        total_out_tokens += result.output_tokens or 0
        if result.thinking_tokens is not None:
            total_think_tokens = (total_think_tokens or 0) + result.thinking_tokens
        total_latency += result.latency_seconds

        # Soft checks on Attempt 1
        warnings = validate_storyboard_soft_checks(
            raw_scenes,
            script_narration,
            subject_anchor=subject_anchor,
            visual_style=visual_style,
            environment_anchor=environment_anchor,
        )

        # Retry once if soft checks fail
        if warnings:
            attempts = 2
            retry_prompt = (
                f"{prompt}\n\n"
                f"[IMPORTANT FEEDBACK: Your previous output had the following soft-check issues:\n"
                + "\n".join(f"- {w}" for w in warnings)
                + "\nPlease regenerate the structured output, strictly fixing these issues while maintaining the StoryboardLLMOutput schema.]"
            )
            try:
                retry_result = llm.generate_structured(
                    prompt=retry_prompt,
                    response_model=StoryboardLLMOutput,
                    task="storyboard",
                )
                output_obj = retry_result.output
                raw_scenes = output_obj.scenes
                subject_anchor = output_obj.subject_anchor or subject_anchor
                environment_anchor = getattr(output_obj, "environment_anchor", "") or environment_anchor
                total_in_tokens += retry_result.input_tokens or 0
                total_out_tokens += retry_result.output_tokens or 0
                if retry_result.thinking_tokens is not None:
                    total_think_tokens = (total_think_tokens or 0) + retry_result.thinking_tokens
                total_latency += retry_result.latency_seconds
                # Re-evaluate warnings for Attempt 2
                warnings = validate_storyboard_soft_checks(
                    raw_scenes,
                    script_narration,
                    subject_anchor=subject_anchor,
                    visual_style=visual_style,
                    environment_anchor=environment_anchor,
                )
            except Exception:
                # If retry fails, retain original output and warnings
                pass

    if not raw_scenes:
        raise PipelineError("Storyboard generation produced no scenes")

    # 8. Compute scene indices, estimated duration, and code-composed final prompts
    final_scenes: list[dict[str, Any]] = []
    accumulated_seconds = 0.0
    for idx, s in enumerate(raw_scenes, start=1):
        words = s.narration.split()
        est_sec = round(len(words) / WORDS_PER_SECOND, 1)
        accumulated_seconds += est_sec

        prev_s = raw_scenes[idx - 2] if idx > 1 else None
        next_s = raw_scenes[idx] if idx < len(raw_scenes) else None

        final_prompt = compose_final_prompt(
            scene=s,
            subject_anchor=subject_anchor,
            visual_style=visual_style,
            environment_anchor=environment_anchor,
            video_topic=getattr(video, "topic", None),
            video_title=getattr(video, "title", None),
            prev_scene=prev_s,
            next_scene=next_s,
            video_bible=vb_data,
            creative_direction=cd_data,
        )

        # Extract or resolve animation plan for this scene
        anim_data = getattr(s, "animation", None)
        if anim_data is not None:
            if hasattr(anim_data, "model_dump"):
                anim_plan = anim_data.model_dump()
            elif hasattr(anim_data, "dict"):
                anim_plan = anim_data.dict()
            else:
                anim_plan = dict(anim_data)
        else:
            anim_plan = plan_scene_animation(
                scene_data={
                    "shot_type": s.shot_type,
                    "focal_point": getattr(s, "focal_point", "") or "",
                    "camera_angle": getattr(s, "camera_angle", "eye-level") or "eye-level",
                    "visual_purpose": getattr(s, "visual_purpose", "") or "",
                    "image_prompt": s.image_prompt,
                    "narration": s.narration,
                },
                duration_seconds=est_sec,
            ).to_dict()

        # Extract and normalize visual beats for this scene
        raw_beats = getattr(s, "visual_beats", None) or []
        normalized_beats = normalize_scene_visual_beats(
            raw_beats=raw_beats,
            scene_index=idx,
            fallback_scene=s,
        )

        # Extract directorial intent fields
        raw_eb = getattr(s, "entity_bindings", None)
        eb_dict = raw_eb.model_dump() if hasattr(raw_eb, "model_dump") else (dict(raw_eb) if isinstance(raw_eb, dict) else None)

        raw_cam = getattr(s, "camera", None)
        cam_dict = raw_cam.model_dump() if hasattr(raw_cam, "model_dump") else (dict(raw_cam) if isinstance(raw_cam, dict) else None)

        raw_cs = getattr(s, "character_state", None)
        cs_dict = raw_cs.model_dump() if hasattr(raw_cs, "model_dump") else (dict(raw_cs) if isinstance(raw_cs, dict) else None)

        raw_fc = getattr(s, "foley_cues", []) or []
        fc_list = [f.model_dump() if hasattr(f, "model_dump") else dict(f) for f in raw_fc]

        final_scenes.append(
            {
                "index": idx,
                "narration": s.narration,
                "visual_purpose": getattr(s, "visual_purpose", "") or "",
                "shot_type": s.shot_type,
                "camera_angle": getattr(s, "camera_angle", "eye-level") or "eye-level",
                "focal_point": getattr(s, "focal_point", "") or "",
                "visual_technique": getattr(s, "visual_technique", "cinematic-photo") or "cinematic-photo",
                "environment": getattr(s, "environment", "") or "",
                "image_prompt": getattr(s, "image_prompt", "") or final_prompt,
                "uses_anchor": bool(s.uses_anchor),
                "final_prompt": final_prompt,
                "est_seconds": est_sec,
                "animation": anim_plan,
                "visual_beats": [b.model_dump() for b in normalized_beats],
                "entity_bindings": eb_dict,
                "camera": cam_dict,
                "character_state": cs_dict,
                "environment_state_override": getattr(s, "environment_state_override", None),
                "transition_from_previous": getattr(s, "transition_from_previous", "establishing") or "establishing",
                "continuity_intent": getattr(s, "continuity_intent", "") or "",
                "foley_cues": fc_list,
            }
        )
    total_seconds = round(accumulated_seconds, 1)
    total_visual_beats = sum(len(sc["visual_beats"]) for sc in final_scenes)

    # 8b. Construct flat typed StoryboardBeat models for connected storyboard
    flat_beats: list[dict[str, Any]] = []
    for sc in final_scenes:
        for b in sc.get("visual_beats", []):
            eb_raw = b.get("entity_bindings") or sc.get("entity_bindings") or {}
            eb_obj = EntityBindings.model_validate(eb_raw) if eb_raw else EntityBindings()

            valid_cam_shots = ("extreme_close_up", "close_up", "medium_close_up", "medium", "wide", "extreme_wide")
            valid_cam_angles = ("eye_level", "low_angle", "high_angle", "top_down", "side_profile", "dutch_angle")
            cam_raw = b.get("camera") or sc.get("camera") or {}
            if cam_raw:
                cam_obj = CameraDirection.model_validate(cam_raw)
            else:
                raw_st = str(sc.get("shot_type", "medium")).replace("-", "_")
                raw_ca = str(sc.get("camera_angle", "eye_level")).replace("-", "_")
                cam_obj = CameraDirection(
                    shot_type=raw_st if raw_st in valid_cam_shots else "medium",
                    camera_angle=raw_ca if raw_ca in valid_cam_angles else "eye_level",
                    focal_point_description=str(sc.get("focal_point", "")),
                )

            cs_raw = b.get("character_state") or sc.get("character_state")
            cs_obj = CharacterState.model_validate(cs_raw) if cs_raw else None

            valid_transitions = (
                "establishing",
                "same_scene_reframe",
                "action_reaction",
                "spatial_cutaway",
                "location_change",
                "explanatory_cutaway",
                "surreal_metaphor",
                "resolution_return",
            )
            raw_trans = str(b.get("transition_from_previous") or sc.get("transition_from_previous") or "same_scene_reframe")
            trans_clean = raw_trans if raw_trans in valid_transitions else "same_scene_reframe"

            sb_beat = StoryboardBeat(
                beat_id=b.get("id") or f"beat_{sc['index']}_{b.get('order', 1)}",
                scene_index=sc["index"],
                order=b.get("order", 1),
                narration_slice=b.get("narration_text") or sc.get("narration", ""),
                duration_est_seconds=float(b.get("duration_seconds") or sc.get("est_seconds", 3.0)),
                entity_bindings=eb_obj,
                transition_from_previous=trans_clean,
                continuity_intent=b.get("continuity_intent") or sc.get("continuity_intent", ""),
                camera=cam_obj,
                character_state=cs_obj,
                environment_state_override=b.get("environment_state_override") or sc.get("environment_state_override"),
                foley_cues=[FoleyCue.model_validate(fc) for fc in b.get("foley_cues", [])],
            )
            flat_beats.append(sb_beat.model_dump())

    # 8c. Run Pre-Image Connected Continuity & Progression Validation
    if vb_data:
        continuity_warnings = validate_connected_storyboard(
            scenes=final_scenes,
            video_bible=vb_data,
            creative_direction=cd_data,
        )
        warnings.extend(continuity_warnings)

    # 9. Return structured payload
    return {
        "storyboard": {
            "visual_style": visual_style,
            "style_source": style_source,
            "subject_anchor": subject_anchor,
            "environment_anchor": environment_anchor,
            "scenes": final_scenes,
            "beats": flat_beats,
            "video_bible": vb_data,
            "total_seconds": total_seconds,
            "total_visual_beats": total_visual_beats,
        },
        "warnings": warnings,
        "meta": {
            "prompt_version": prompt_version,
            "provider": llm.provider_name,
            "model": llm.model,
            "task": "storyboard",
            "input_tokens": total_in_tokens,
            "output_tokens": total_out_tokens,
            "reasoning_tokens": total_think_tokens,
            "latency_seconds": round(total_latency, 4),
            "attempts": attempts,
            "revision": is_revision,
        },
    }
