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

from app.llm import LLMError, get_llm
from app.models import Video, VideoStep
from app.steps.prompt_compose import (
    COMPOSITION_RULES,
    MAX_FINAL_PROMPT_LENGTH,
    compose_final_prompt,
    recompute_final_prompts,
)

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

PROMPT_FILE_PATH = Path(__file__).resolve().parent.parent / "prompts" / "storyboard.md"
DEFAULT_STYLE_PATH = Path(__file__).resolve().parent.parent / "prompts" / "default_visual_style.txt"


class PipelineError(Exception):
    """Raised when pipeline preconditions or unrecoverable step errors occur."""


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
    image_prompt: str = Field(
        description="Rich scene-specific visual description specifying subject, environment, lighting, composition, and focal point. No text, logos, or watermarks."
    )
    uses_anchor: bool = Field(
        default=False,
        description="True ONLY if the recurring anchor object is the central physical subject of this specific shot. Set to false for establishing shots, interior POV, diagrams, cutaways, etc.",
    )


class StoryboardLLMOutput(BaseModel):
    """Structured storyboard output returned by the LLM."""

    subject_anchor: str = Field(
        default="",
        description=(
            "One or two plain sentences describing the main recurring object(s) of the video: "
            "shape, colors, materials, size cues. No style words, no text, letters, numbers, logos, brands or people."
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

        # Image prompt checks: non-empty, max 300 chars, no forbidden words
        if not img_prompt or not str(img_prompt).strip():
            warnings.append(f"Scene {idx} image prompt is empty.")
        else:
            if len(img_prompt) > 300:
                warnings.append(
                    f"Scene {idx} image prompt length is {len(img_prompt)} chars (target: at most 300 chars)."
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
            )

        if not resolved_final_prompt or not str(resolved_final_prompt).strip():
            warnings.append(f"Scene {idx} final prompt is empty.")
        elif len(str(resolved_final_prompt)) > MAX_FINAL_PROMPT_LENGTH:
            warnings.append(
                f"Scene {idx} final prompt length is {len(str(resolved_final_prompt))} chars (target: at most {MAX_FINAL_PROMPT_LENGTH} chars)."
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
    },
    {
        "purpose": "Direct attention to specific focus",
        "shot_type": "medium",
        "camera_angle": "exterior 45-degree angle",
        "technique": "cinematic-photo",
        "focal_point": "highlighted passenger window array",
        "prompt": "Exterior medium framing directing viewer focus towards the key component with focused illumination",
        "uses_anchor": True,
    },
    {
        "purpose": "Reveal tiny detail and texture",
        "shot_type": "macro",
        "camera_angle": "extreme close-up 90-degree",
        "technique": "macro-photo",
        "focal_point": "miniature bleed opening and texture",
        "prompt": "Macro extreme close-up revealing fine physical detail and miniature circular opening in clear material",
        "uses_anchor": True,
    },
    {
        "purpose": "Interior passenger perspective",
        "shot_type": "interior",
        "camera_angle": "passenger-perspective eye-level",
        "technique": "passenger-pov",
        "focal_point": "cabin frame looking outward",
        "prompt": "First-person passenger perspective from inside cabin looking past contoured bezel to distant sky",
        "uses_anchor": False,
    },
    {
        "purpose": "Explain construction and layers",
        "shot_type": "cutaway",
        "camera_angle": "isometric technical cutaway",
        "technique": "technical-illustration",
        "focal_point": "three distinct separated layer assembly",
        "prompt": "Exploded technical cutaway diagram illustrating the three distinct structural layers in separation",
        "uses_anchor": False,
    },
    {
        "purpose": "Schematic diagram of functional forces",
        "shot_type": "diagram",
        "camera_angle": "side cross-section schematic",
        "technique": "schematic",
        "focal_point": "directional flow arrows balancing pressure",
        "prompt": "Cross-section schematic diagram illustrating dynamic pressure balance with directional flow arrows",
        "uses_anchor": False,
    },
    {
        "purpose": "Proof of real-world operation",
        "shot_type": "close-up",
        "camera_angle": "low-angle close-up",
        "technique": "cinematic-photo",
        "focal_point": "crystal clear surface resisting frost",
        "prompt": "Detailed close-up demonstrating clear unblemished surface remaining free of frost in subzero air",
        "uses_anchor": True,
    },
    {
        "purpose": "Visual closure and resolution",
        "shot_type": "wide",
        "camera_angle": "wide cinematic horizon",
        "technique": "cinematic-photo",
        "focal_point": "peaceful horizon in twilight sunset",
        "prompt": "Cinematic wide composition with warm twilight rim lighting bringing elegant visual closure",
        "uses_anchor": False,
    },
)


def split_script_for_mock(narration: str) -> tuple[list[SceneInput], str]:
    """Deterministically split script narration into scenes along sentence/clause boundaries.

    Aims for 8-12 scenes. If fewer than 8 sentences, splits long sentences at commas.
    If the text is too short (<8), produces as many valid scenes as possible without crashing.
    Assigns rotating shot types, clean placeholder image prompts, and sets uses_anchor by scene index
    membership (false for scenes 2 and 5, true for all others).

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

    # Step 2: If fewer than 8 chunks, iteratively split long chunks at natural grammatical boundaries
    while len(chunks) < 8:
        best_idx = -1
        best_score = -1.0
        best_split_parts: tuple[str, str] | None = None

        for idx, chunk in enumerate(chunks):
            words = chunk.split()
            if len(words) < 8:
                continue

            candidates: list[tuple[float, str, str]] = []
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
                if p2.lower().startswith(("and ", "but ", "while ", "because ", "so ", "though ")):
                    bonus += 8.0

                balance_penalty = abs(len(w1) - len(w2))
                score = 50.0 + bonus - balance_penalty
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

    # Step 4: Construct scenes with rotating story beats, shot types, and distinct visual purposes
    scenes: list[SceneInput] = []
    for i, chunk in enumerate(chunks):
        scene_idx = i + 1
        beat = MOCK_VISUAL_BEATS[i % len(MOCK_VISUAL_BEATS)]
        uses_anchor = scene_idx not in (2, 5)
        scenes.append(
            SceneInput(
                narration=chunk,
                visual_purpose=beat["purpose"],
                shot_type=beat["shot_type"],
                camera_angle=beat["camera_angle"],
                focal_point=beat["focal_point"],
                visual_technique=beat["technique"],
                image_prompt=beat["prompt"],
                uses_anchor=uses_anchor,
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
) -> str:
    """Assemble the comprehensive prompt including script context, visual style, and revision details."""
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

    if channel_name or channel_niche:
        sections.append("\n### Channel Context")
        if channel_name:
            sections.append(f"- **Channel Name**: {channel_name}")
        if channel_niche:
            sections.append(f"- **Niche**: {channel_niche}")

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

    if llm.provider_name == "mock":
        # Deterministic local splitter for mock provider with zero AI cost
        raw_scenes, subject_anchor = split_script_for_mock(script_narration)
        total_in_tokens = max(1, len(prompt.split()))
        total_out_tokens = max(25, 20 * len(raw_scenes))
        total_latency = 0.001
        warnings = validate_storyboard_soft_checks(
            raw_scenes,
            script_narration,
            subject_anchor=subject_anchor,
            visual_style=visual_style,
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

        final_prompt = compose_final_prompt(
            scene=s,
            subject_anchor=subject_anchor,
            visual_style=visual_style,
        )

        final_scenes.append(
            {
                "index": idx,
                "narration": s.narration,
                "visual_purpose": getattr(s, "visual_purpose", "") or "",
                "shot_type": s.shot_type,
                "camera_angle": getattr(s, "camera_angle", "eye-level") or "eye-level",
                "focal_point": getattr(s, "focal_point", "") or "",
                "visual_technique": getattr(s, "visual_technique", "cinematic-photo") or "cinematic-photo",
                "image_prompt": s.image_prompt,
                "uses_anchor": bool(s.uses_anchor),
                "final_prompt": final_prompt,
                "est_seconds": est_sec,
            }
        )
    total_seconds = round(accumulated_seconds, 1)

    # 9. Return structured payload
    return {
        "storyboard": {
            "visual_style": visual_style,
            "style_source": style_source,
            "subject_anchor": subject_anchor,
            "scenes": final_scenes,
            "total_seconds": total_seconds,
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
