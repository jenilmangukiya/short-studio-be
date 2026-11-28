"""Creative Direction step implementation for Shorts Studio.

Evaluates video topic, description, and channel context to establish format archetype,
visual genre, protagonist profile, emotional arc, and creative constraints before script generation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

from pydantic import ValidationError

from app.llm import LLMError, get_llm
from app.models import Video, VideoStep
from app.schemas import CreativeDirectionOutput
from app.steps.duration import clamp_duration_seconds


class PipelineError(Exception):
    """Raised when pipeline preconditions or execution fail."""

PROMPT_FILE_PATH = Path(__file__).resolve().parent.parent / "prompts" / "creative_direction.md"

# Targeted sensory and bodily reflex keywords for non-blocking advisory warnings
HUMAN_SENSORY_TOPIC_REGEX = re.compile(
    r"\b(?:"
    r"hypnic\s+jerk|"
    r"falling\s+asleep|"
    r"sleep\s+paralysis|"
    r"(?:why\s+(?:we|do\s+we|people))\s+(?:yawn|dream|shiver|blush|sweat)|"
    r"brain\s+freeze|"
    r"goosebumps|"
    r"deja\s+vu"
    r")\b",
    re.IGNORECASE,
)


def load_prompt_template(path: Path = PROMPT_FILE_PATH) -> tuple[str, str]:
    """Load the creative direction rules from disk and compute a version hash."""
    if not path.exists():
        raise FileNotFoundError(f"Creative direction prompt template not found at {path}")

    content = path.read_text(encoding="utf-8")
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
    return content, f"cd-rules-{content_hash}"


def filter_conflicting_channel_settings(
    channel_settings: dict[str, Any] | None,
    has_active_preset: bool = True,
) -> dict[str, Any]:
    """Filter out conflicting artistic instructions inherited from channel settings.

    Preserves unrelated channel preferences such as tone, target audience, voice speed,
    brand colors, etc., while ensuring the active visual style preset remains the single
    source of truth for artistic presentation.
    """
    if not channel_settings:
        return {}
    if not has_active_preset:
        return dict(channel_settings)

    conflicting_artistic_keys = {
        "visual_style", "style_prompt", "visual_direction", "aesthetic",
        "artistic_direction", "rendering_style", "art_style", "render_style",
        "drawing_style"
    }
    return {
        k: v for k, v in channel_settings.items()
        if k.lower() not in conflicting_artistic_keys
    }


def build_creative_direction_prompt(
    rules_text: str,
    topic: str,
    description: str | None = None,
    channel_name: str | None = None,
    channel_niche: str | None = None,
    channel_settings: dict[str, Any] | None = None,
    previous_creative_direction: dict[str, Any] | None = None,
    revision_instruction: str | None = None,
    visual_style_snapshot: dict[str, Any] | None = None,
    visual_style_id: str | None = None,
    duration_mode: str = "auto",
    target_duration: int | None = None,
) -> str:
    """Build the complete prompt for creative direction generation."""
    sections: list[str] = [rules_text, "\n## Project Context"]

    sections.append(f"- **Video Topic**: {topic}")
    if description:
        sections.append(f"- **Topic Description / Additional Details**: {description}")

    # Inject duration configuration
    sections.append("\n### Duration Configuration")
    if duration_mode == "manual" and target_duration is not None:
        sections.append(f"- **Duration Mode**: Manual (User Specified)")
        sections.append(f"- **Target Duration**: {target_duration} seconds")
        sections.append(
            f"- **Directive**: Design a narrative progression, sensory depth, and pacing tailored specifically for a {target_duration}-second short. "
            f"Set recommended_duration_seconds={target_duration} and provide your storytelling pacing rationale."
        )
    else:
        sections.append("- **Duration Mode**: Automatic (AI Storytelling Recommendation)")
        sections.append(
            "- **Directive**: Evaluate the narrative complexity, emotional arc, and educational depth of this topic. "
            "Recommend an optimal target duration between 30 and 180 seconds in `recommended_duration_seconds` (e.g., 30-45s for quick curious facts, 50-75s for focused narratives, 80-110s for multi-stage dilemmas, 120-180s for deep experiential journeys). "
            "State your rationale in `duration_rationale`."
        )

    # Inject frozen canonical visual style context if present
    if visual_style_snapshot or visual_style_id:
        sections.append("\n### Visual Style Context (Canonical Single Source of Truth)")
        style_id = visual_style_id or (visual_style_snapshot.get("id") if visual_style_snapshot else None)
        if style_id:
            sections.append(f"- **Visual Style ID**: {style_id}")
        if visual_style_snapshot:
            d_name = visual_style_snapshot.get("display_name")
            desc = visual_style_snapshot.get("description")
            art = visual_style_snapshot.get("art_direction") or {}
            if d_name:
                sections.append(f"- **Style Name**: {d_name}")
            if desc:
                sections.append(f"- **Style Summary**: {desc}")
            if art.get("rendering_texture"):
                sections.append(f"- **Rendering Texture**: {art.get('rendering_texture')}")
            if art.get("lighting_philosophy"):
                sections.append(f"- **Lighting Philosophy**: {art.get('lighting_philosophy')}")
        sections.append(
            "- **Note**: Artistic rendering and visual presentation are governed canonical by this frozen visual style preset. "
            "Creative Direction defines the narrative structure, emotional progression, storytelling angle, and human presence."
        )

    if channel_name or channel_niche or channel_settings:
        sections.append("\n### Channel Context")
        if channel_name:
            sections.append(f"- **Channel Name**: {channel_name}")
        if channel_niche:
            sections.append(f"- **Niche**: {channel_niche}")
        if channel_settings:
            # If a visual style preset is active, filter out conflicting artistic prompt strings from channel settings
            has_preset = bool(visual_style_snapshot or visual_style_id)
            filtered_settings = filter_conflicting_channel_settings(channel_settings, has_active_preset=has_preset)
            if filtered_settings:
                sections.append(f"- **Brand Settings**: {json.dumps(filtered_settings)}")

    if previous_creative_direction and revision_instruction:
        sections.append("\n## REVISION MODE")
        sections.append(
            "A previous version of the creative direction exists. Modify ONLY what the revision instruction requests, "
            "and preserve everything else."
        )
        sections.append(f"- **Revision Instruction**: {revision_instruction}")
        sections.append(f"- **Previous Creative Direction**: {json.dumps(previous_creative_direction, indent=2)}")

    sections.append(
        "\n## Instructions\n"
        "Generate a structured CreativeDirectionOutput conforming strictly to the schema. "
        "Select the most appropriate format archetype, storytelling angle, and emotional arc for this topic.\n"
        "Apply the balanced character and subject selection guidelines:\n"
        "- When the topic explores human bodily sensations, sleep, dreams, emotions, or physical reflexes, strongly consider defining a relatable human character/subject.\n"
        "- When the topic involves historical crises, disasters, or societal impacts affecting people, consider human_presence='contextual' to represent witnesses, workers, or communities without forcing a recurring hero.\n"
        "- When the topic focuses on animal biology or instincts, consider an appropriate animal subject.\n"
        "- Do NOT force a character if the topic is purely mechanical, cosmic, chemical, or abstract."
    )

    return "\n".join(sections)


def validate_creative_direction(cd: CreativeDirectionOutput, topic: str = "") -> list[str]:
    """Validate creative direction consistency and return any soft warnings."""
    warnings: list[str] = []

    if cd.protagonist_config.human_presence in ("single", "ensemble") and not cd.protagonist_config.archetype_summary:
        warnings.append("Protagonist config specified characters but provided no archetype summary.")

    if len(cd.emotional_progression) < 3:
        warnings.append("Emotional progression has fewer than 3 stages; aim for a complete 5-stage emotional arc.")

    if not cd.creative_constraints:
        warnings.append("No creative constraints specified; provide 3-5 rules to prevent downstream production drift.")

    if cd.recommended_duration_seconds < 30 or cd.recommended_duration_seconds > 180:
        warnings.append(
            f"Recommended duration {cd.recommended_duration_seconds}s is outside valid range (30-180 seconds)."
        )

    if cd.narrative_plan and len(cd.narrative_plan.story_progression) < 2:
        warnings.append("Narrative plan has fewer than 2 story progression stages.")

    # Non-blocking advisory check for human sensory/bodily topics with human_presence == "none"
    if cd.protagonist_config.human_presence == "none":
        context_str = f"{topic} {cd.central_thesis} {cd.storytelling_angle}".strip()
        if HUMAN_SENSORY_TOPIC_REGEX.search(context_str):
            warnings.append(
                "Topic involves a human physical or sensory experience, but human_presence is set to 'none'. "
                "Consider whether a relatable human subject would improve visual storytelling."
            )

    return warnings


def generate_creative_direction_step(
    video: Video,
    prev: dict[str, Any],
    step: VideoStep,
) -> dict[str, Any]:
    """Pipeline step handler for generating Creative Direction.

    Signature conforms to pipeline handler interface: (video, prev, step) -> dict.
    """
    topic = getattr(video, "topic", "")
    if not topic or not topic.strip():
        from app.pipeline import PipelineError as AppPipelineError
        raise AppPipelineError("Video topic is required to generate creative direction")

    channel = getattr(video, "channel", None)
    channel_name = getattr(channel, "name", None) if channel else None
    channel_niche = getattr(channel, "niche", None) if channel else None
    channel_settings = getattr(channel, "settings", None) if channel else None
    duration_mode = getattr(video, "duration_mode", "auto")
    target_duration = getattr(video, "target_duration", None)

    # Check revision mode
    previous_cd = None
    if step.user_prompt and isinstance(step.output, dict) and step.output.get("creative_direction"):
        previous_cd = step.output["creative_direction"]

    rules_text, prompt_version = load_prompt_template()

    prompt = build_creative_direction_prompt(
        rules_text=rules_text,
        topic=topic.strip(),
        description=getattr(video, "description", None),
        channel_name=channel_name,
        channel_niche=channel_niche,
        channel_settings=channel_settings,
        previous_creative_direction=previous_cd,
        revision_instruction=step.user_prompt,
        visual_style_snapshot=getattr(video, "visual_style_snapshot", None),
        visual_style_id=getattr(video, "visual_style_id", None),
        duration_mode=duration_mode,
        target_duration=target_duration,
    )

    llm = get_llm(task="creative_direction")

    try:
        result = llm.generate_structured(
            prompt=prompt,
            response_model=CreativeDirectionOutput,
            task="creative_direction",
        )
    except Exception as exc:
        raise LLMError(f"Creative direction generation failed: {exc}") from exc

    cd_output: CreativeDirectionOutput = result.output

    # Enforce canonical duration resolution and manual precedence
    cd_output.recommended_duration_seconds = clamp_duration_seconds(cd_output.recommended_duration_seconds)
    if duration_mode == "manual" and target_duration is not None:
        cd_output.resolved_target_duration = clamp_duration_seconds(target_duration)
    else:
        cd_output.resolved_target_duration = clamp_duration_seconds(cd_output.recommended_duration_seconds)

    warnings = validate_creative_direction(cd_output, topic=topic)

    return {
        "creative_direction": cd_output.model_dump(),
        "warnings": warnings,
        "meta": {
            "provider": llm.provider_name,
            "model": getattr(llm, "model", "mock"),
            "task": "creative_direction",
            "prompt_version": prompt_version,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_seconds": round(result.latency_seconds, 3),
        },
    }
