"""Creative Direction step implementation for Shorts Studio.

Evaluates video topic, description, and channel context to establish format archetype,
visual genre, protagonist profile, emotional arc, and creative constraints before script generation.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.llm import LLMError, get_llm
from app.models import Video, VideoStep
from app.pipeline import PipelineError
from app.schemas import CreativeDirectionOutput

PROMPT_FILE_PATH = Path(__file__).resolve().parent.parent / "prompts" / "creative_direction.md"


def load_prompt_template(path: Path = PROMPT_FILE_PATH) -> tuple[str, str]:
    """Load the creative direction rules from disk and compute a version hash."""
    if not path.exists():
        raise FileNotFoundError(f"Creative direction prompt template not found at {path}")

    content = path.read_text(encoding="utf-8")
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
    return content, f"cd-rules-{content_hash}"


def build_creative_direction_prompt(
    rules_text: str,
    topic: str,
    description: str | None = None,
    channel_name: str | None = None,
    channel_niche: str | None = None,
    channel_settings: dict[str, Any] | None = None,
    previous_creative_direction: dict[str, Any] | None = None,
    revision_instruction: str | None = None,
) -> str:
    """Build the complete prompt for creative direction generation."""
    sections: list[str] = [rules_text, "\n## Project Context"]

    sections.append(f"- **Video Topic**: {topic}")
    if description:
        sections.append(f"- **Topic Description / Additional Details**: {description}")

    if channel_name or channel_niche or channel_settings:
        sections.append("\n### Channel Context")
        if channel_name:
            sections.append(f"- **Channel Name**: {channel_name}")
        if channel_niche:
            sections.append(f"- **Niche**: {channel_niche}")
        if channel_settings:
            # Avoid exposing internal secrets, pass relevant visual/brand settings
            brand_keys = {k: v for k, v in channel_settings.items() if k in ("visual_style", "tone", "target_audience")}
            if brand_keys:
                sections.append(f"- **Brand Settings**: {json.dumps(brand_keys)}")

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
        "Select the most appropriate format archetype and visual genre for this topic. "
        "Do NOT force a character if the topic is purely mechanical, cosmic, or abstract."
    )

    return "\n".join(sections)


def validate_creative_direction(cd: CreativeDirectionOutput) -> list[str]:
    """Validate creative direction consistency and return any soft warnings."""
    warnings: list[str] = []

    if cd.protagonist_config.character_count != "none" and not cd.protagonist_config.archetype_summary:
        warnings.append("Protagonist config specified characters but provided no archetype summary.")

    if len(cd.emotional_progression) < 3:
        warnings.append("Emotional progression has fewer than 3 stages; aim for a complete 5-stage emotional arc.")

    if not cd.creative_constraints:
        warnings.append("No creative constraints specified; provide 3-5 rules to prevent downstream production drift.")

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
        raise PipelineError("Video topic is required to generate creative direction")

    channel = getattr(video, "channel", None)
    channel_name = getattr(channel, "name", None) if channel else None
    channel_niche = getattr(channel, "niche", None) if channel else None
    channel_settings = getattr(channel, "settings", None) if channel else None

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
    warnings = validate_creative_direction(cd_output)

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
