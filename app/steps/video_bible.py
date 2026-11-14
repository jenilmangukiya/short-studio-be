"""Video Bible step implementation for Shorts Studio.

Consumes creative direction and the locked script to synthesize a persistent production blueprint:
Entity Catalog (characters, environments, objects, concepts), art direction, cinematography safe zones,
audio direction, and subtitle styling.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.llm import LLMError, get_llm
from app.models import Video, VideoStep
from app.pipeline import PipelineError
from app.schemas import VideoBibleOutput

PROMPT_FILE_PATH = Path(__file__).resolve().parent.parent / "prompts" / "video_bible.md"


def load_prompt_template(path: Path = PROMPT_FILE_PATH) -> tuple[str, str]:
    """Load the video bible rules from disk and compute a version hash."""
    if not path.exists():
        raise FileNotFoundError(f"Video Bible prompt template not found at {path}")

    content = path.read_text(encoding="utf-8")
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]
    return content, f"vb-rules-{content_hash}"


def build_video_bible_prompt(
    rules_text: str,
    topic: str,
    script_data: dict[str, Any],
    creative_direction_data: dict[str, Any] | None = None,
    channel_visual_style: str | None = None,
    previous_bible: dict[str, Any] | None = None,
    revision_instruction: str | None = None,
) -> str:
    """Build the prompt for Video Bible generation."""
    sections: list[str] = [rules_text, "\n## Project Ground Truth"]

    sections.append(f"- **Video Topic**: {topic}")

    narration = script_data.get("narration", "")
    hook = script_data.get("hook", "")
    title = script_data.get("title", "")
    sections.append(f"- **Video Title**: {title}")
    sections.append(f"- **Opening Hook**: {hook}")
    sections.append(f"- **Locked Narration Script**:\n{narration}")

    if creative_direction_data:
        sections.append("\n### Approved Creative Direction")
        sections.append(f"- **Format Archetype**: {creative_direction_data.get('format_archetype', 'explainer')}")
        sections.append(f"- **Narrative Angle**: {creative_direction_data.get('storytelling_angle', '')}")
        sections.append(f"- **Central Thesis**: {creative_direction_data.get('central_thesis', '')}")
        sections.append(f"- **Tone**: {creative_direction_data.get('tone', 'conversational')}")
        sections.append(f"- **Visual Genre**: {creative_direction_data.get('visual_genre', 'styled_3d')}")

        protagonist = creative_direction_data.get("protagonist_config", {})
        char_count = protagonist.get("character_count", "none")
        sections.append(f"- **Protagonist Configuration**: {char_count}")
        if char_count != "none" and protagonist.get("archetype_summary"):
            sections.append(f"- **Character Summary**: {protagonist.get('archetype_summary')}")

        constraints = creative_direction_data.get("creative_constraints", [])
        if constraints:
            sections.append("- **Creative Constraints**:\n  " + "\n  ".join(f"- {c}" for c in constraints))

    if channel_visual_style and channel_visual_style.strip():
        sections.append(f"\n### Channel Visual Style Reference\n{channel_visual_style.strip()}")

    if previous_bible and revision_instruction:
        sections.append("\n## REVISION MODE")
        sections.append(
            "A previous version of the Video Bible exists. Modify ONLY what the revision instruction requests, "
            "and preserve all other entities, palettes, and rules."
        )
        sections.append(f"- **Revision Instruction**: {revision_instruction}")
        sections.append(f"- **Previous Video Bible**: {json.dumps(previous_bible, indent=2)}")

    sections.append(
        "\n## Instructions\n"
        "Generate a structured VideoBibleOutput adhering strictly to the schema. "
        "Create an Entity Catalog containing only entities directly utilized by the narration. "
        "Define at least one primary environment. Do NOT create characters if the creative direction specified character_count='none'."
    )

    return "\n".join(sections)


def validate_video_bible(vb: VideoBibleOutput, cd_data: dict[str, Any] | None = None) -> list[str]:
    """Validate Video Bible consistency and return any soft warnings."""
    warnings: list[str] = []

    if not vb.entities.environments:
        warnings.append("No environments defined in Entity Catalog; at least 1 primary environment is required.")

    if cd_data:
        protagonist = cd_data.get("protagonist_config", {})
        char_count = protagonist.get("character_count", "none")
        if char_count in ("single", "ensemble") and not vb.entities.characters:
            warnings.append("Creative direction specified characters, but none were defined in the Video Bible.")
        elif char_count == "none" and vb.entities.characters:
            warnings.append("Creative direction specified character_count='none', but characters were defined.")

    if len(vb.art_direction.primary_palette_hex) < 2:
        warnings.append("Fewer than 2 palette hex codes specified; provide 3-5 cohesive colors.")

    return warnings


def generate_video_bible_step(
    video: Video,
    prev: dict[str, Any],
    step: VideoStep,
) -> dict[str, Any]:
    """Pipeline step handler for generating the Video Bible.

    Signature conforms to pipeline handler interface: (video, prev, step) -> dict.
    """
    # 1. Precondition: Script must be completed
    script_step = prev.get("script")
    if not isinstance(script_step, dict):
        raise PipelineError("Script step must be completed before generating a Video Bible")

    script_data = script_step.get("script")
    if not isinstance(script_data, dict) or not script_data.get("narration"):
        raise PipelineError("Script step must be completed with valid narration before generating a Video Bible")

    # 2. Extract Creative Direction if present (optional for legacy runs, standard for 8-step flow)
    cd_step = prev.get("creative_direction")
    cd_data = cd_step.get("creative_direction") if isinstance(cd_step, dict) else None

    # 3. Channel Visual Style
    channel = getattr(video, "channel", None)
    channel_settings = getattr(channel, "settings", None) if channel else None
    channel_style = channel_settings.get("visual_style") if isinstance(channel_settings, dict) else None

    # 4. Check revision mode
    previous_bible = None
    if step.user_prompt and isinstance(step.output, dict) and step.output.get("video_bible"):
        previous_bible = step.output["video_bible"]

    rules_text, prompt_version = load_prompt_template()

    prompt = build_video_bible_prompt(
        rules_text=rules_text,
        topic=getattr(video, "topic", "Video Topic"),
        script_data=script_data,
        creative_direction_data=cd_data,
        channel_visual_style=channel_style,
        previous_bible=previous_bible,
        revision_instruction=step.user_prompt,
    )

    llm = get_llm(task="video_bible")

    try:
        result = llm.generate_structured(
            prompt=prompt,
            response_model=VideoBibleOutput,
            task="video_bible",
        )
    except Exception as exc:
        raise LLMError(f"Video Bible generation failed: {exc}") from exc

    vb_output: VideoBibleOutput = result.output
    warnings = validate_video_bible(vb_output, cd_data=cd_data)

    return {
        "video_bible": vb_output.model_dump(),
        "warnings": warnings,
        "meta": {
            "provider": llm.provider_name,
            "model": getattr(llm, "model", "mock"),
            "task": "video_bible",
            "prompt_version": prompt_version,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_seconds": round(result.latency_seconds, 3),
        },
    }
