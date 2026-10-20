"""Script step implementation for Shorts Studio.

Generates structured video script outputs using task-routed LLMs (Mock, Gemini, OpenAI).
Includes code-side validation for duration, formatting, and angle diversity.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, object_session

from app.llm import LLMError, get_llm
from app.models import Video, VideoStep

# Single source of truth for allowed angles
ALLOWED_ANGLES: tuple[str, ...] = (
    "why it's designed this way",
    "myth vs reality",
    "origin story",
    "what goes wrong without it",
    "hidden feature you've ignored",
    "numbers that surprise",
)

AngleType = Literal[
    "why it's designed this way",
    "myth vs reality",
    "origin story",
    "what goes wrong without it",
    "hidden feature you've ignored",
    "numbers that surprise",
]

PROMPT_FILE_PATH = Path(__file__).resolve().parent.parent / "prompts" / "script.md"

EMOJI_PATTERN = re.compile(
    r"[\U00010000-\U0010ffff"
    r"\U00002600-\U000027BF"
    r"\U0001F300-\U0001F9FF"
    r"\U0001FA00-\U0001FAFF]",
    flags=re.UNICODE,
)


class ScriptOutput(BaseModel):
    """Structured narration script output for a short-form video."""

    title: str = Field(
        description="Short video title, up to about 60 characters, no ALL CAPS, no misleading claims"
    )
    hook: str = Field(
        description="Attention-grabbing opening hook, exactly ONE sentence, up to about 15 words"
    )
    narration: str = Field(
        description="The full spoken narration text, 110-150 words, plain spoken text only, must begin with the hook sentence"
    )
    angle: AngleType = Field(
        description="The storytelling angle chosen for this video from the allowed list"
    )
    claims_to_verify: list[str] = Field(
        description="1 to 6 standalone factual claims made in the narration that require verification"
    )
    hashtags: list[str] = Field(
        description="3 to 5 relevant hashtags for the video"
    )


def load_prompt_template(path: Path = PROMPT_FILE_PATH) -> tuple[str, str]:
    """Load the writing rules from disk and derive a version string with a content hash.

    Returns:
        tuple of (prompt_text, prompt_version_identifier)
    """
    if not path.exists():
        raise FileNotFoundError(f"Script prompt template not found at {path}")

    content = path.read_text(encoding="utf-8")
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:8]

    # Optional version label from first line (e.g. 'version: script-v1' or '# script-v1')
    version_label = "script-v1"
    first_line = content.splitlines()[0].strip() if content.splitlines() else ""
    if first_line.lower().startswith("version:"):
        version_label = first_line.split(":", 1)[1].strip()
    elif first_line.startswith("#"):
        version_label = first_line.lstrip("#").strip().split()[0]

    prompt_version = f"{version_label}:{content_hash}"
    return content, prompt_version


def normalize_for_comparison(text: str) -> str:
    """Normalize whitespace, dashes, and smart quotes for reliable matching."""
    t = text.strip()
    # Replace curly/smart quotes with standard quotes
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    # Normalize dashes and hyphens
    t = t.replace("—", "-").replace("–", "-")
    # Collapse multiple whitespace
    t = re.sub(r"\s+", " ", t)
    return t


def validate_script_soft_checks(script: ScriptOutput) -> list[str]:
    """Execute code-side soft checks on the generated script output.

    Returns a list of warning messages for any failed checks.
    """
    warnings: list[str] = []

    # 1. Narration word count (110 - 150 words)
    words = script.narration.split()
    word_count = len(words)
    if word_count < 110 or word_count > 150:
        warnings.append(
            f"Narration word count is {word_count} words (target: 110-150 words)."
        )

    # 2. Hook length (<= 15 words) and narration start
    hook_words = script.hook.split()
    if len(hook_words) > 15:
        warnings.append(
            f"Hook length is {len(hook_words)} words (target: at most 15 words)."
        )

    norm_hook = normalize_for_comparison(script.hook).rstrip(".!? ")
    norm_narr = normalize_for_comparison(script.narration)
    if not norm_narr.startswith(norm_hook):
        warnings.append("Narration does not begin with the hook sentence.")

    # 3. No bracketed directions or parentheses
    if re.search(r"\[.*?\]|\(.*?\)", script.narration):
        warnings.append("Narration contains bracketed directions or parentheses.")

    # 4. No markdown formatting characters
    if re.search(r"[*#_`]", script.narration):
        warnings.append("Narration contains markdown formatting characters.")

    # 5. No emojis
    if EMOJI_PATTERN.search(script.narration):
        warnings.append("Narration contains emoji characters.")

    # 6. Angle membership
    if script.angle not in ALLOWED_ANGLES:
        warnings.append(
            f"Angle '{script.angle}' is not in the allowed angles list: {list(ALLOWED_ANGLES)}."
        )

    # 7. Claims to verify count (1 - 6)
    claims_count = len(script.claims_to_verify)
    if claims_count < 1 or claims_count > 6:
        warnings.append(
            f"Claims count is {claims_count} (target: 1-6 standalone claims)."
        )

    # 8. Hashtags count (3 - 5)
    tags_count = len(script.hashtags)
    if tags_count < 3 or tags_count > 5:
        warnings.append(
            f"Hashtags count is {tags_count} (target: 3-5 tags)."
        )

    return warnings


def get_recent_angles(
    db: Session | None, channel_id: Any, current_video_id: Any, limit: int = 8
) -> list[str]:
    """Retrieve up to `limit` angles from previous completed scripts for this channel."""
    if db is None or not channel_id:
        return []

    try:
        stmt = (
            select(VideoStep.output)
            .join(Video, VideoStep.video_id == Video.id)
            .where(
                Video.channel_id == channel_id,
                Video.id != current_video_id,
                VideoStep.name == "script",
                VideoStep.status == "done",
            )
            .order_by(VideoStep.finished_at.desc().nullslast(), VideoStep.created_at.desc())
            .limit(limit)
        )
        rows = db.scalars(stmt).all()
        recent_angles: list[str] = []
        for out in rows:
            if isinstance(out, dict):
                # Stored shape: {"script": {"angle": ...}} or legacy {"angle": ...}
                script_data = out.get("script")
                if isinstance(script_data, dict) and script_data.get("angle"):
                    recent_angles.append(script_data["angle"])
                elif out.get("angle"):
                    recent_angles.append(out["angle"])
        return recent_angles
    except Exception:
        return []


def build_script_prompt(
    rules_text: str,
    topic: str,
    description: str | None,
    channel_name: str | None = None,
    channel_niche: str | None = None,
    channel_settings: dict | None = None,
    recent_angles: list[str] | None = None,
    previous_script: dict | None = None,
    revision_instruction: str | None = None,
) -> str:
    """Assemble the comprehensive prompt including channel context, constraints, and revision details."""
    sections: list[str] = [rules_text.strip(), "\n## Video Project Details"]

    sections.append(f"- **Video Topic**: {topic}")
    if description:
        sections.append(f"- **Topic Description / Context**: {description}")

    if channel_name or channel_niche or channel_settings:
        sections.append("\n### Channel Context")
        if channel_name:
            sections.append(f"- **Channel Name**: {channel_name}")
        if channel_niche:
            sections.append(f"- **Niche**: {channel_niche}")
        if channel_settings:
            sections.append(f"- **Channel Settings**: {json.dumps(channel_settings)}")

    if recent_angles:
        angles_list = ", ".join(f"'{a}'" for a in recent_angles)
        sections.append(
            f"\n### Angle Selection Guidance\n"
            f"The following angles were recently used in this channel: {angles_list}.\n"
            f"Please avoid repeating these recent angles if another allowed angle fits the topic."
        )

    if previous_script and revision_instruction:
        sections.append("\n## REVISION MODE")
        sections.append(
            "A previous draft of the script exists. Modify ONLY what the revision instruction requests, "
            "and preserve everything else (tone, verified facts, structure)."
        )
        sections.append(f"- **Revision Instruction**: {revision_instruction}")
        sections.append(f"- **Previous Script**: {json.dumps(previous_script, indent=2)}")

    sections.append(
        "\n## Instructions\n"
        "Generate a structured script that strictly conforms to the ScriptOutput schema."
    )

    return "\n".join(sections)


def generate_script_step(
    video: Video,
    prev: dict[str, Any],
    step: VideoStep,
) -> dict[str, Any]:
    """Pipeline step handler for generating video scripts.

    Signature conforms to pipeline handler interface: (video, prev, step) -> dict.
    """
    # 1. Acquire DB session if bound to model (clean fallback for CLI)
    db = None
    try:
        db = object_session(video)
    except Exception:
        db = None

    # 2. Extract Channel details
    channel = getattr(video, "channel", None)
    channel_id = getattr(video, "channel_id", None)
    channel_name = getattr(channel, "name", None) if channel else None
    channel_niche = getattr(channel, "niche", None) if channel else None
    channel_settings = getattr(channel, "settings", None) if channel else None

    # 3. Retrieve recent angles
    recent_angles = get_recent_angles(
        db=db, channel_id=channel_id, current_video_id=video.id, limit=8
    )

    # 4. Check Revision Mode (previous script comes ONLY from step.output)
    is_revision = False
    previous_script = None
    if step.user_prompt and isinstance(step.output, dict) and step.output.get("script"):
        is_revision = True
        previous_script = step.output["script"]

    # 5. Load writing prompt template
    rules_text, prompt_version = load_prompt_template()

    # 6. Build prompt
    prompt = build_script_prompt(
        rules_text=rules_text,
        topic=video.topic,
        description=video.description,
        channel_name=channel_name,
        channel_niche=channel_niche,
        channel_settings=channel_settings,
        recent_angles=recent_angles,
        previous_script=previous_script,
        revision_instruction=step.user_prompt,
    )

    # 7. Call LLM with task routing
    llm = get_llm(task="script")

    total_in_tokens = 0
    total_out_tokens = 0
    total_think_tokens: int | None = None
    total_latency = 0.0
    attempts = 1

    try:
        result = llm.generate_structured(
            prompt=prompt,
            response_model=ScriptOutput,
            task="script",
        )
    except Exception as exc:
        raise LLMError(f"Script generation failed: {exc}") from exc

    script_output: ScriptOutput = result.output
    total_in_tokens += result.input_tokens or 0
    total_out_tokens += result.output_tokens or 0
    if result.thinking_tokens is not None:
        total_think_tokens = (total_think_tokens or 0) + result.thinking_tokens
    total_latency += result.latency_seconds

    # 8. Soft-checks validation
    warnings = validate_script_soft_checks(script_output)

    # 9. Retry once if soft checks fail and not in mock mode
    if warnings and llm.provider_name != "mock":
        attempts = 2
        retry_prompt = (
            f"{prompt}\n\n"
            f"[IMPORTANT FEEDBACK: Your previous output had the following soft-check issues:\n"
            + "\n".join(f"- {w}" for w in warnings)
            + "\nPlease regenerate the structured output, strictly fixing these issues while maintaining the ScriptOutput schema.]"
        )
        try:
            retry_result = llm.generate_structured(
                prompt=retry_prompt,
                response_model=ScriptOutput,
                task="script",
            )
            script_output = retry_result.output
            total_in_tokens += retry_result.input_tokens or 0
            total_out_tokens += retry_result.output_tokens or 0
            if retry_result.thinking_tokens is not None:
                total_think_tokens = (total_think_tokens or 0) + retry_result.thinking_tokens
            total_latency += retry_result.latency_seconds
            # Re-evaluate warnings for Attempt 2
            warnings = validate_script_soft_checks(script_output)
        except Exception:
            # If retry call fails, retain original output and its warnings
            pass

    # 10. Formulate output payload
    return {
        "script": script_output.model_dump(),
        "warnings": warnings,
        "meta": {
            "prompt_version": prompt_version,
            "provider": llm.provider_name,
            "model": llm.model,
            "task": "script",
            "input_tokens": total_in_tokens,
            "output_tokens": total_out_tokens,
            "reasoning_tokens": total_think_tokens,
            "latency_seconds": round(total_latency, 4),
            "attempts": attempts,
            "revision": is_revision,
        },
    }
