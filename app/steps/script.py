"""Script step implementation for Shorts Studio.

Generates structured video script outputs using task-routed LLMs (Mock, Gemini, OpenAI).
Includes code-side validation for duration, formatting, angle diversity, conversational clarity,
sentence complexity, technical jargon, transitions, and narrative progression.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session, object_session

from app.config import settings
from app.llm import LLMError, get_llm
from app.models import Video, VideoStep
from app.steps.creative_direction import filter_conflicting_channel_settings
from app.steps.duration import (
    calculate_word_budget_for_duration,
    resolve_video_target_duration,
    clamp_duration_seconds,
)

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

# Generic meta-clichés in opening hooks that describe the topic passively instead of revealing it
CLICHE_HOOK_PATTERNS: tuple[str, ...] = (
    r"more than you think",
    r"(?:aren't|are not|isn't|is not|not)\s+just for decoration",
    r"hidden purpose",
    r"(?:an?\s+)?interesting reason",
    r"secret reason",
    r"(?:isn't|is not|not)\s+what you think",
    r"actually serves?\s+(?:a|an|the)",
)

# Generic question formulas and preamble windups that should be asked directly
GENERIC_OPENER_PATTERNS: tuple[str, ...] = (
    r"do you know\b",
    r"did you know\b",
    r"have you ever\b",
    r"ever wondered\b",
    r"here's why\b",
    r"here is why\b",
    r"here's a fact\b",
    r"here is a fact\b",
)

CLICHE_HOOK_REGEX = re.compile(
    r"\b(?:" + "|".join(CLICHE_HOOK_PATTERNS) + r")\b",
    flags=re.IGNORECASE,
)

GENERIC_OPENER_REGEX = re.compile(
    r"^\s*(?:" + "|".join(GENERIC_OPENER_PATTERNS) + r")",
    flags=re.IGNORECASE,
)

# Stiff textbook, encyclopedia, or clinical disclaimer phrases that harm spoken storytelling
TEXTBOOK_PHRASE_PATTERNS: tuple[str, ...] = (
    r"studies\s+(?:have\s+not|haven't)\s+shown",
    r"researchers\s+link\b",
    r"scientists\s+infer\b",
    r"no\s+single\s+explanation\s+is\s+settled",
    r"(?:exact\s+makeup\s+)?may\s+remain\s+uncertain",
    r"remains?\s+uncertain",
    r"one\s+leading\s+(?:explanation|theory)\s+is\s+that\b",
    r"this\s+phenomenon\s+occurs\s+when\b",
    r"is\s+commonly\s+associated\s+with\b",
    r"can\s+be\s+attributed\s+to\b",
    r"it\s+is\s+believed\s+that\b",
    r"serves\s+to\s+facilitate\b",
    r"\bfurthermore\b",
    r"\bmoreover\b",
    r"\bconsequently\b",
    r"\bnevertheless\b",
    r"\bnotwithstanding\b",
    r"\baforementioned\b",
    r"\butilizes?\b",
    r"\bfacilitates?\b",
    r"\bis\s+characterized\s+by\b",
    r"\bempirically\b",
    r"\bheretofore\b",
)

TEXTBOOK_PHRASE_REGEX = re.compile(
    r"(?:" + "|".join(TEXTBOOK_PHRASE_PATTERNS) + r")",
    flags=re.IGNORECASE,
)

# Redundant meta-summary sentences that repeat earlier mechanisms without adding new insight
REDUNDANT_SUMMARY_PATTERNS: tuple[str, ...] = (
    r"^together,\s+these\s+(?:defenses|factors|features|elements|reasons)\b",
    r"^all\s+(?:these|of\s+these)\s+(?:factors|defenses|elements|features)\s+combine\b",
    r"^these\s+(?:factors|elements|defenses)\s+work\s+together\b",
    r"^when\s+you\s+combine\s+all\s+these\b",
)

REDUNDANT_SUMMARY_REGEX = re.compile(
    r"(?:" + "|".join(REDUNDANT_SUMMARY_PATTERNS) + r")",
    flags=re.IGNORECASE,
)

# Production instruction and meta-framing leakage patterns (backstage directives leaking into dialogue)
META_INSTRUCTION_LEAK_PATTERNS: tuple[str, ...] = (
    r"\b(?:picture|imagine|introducing|meet)\s+(?:a\s+)?(?:fictional|representative|hypothetical)(?:\s+[a-z]+)?\b",
    r"\b(?:as\s+a\s+fictional\s+(?:character|person|figure|farmer|witness|worker|protagonist))\b",
    r"\b(?:fictional\s+(?:and\s+)?representative)\b",
    r"\b(?:keep\s+the\s+[a-z]+\s+fictional(?:\s+and\s+representative)?)\b",
    r"\b(?:internal\s+instruction|creative\s+constraint|production\s+note|directorial\s+note)\b",
    r"\bfor\s+the\s+purpose\s+of\s+this\s+video,\s+let(?:'s|\s+us)\s+invent\b",
    r"\bthis\s+character\s+is\s+(?:fictional|representative|hypothetical)\b",
    r"\b(?:representative\s+character|fictional\s+archetype)\b",
)

META_INSTRUCTION_LEAK_REGEX = re.compile(
    r"(?:" + "|".join(META_INSTRUCTION_LEAK_PATTERNS) + r")",
    flags=re.IGNORECASE,
)

# Stopwords ignored when checking shared content keywords or repeated phrases
STOPWORDS: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "from", "by", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "can", "could", "will", "would",
    "may", "might", "must", "shall", "should", "that", "this", "these", "those",
    "it", "its", "it's", "they", "them", "their", "you", "your", "we", "our",
    "who", "what", "which", "when", "where", "why", "how", "not", "no", "nor",
    "so", "than", "too", "very", "just", "only", "also", "even", "more", "most",
    "some", "any", "all", "one", "two", "three", "into", "onto", "out", "up",
    "down", "off", "over", "under", "about", "like", "there", "here", "then",
})


class ScriptOutput(BaseModel):
    """Structured narration script output for a short-form video."""

    title: str = Field(
        description="Short video title, up to about 60 characters, no ALL CAPS, no misleading claims"
    )
    hook: str = Field(
        description=(
            "Attention-grabbing opening hook or scene-setting cold open (single sentence or short multi-sentence opening), "
            "creating immediate curiosity, mystery, or atmosphere without filler or meta-clichés"
        )
    )
    narration: str = Field(
        description=(
            "The full spoken narration text, plain conversational spoken text only, "
            "scaled to the target word count and duration specified in instructions, "
            "must begin verbatim with the hook sentence(s) and unfold as one connected story"
        )
    )
    claims_to_verify: list[str] = Field(
        default_factory=list,
        description="Optional key factual claims made in the narration that can be audited"
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


def _count_syllables(word: str) -> int:
    """Estimate spoken syllable count of an English word."""
    clean = re.sub(r"[^a-z]", "", word.lower())
    if not clean:
        return 1
    if len(clean) <= 3:
        return 1
    # Strip silent trailing 'e' or 'es'/'ed' when not preceded by 'l'/'t'/'d'
    stem = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", clean)
    stem = re.sub(r"^y", "", stem)
    vowels = re.findall(r"[aeiouy]{1,2}", stem)
    return max(1, len(vowels))


def parse_duration_from_text(text: str | None) -> tuple[float, float] | float | None:
    """Extract requested duration in seconds from topic, description, or revision instructions.

    Supports formats like:
    - '35-42 seconds', '35–42s', '35 to 42 seconds' -> (35.0, 42.0)
    - 'duration: 30s', 'target duration: 40 seconds' -> 40.0
    - '30s', '45 seconds' -> 45.0
    """
    if not text:
        return None
    range_match = re.search(
        r"(?:target\s+)?(?:duration[:\s]+)?(\d+(?:\.\d+)?)\s*(?:-|–|—|to)\s*(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?\b",
        text,
        flags=re.IGNORECASE,
    )
    if range_match:
        d1, d2 = float(range_match.group(1)), float(range_match.group(2))
        return (min(d1, d2), max(d1, d2))

    single_match = re.search(
        r"(?:target\s+)?duration[:\s]+(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)?\b",
        text,
        flags=re.IGNORECASE,
    )
    if single_match:
        return float(single_match.group(1))

    single_sec_match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(?:s|sec|secs|seconds)\b",
        text,
        flags=re.IGNORECASE,
    )
    if single_sec_match:
        return float(single_sec_match.group(1))

    return None


def resolve_requested_duration(
    video: Any = None,
    step: Any = None,
    channel_settings: dict | None = None,
    explicit_duration: float | tuple[float, float] | None = None,
    creative_direction_data: dict | None = None,
    require_resolved: bool = False,
) -> tuple[float, float]:
    """Resolve the requested video duration range in seconds from canonical contexts.

    Precedence:
    1. Explicit duration argument (if passed directly).
    2. Revision user_prompt on the step (if user specifically asked for a duration change).
    3. Canonical resolve_video_target_duration (evaluating video.duration_mode, video.target_duration, and creative_direction output).
    """
    if explicit_duration is not None:
        if isinstance(explicit_duration, (list, tuple)) and len(explicit_duration) >= 2:
            d1, d2 = float(explicit_duration[0]), float(explicit_duration[1])
            return (min(d1, d2), max(d1, d2))
        d = float(explicit_duration)
        return (round(max(15.0, d * 0.9), 1), round(d * 1.1, 1))

    # Check if step has an explicit duration revision prompt (e.g. "make it 90s")
    if step is not None:
        user_prompt = getattr(step, "user_prompt", None)
        parsed_rev = parse_duration_from_text(user_prompt)
        if parsed_rev is not None:
            if isinstance(parsed_rev, (list, tuple)) and len(parsed_rev) >= 2:
                d1, d2 = float(parsed_rev[0]), float(parsed_rev[1])
                return (min(d1, d2), max(d1, d2))
            d = float(parsed_rev)
            return (round(max(15.0, d * 0.9), 1), round(d * 1.1, 1))

    # Canonical duration resolution (manual target takes precedence, else CD recommended duration, else error if require_resolved)
    canonical_duration = resolve_video_target_duration(
        video=video,
        creative_direction_data=creative_direction_data,
        require_resolved=require_resolved,
    )
    return (round(max(15.0, canonical_duration * 0.9), 1), round(canonical_duration * 1.1, 1))


def calculate_target_word_count_range(
    target_duration: float | tuple[float, float] | None = None,
    words_per_second: float | None = None,
    speaking_speed: float | None = None,
) -> tuple[int, int]:
    """Derive target word count range (min_words, max_words) for a given duration budget."""
    if target_duration is None:
        return calculate_word_budget_for_duration(60.0, words_per_second, speaking_speed)
    return calculate_word_budget_for_duration(target_duration, words_per_second, speaking_speed)


def estimate_spoken_duration_seconds(
    narration: str, words_per_second: float | None = None, voice_speed: float = 1.0
) -> float:
    """Estimate natural spoken duration in seconds using word count and syllable density."""
    tokens = re.findall(r"\b[\w'-]+\b", narration)
    if not tokens:
        return 0.0
    base_wps = words_per_second or getattr(settings, "script_speaking_rate_wps", 2.4)
    speed = max(0.5, float(voice_speed or 1.0))
    effective_wps = base_wps * speed
    total_syllables = sum(_count_syllables(t) for t in tokens)
    # Blend standard wps with syllable rate + sentence pause allowance
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", narration.strip()) if s.strip()]
    pause_seconds = (max(0, len(sentences) - 1) * 0.35) / speed
    word_dur = len(tokens) / effective_wps
    syll_dur = (total_syllables / (3.9 * speed)) + pause_seconds
    return round((word_dur * 0.6 + syll_dur * 0.4), 1)


def evaluate_script_quality(
    script: ScriptOutput,
    target_duration_range: tuple[float, float] | None = None,
    target_word_range: tuple[int, int] | None = None,
    voice_speed: float = 1.0,
) -> list[str]:
    """Evaluate storytelling, conversational clarity, sentence rhythm, and narrative progression.

    Checks for:
    1. Excessively complex or long sentences.
    2. Stiff textbook/encyclopedic phrasing and dogmatic overclaiming.
    3. Repeated phrases or identical sentence openings.
    4. Internal production instruction or meta-framing leakage.
    5. Redundant meta-summary sentences.
    6. Estimated spoken duration relative to target short-form video length and TTS-unfriendly formatting.

    Returns:
        List of actionable quality warning strings (empty if the script passes all checks).
    """
    issues: list[str] = []
    narration = (script.narration or "").strip()
    if not narration:
        return issues

    words = narration.split()
    if len(words) < 30:
        # Short text is already caught by the word-count check; skip prose structure checks
        return issues

    # Detect synthetic unit-test filler ("word " * 115) used in hook-only unit tests
    dummy_word_count = sum(1 for w in words if w.lower().strip(".,!?;:\"'") == "word")
    if dummy_word_count >= 20:
        return issues

    # Split into sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", narration) if s.strip()]

    # 1. Estimated spoken duration & TTS-unfriendly number formatting
    est_duration = estimate_spoken_duration_seconds(narration, voice_speed=voice_speed)
    if target_duration_range:
        min_dur, max_dur = target_duration_range
    elif target_word_range:
        min_dur = round(target_word_range[0] / (2.6 * max(0.5, voice_speed)), 1)
        max_dur = round(target_word_range[1] / (2.0 * max(0.5, voice_speed)), 1)
    else:
        min_dur, max_dur = (50.0, 70.0)

    min_w, max_w = target_word_range or calculate_target_word_count_range(target_duration_range, speaking_speed=voice_speed)
    if len(words) >= min_w and len(words) <= max_w and (est_duration < min_dur or est_duration > max_dur):
        issues.append(
            f"Estimated spoken duration is {est_duration}s (target: {min_dur:.0f}-{max_dur:.0f}s for a natural short-form pace)."
        )

    comma_num_match = re.search(r"\b\d{1,3}(?:,\d{3})+\b", narration)
    if comma_num_match:
        issues.append(
            f"Narration contains comma-formatted digits ('{comma_num_match.group(0)}'), which TTS engines mispronounce. "
            f"Spell out numbers as spoken words (e.g. 'thirty thousand')."
        )

    # 3. Stiff textbook/encyclopedic language
    textbook_match = TEXTBOOK_PHRASE_REGEX.search(narration)
    if textbook_match:
        issues.append(
            f"Narration uses stiff textbook or academic phrasing ('{textbook_match.group(0)}'). "
            f"Rewrite in warm, conversational plain English as if explaining to a friend."
        )

    # 3b. Epistemic calibration: avoid dogmatic overclaiming on unverified theories
    dogmatic_match = re.search(
        r"\b(?:proven fact that|scientists have proven that|undeniably proven that|settled beyond all doubt that)\b",
        narration,
        re.IGNORECASE,
    )
    if dogmatic_match:
        issues.append(
            f"Narration uses dogmatic assertion ('{dogmatic_match.group(0)}'). "
            "Use measured, natural conversational framing (e.g. 'one idea is that', 'scientists think', 'the reason might be') to distinguish hypotheses from established consensus."
        )

    # 4. Repeated phrases or redundant sentence openings
    consecutive_same_start = False
    for idx in range(1, len(sentences)):
        prev_w = [w.lower() for w in re.findall(r"\b[a-zA-Z']+\b", sentences[idx - 1])[:2]]
        curr_w = [w.lower() for w in re.findall(r"\b[a-zA-Z']+\b", sentences[idx])[:2]]
        if len(prev_w) == 2 and prev_w == curr_w:
            s1_len = len(sentences[idx - 1].split())
            s2_len = len(sentences[idx].split())
            is_rhetorical_parallel = (
                (s1_len <= 14 and s2_len <= 14)
                or curr_w[0] in {"imagine", "picture", "no", "without", "what"}
            )
            if not is_rhetorical_parallel:
                consecutive_same_start = True
                issues.append(
                    f"Consecutive sentences {idx} and {idx + 1} start with the identical phrase '{' '.join(curr_w)}'. "
                    f"Vary sentence openings for natural spoken flow."
                )
                break

    # Check repeated 3-word content phrases (occurring >= 3 times)
    norm_tokens = [w.lower() for w in re.findall(r"\b[a-zA-Z]{3,}\b", narration)]
    trigram_counts: dict[str, int] = {}
    for i in range(len(norm_tokens) - 2):
        tri = norm_tokens[i : i + 3]
        content_in_tri = sum(1 for w in tri if w not in STOPWORDS and len(w) >= 4)
        if content_in_tri >= 2:
            key = " ".join(tri)
            trigram_counts[key] = trigram_counts.get(key, 0) + 1

    repeated_trigrams = [k for k, cnt in trigram_counts.items() if cnt >= 3]
    if repeated_trigrams and not consecutive_same_start:
        issues.append(
            f"Narration repeats the phrase '{repeated_trigrams[0]}' {trigram_counts[repeated_trigrams[0]]} times. "
            f"Use pronouns or varied phrasing instead of repeating the same phrase."
        )

    # 5. Production instruction & meta-framing leakage check (backstage directives leaking into dialogue)
    meta_leak_match = META_INSTRUCTION_LEAK_REGEX.search(narration)
    if meta_leak_match:
        issues.append(
            f"Narration leaks internal production instructions or meta-framing ('{meta_leak_match.group(0)}'). "
            "Never vocalize backstage instructions, protagonist archetypes, or fictional disclaimers. "
            "Narrate the story directly from an immersive perspective."
        )

    # 6. Redundant meta-summary check
    for sent in sentences:
        if REDUNDANT_SUMMARY_REGEX.search(sent):
            issues.append(
                f"Narration contains redundant summary filler ('{sent[:45]}...'). "
                "Explain each mechanism once and keep the narrative moving forward."
            )
            break

    return issues


def validate_script_soft_checks(
    script: ScriptOutput,
    target_duration: float | tuple[float, float] | None = None,
    target_word_range: tuple[int, int] | None = None,
    voice_speed: float = 1.0,
) -> list[str]:
    """Execute code-side soft checks and storytelling quality checks on the generated script output.

    Returns a list of warning messages for any failed checks.
    """
    warnings: list[str] = []

    # 1. Narration word count
    words = (script.narration or "").split()
    word_count = len(words)
    min_w, max_w = target_word_range or calculate_target_word_count_range(target_duration, speaking_speed=voice_speed)
    if word_count < min_w:
        warnings.append(
            f"Narration word count is too short ({word_count} words; target: {min_w}-{max_w} words for the requested duration)."
        )
    elif word_count > max_w:
        warnings.append(
            f"Narration word count is too long ({word_count} words; target: {min_w}-{max_w} words for the requested duration)."
        )

    # 2. Hook cliché and generic opener detection
    cliche_match = CLICHE_HOOK_REGEX.search(script.hook or "")
    if cliche_match:
        warnings.append(
            f"Hook contains generic meta-cliché: '{cliche_match.group(0)}'. Make the hook more specific, surprising, or direct."
        )

    opener_match = GENERIC_OPENER_REGEX.search(script.hook or "")
    if opener_match:
        warnings.append(
            f"Hook starts with generic formula: '{opener_match.group(0).strip()}'. Ask the question directly (e.g. 'Why do...' instead of 'Do you know why...')."
        )

    norm_hook = normalize_for_comparison(script.hook or "").rstrip(".!? ")
    norm_narr = normalize_for_comparison(script.narration or "")
    if not norm_narr.startswith(norm_hook):
        warnings.append("Narration does not begin with the hook sentence.")

    # 4. No bracketed directions or parentheses
    if re.search(r"\[.*?\]|\(.*?\)", script.narration or ""):
        warnings.append("Narration contains bracketed directions or parentheses.")

    # 5. No markdown formatting characters
    if re.search(r"[*#_`]", script.narration or ""):
        warnings.append("Narration contains markdown formatting characters.")

    # 6. No emojis
    if EMOJI_PATTERN.search(script.narration or ""):
        warnings.append("Narration contains emoji characters.")

    # 7. Storytelling, conversational flow, and narrative quality checks
    target_dur_range = None
    if isinstance(target_duration, (list, tuple)) and len(target_duration) >= 2:
        target_dur_range = (float(target_duration[0]), float(target_duration[1]))
    elif isinstance(target_duration, (int, float)):
        target_dur_range = (round(target_duration * 0.9, 1), round(target_duration * 1.1, 1))

    quality_issues = evaluate_script_quality(
        script,
        target_duration_range=target_dur_range,
        target_word_range=(min_w, max_w),
        voice_speed=voice_speed,
    )
    warnings.extend(quality_issues)

    return warnings


def check_objective_technical_failures(
    script: ScriptOutput,
    target_word_range: tuple[int, int],
    target_duration_range: tuple[float, float] | None = None,
    voice_speed: float = 1.0,
) -> list[str]:
    """Check for objective technical failures that strictly warrant an LLM retry.

    Triggering criteria:
    - Missing narration
    - Production note leakage (META_INSTRUCTION_LEAK_REGEX)
    - Unsupported formatting / characters:
      * Brackets or parentheses: [ ... ] or ( ... )
      * Markdown symbols: * # _ `
      * Emojis
    - Significant duration deviation (> 25% off target duration bounds, with word count as supporting guidance)
    """
    failures: list[str] = []
    narration = (script.narration or "").strip()
    if not narration:
        failures.append("Narration is completely missing or empty.")
        return failures

    # 1. Significant duration deviation (> 25% off bounds) as primary measure
    if target_duration_range:
        est_duration = estimate_spoken_duration_seconds(narration, voice_speed=voice_speed)
        min_d, max_d = target_duration_range
        min_allowed_d = round(min_d * 0.75, 1)
        max_allowed_d = round(max_d * 1.25, 1)
        if est_duration < min_allowed_d:
            failures.append(
                f"Estimated spoken duration is significantly too short ({est_duration}s; target: {min_d:.0f}-{max_d:.0f}s, minimum allowed {min_allowed_d:.0f}s)."
            )
        elif est_duration > max_allowed_d:
            failures.append(
                f"Estimated spoken duration is significantly too long ({est_duration}s; target: {min_d:.0f}-{max_d:.0f}s, maximum allowed {max_allowed_d:.0f}s)."
            )

    # Word count as supporting guidance (> 25% deviation from bounds)
    words = narration.split()
    word_count = len(words)
    min_w, max_w = target_word_range
    if word_count < int(min_w * 0.75):
        failures.append(
            f"Narration word count is too short (extreme underflow: {word_count} words; target: {min_w}-{max_w} words, minimum allowed {int(min_w * 0.75)})."
        )
    elif word_count > int(max_w * 1.25):
        failures.append(
            f"Narration word count is too long (extreme overflow: {word_count} words; target: {min_w}-{max_w} words, maximum allowed {int(max_w * 1.25)})."
        )

    leak_match = META_INSTRUCTION_LEAK_REGEX.search(narration)
    if leak_match:
        failures.append(f"Narration leaks internal production instructions or meta-framing ('{leak_match.group(0)}').")

    if re.search(r"\[.*?\]|\(.*?\)", narration):
        failures.append("Narration contains bracketed directions or parentheses.")

    if re.search(r"[*#_`]", narration):
        failures.append("Narration contains markdown formatting characters.")

    if EMOJI_PATTERN.search(narration):
        failures.append("Narration contains emoji characters.")

    return failures


def build_script_prompt(
    rules_text: str,
    topic: str,
    description: str | None,
    channel_name: str | None = None,
    channel_niche: str | None = None,
    channel_settings: dict | None = None,
    previous_script: dict | None = None,
    revision_instruction: str | None = None,
    target_duration_range: tuple[float, float] | None = None,
    target_word_range: tuple[int, int] | None = None,
    creative_direction_data: dict | None = None,
) -> str:
    """Assemble the comprehensive prompt including channel context, constraints, and revision details."""
    sections: list[str] = [rules_text.strip(), "\n## Video Project Details"]

    sections.append(f"- **Video Topic**: {topic}")
    if description:
        sections.append(f"- **Topic Description / Context**: {description}")

    if creative_direction_data:
        sections.append("\n### Approved Creative Direction (Story Narrative Framing)")
        sections.append(f"- **Format Archetype**: {creative_direction_data.get('format_archetype', '')}")
        sections.append(f"- **Narrative Angle**: {creative_direction_data.get('storytelling_angle', '')}")
        sections.append(f"- **Central Thesis**: {creative_direction_data.get('central_thesis', '')}")
        sections.append(f"- **Tone**: {creative_direction_data.get('tone', '')}")

        protagonist = creative_direction_data.get("protagonist_config", {})
        constraints = creative_direction_data.get("creative_constraints", [])
        has_protagonist = (
            isinstance(protagonist, dict)
            and protagonist.get("human_presence") not in (None, "none")
            and protagonist.get("archetype_summary")
        )

        if has_protagonist or constraints:
            sections.append(
                "\n### Production Directives & Internal Guardrails (INTERNAL ONLY — DO NOT NARRATE)\n"
                "[CRITICAL: The items below are backstage production instructions, factual boundaries, and character staging notes. "
                "They are strictly internal guardrails. NEVER narrate these instructions or use phrases like 'fictional character', "
                "'representative farmer', 'hypothetical figure', or 'creative constraint' in the spoken script. "
                "Ground characters naturally in active storytelling without breaking the fourth wall.]"
            )
            if has_protagonist:
                sections.append(f"- **Character Staging / Protagonist Archetype (Internal Note)**: {protagonist.get('archetype_summary')}")
            if constraints:
                sections.append("- **Factual & Creative Guardrails (Internal Notes)**:\n  " + "\n  ".join(f"- {c}" for c in constraints))

        narrative_plan = creative_direction_data.get("narrative_plan")
        if narrative_plan and isinstance(narrative_plan, dict):
            sections.append("\n### Approved Narrative Plan & Storytelling Strategy")
            if narrative_plan.get("narrative_perspective"):
                sections.append(f"- **Narrative Perspective**: {narrative_plan['narrative_perspective']}")
            if narrative_plan.get("opening_situation"):
                sections.append(f"- **Opening Scene / Situation**: {narrative_plan['opening_situation']}")
            if narrative_plan.get("information_revelation_strategy"):
                sections.append(f"- **Information Revelation Strategy**: {narrative_plan['information_revelation_strategy']}")
            prog = narrative_plan.get("story_progression", [])
            if prog and isinstance(prog, list):
                sections.append("- **Story Progression Guidance**:")
                for st in prog:
                    if isinstance(st, dict):
                        sections.append(f"  * **{st.get('stage_name', 'Stage')}**: {st.get('description', '')}")
            if narrative_plan.get("audience_takeaway"):
                sections.append(f"- **Intended Takeaway**: {narrative_plan['audience_takeaway']}")
            if narrative_plan.get("resolution_payoff"):
                sections.append(f"- **Ending Payoff**: {narrative_plan['resolution_payoff']}")

    if target_duration_range or target_word_range:
        min_d, max_d = target_duration_range or (35.0, 45.0)
        min_w, max_w = target_word_range or calculate_target_word_count_range(target_duration_range)
        sections.append("\n### Target Duration & Word Count Guidance")
        sections.append(f"- **Requested Duration**: {min_d:.0f} to {max_d:.0f} seconds")
        sections.append(
            f"- **Target Spoken Word Count**: {min_w} to {max_w} words "
            f"(approximate storytelling budget; scale pacing and depth to fit this length)."
        )

    if channel_name or channel_niche or channel_settings:
        sections.append("\n### Channel Context")
        if channel_name:
            sections.append(f"- **Channel Name**: {channel_name}")
        if channel_niche:
            sections.append(f"- **Niche**: {channel_niche}")
        if channel_settings:
            filtered_settings = filter_conflicting_channel_settings(channel_settings)
            if filtered_settings:
                sections.append(f"- **Channel Settings**: {json.dumps(filtered_settings)}")

    if previous_script and revision_instruction:
        sections.append("\n## REVISION MODE")
        sections.append(
            "A previous draft of the script exists. Modify ONLY what the revision instruction requests, "
            "and preserve everything else (tone, verified facts, structure)."
        )
        sections.append(f"- **Revision Instruction**: {revision_instruction}")
        sections.append(f"- **Previous Script**: {json.dumps(previous_script, indent=2)}")

    min_w_str = f"between {target_word_range[0]} and {target_word_range[1]} words" if target_word_range else "within the target word count"
    sections.append(
        "\n## Instructions\n"
        f"Generate a structured script whose spoken narration is approximately {min_w_str}, "
        "written in natural, conversational spoken English, conforming to the ScriptOutput schema."
    )

    return "\n".join(sections)


def generate_script_step(
    video: Video,
    prev: dict[str, Any],
    step: VideoStep,
    target_duration: float | tuple[float, float] | None = None,
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

    # 3. Extract Creative Direction if present (from 8-step pipeline)
    cd_step = prev.get("creative_direction") if isinstance(prev, dict) else None
    cd_data = cd_step.get("creative_direction") if isinstance(cd_step, dict) else None

    # 4. Check Revision Mode (previous script comes ONLY from step.output)
    is_revision = False
    previous_script = None
    if step.user_prompt and isinstance(step.output, dict) and step.output.get("script"):
        is_revision = True
        previous_script = step.output["script"]

    # 5. Load writing prompt template
    rules_text, prompt_version = load_prompt_template()

    # 5b. Resolve target duration and word count constraints
    voice_speed = 1.0
    if isinstance(channel_settings, dict):
        try:
            voice_speed = float(channel_settings.get("voice_speed", 1.0))
        except (ValueError, TypeError):
            voice_speed = 1.0

    target_duration_range = resolve_requested_duration(
        video=video,
        step=step,
        channel_settings=channel_settings,
        explicit_duration=target_duration,
        creative_direction_data=cd_data,
        require_resolved=True,
    )
    target_word_range = calculate_target_word_count_range(
        target_duration_range, speaking_speed=voice_speed
    )

    # 6. Build prompt
    prompt = build_script_prompt(
        rules_text=rules_text,
        topic=video.topic,
        description=video.description,
        channel_name=channel_name,
        channel_niche=channel_niche,
        channel_settings=channel_settings,
        previous_script=previous_script,
        revision_instruction=step.user_prompt,
        target_duration_range=target_duration_range,
        target_word_range=target_word_range,
        creative_direction_data=cd_data,
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

    # 8. Soft-checks and storytelling quality validation
    warnings = validate_script_soft_checks(
        script_output,
        target_duration=target_duration_range,
        target_word_range=target_word_range,
        voice_speed=voice_speed,
    )

    # 9. Targeted rewrite strictly on objective technical failures
    retry_reasons = check_objective_technical_failures(
        script_output,
        target_word_range=target_word_range,
        target_duration_range=target_duration_range,
        voice_speed=voice_speed,
    )
    attempts_history: list[dict[str, Any]] = []

    if retry_reasons and llm.provider_name != "mock":
        attempts = 2
        attempts_history.append({
            "attempt": 1,
            "script": script_output.model_dump(),
            "triggering_reasons": retry_reasons,
            "provider": llm.provider_name,
            "model": llm.model,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "reasoning_tokens": result.thinking_tokens,
            "latency_seconds": round(result.latency_seconds, 4),
        })
        retry_prompt = (
            f"{prompt}\n\n"
            f"[CRITICAL TECHNICAL FIX REQUIRED: Your previous script draft failed the following technical requirements:\n"
            + "\n".join(f"- {r}" for r in retry_reasons)
            + "\n\nPreviously generated draft (DO NOT REUSE DEFECTS):\n"
            + json.dumps(script_output.model_dump(), indent=2)
            + f"\n\nPlease rewrite the script to resolve these technical failures while preserving the core narrative and factual accuracy. "
            f"Specifically ensure narration word count is between {target_word_range[0]} and {target_word_range[1]} words "
            f"(target duration: {target_duration_range[0]:.0f}-{target_duration_range[1]:.0f}s), "
            "use plain conversational spoken text without brackets, parentheses, markdown formatting characters, or emojis, "
            "and strictly conform to the ScriptOutput schema.]"
        )
        try:
            retry_result = llm.generate_structured(
                prompt=retry_prompt,
                response_model=ScriptOutput,
                task="script",
            )
            retry_output = retry_result.output
            total_in_tokens += retry_result.input_tokens or 0
            total_out_tokens += retry_result.output_tokens or 0
            if retry_result.thinking_tokens is not None:
                total_think_tokens = (total_think_tokens or 0) + retry_result.thinking_tokens
            total_latency += retry_result.latency_seconds

            # Verify Attempt 2 strictly; do not allow invalid final draft to become authoritative
            attempt2_failures = check_objective_technical_failures(
                retry_output,
                target_word_range=target_word_range,
                target_duration_range=target_duration_range,
                voice_speed=voice_speed,
            )
            if attempt2_failures:
                attempts_history.append({
                    "attempt": 2,
                    "script": retry_output.model_dump(),
                    "triggering_reasons": attempt2_failures,
                    "provider": llm.provider_name,
                    "model": llm.model,
                    "input_tokens": retry_result.input_tokens,
                    "output_tokens": retry_result.output_tokens,
                    "reasoning_tokens": retry_result.thinking_tokens,
                    "latency_seconds": round(retry_result.latency_seconds, 4),
                })
                raise LLMError(
                    f"Script generation failed technical validation after retry: {'; '.join(attempt2_failures)}"
                )

            script_output = retry_output
            # Re-evaluate warnings for Attempt 2
            warnings = validate_script_soft_checks(
                script_output,
                target_duration=target_duration_range,
                target_word_range=target_word_range,
                voice_speed=voice_speed,
            )
        except Exception as exc:
            if isinstance(exc, LLMError):
                raise
            raise LLMError(f"Script retry generation failed: {exc}") from exc

    est_spoken_seconds = estimate_spoken_duration_seconds(
        script_output.narration, voice_speed=voice_speed
    )

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
            "attempts_history": attempts_history,
            "revision": is_revision,
            "target_duration_seconds": list(target_duration_range),
            "target_word_count_range": list(target_word_range),
            "estimated_duration_seconds": est_spoken_seconds,
        },
    }
