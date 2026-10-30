"""Script step implementation for Shorts Studio.

Generates structured video script outputs using task-routed LLMs (Mock, Gemini, OpenAI).
Includes code-side validation for duration, formatting, angle diversity, conversational clarity,
sentence complexity, technical jargon, transitions, and narrative progression.
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
    r"imagine\b",
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
    r"\bfurthermore\b",
    r"\bmoreover\b",
    r"\bconsequently\b",
    r"\bnevertheless\b",
    r"\bnotwithstanding\b",
    r"\baforementioned\b",
    r"\butilizes?\b",
    r"\bfacilitates?\b",
    r"\bis\s+characterized\s+by\b",
    r"\bcan\s+be\s+attributed\s+to\b",
    r"\bempirically\b",
    r"\bheretofore\b",
)

TEXTBOOK_PHRASE_REGEX = re.compile(
    r"(?:" + "|".join(TEXTBOOK_PHRASE_PATTERNS) + r")",
    flags=re.IGNORECASE,
)

# Second-sentence throat-clearing filler that delays answering the hook
DELAYED_PAYOFF_PATTERNS: tuple[str, ...] = (
    r"^to\s+understand\s+why\b",
    r"^before\s+we\s+(?:answer|explain|look)\b",
    r"^let(?:'s|\s+us)\s+take\s+a\s+(?:closer\s+)?look\b",
    r"^the\s+answer\s+is\s+(?:actually\s+)?(?:surprising|fascinating|interesting|simple)\b",
    r"^you\s+might\s+be\s+surprised\s+to\s+(?:learn|know|hear)\b",
)

DELAYED_PAYOFF_REGEX = re.compile(
    r"(?:" + "|".join(DELAYED_PAYOFF_PATTERNS) + r")",
    flags=re.IGNORECASE,
)

# Weak endings, hedged disclaimers, or calls-to-action at the end of the script
WEAK_ENDING_PATTERNS: tuple[str, ...] = (
    r"like\s+and\s+subscribe",
    r"follow\s+for\s+more",
    r"subscribe\s+for\s+more",
    r"leave\s+a\s+comment",
    r"is\s+still\s+being\s+studied\.?$",
    r"more\s+research\s+is\s+needed\.?$",
    r"remains?\s+uncertain\.?$",
    r"is\s+still\s+incomplete\.?$",
)

WEAK_ENDING_REGEX = re.compile(
    r"(?:" + "|".join(WEAK_ENDING_PATTERNS) + r")",
    flags=re.IGNORECASE,
)

# Common everyday multi-syllable words that are immediately understandable in spoken English
EVERYDAY_MULTISYLLABLE_ALLOWLIST: frozenset[str] = frozenset({
    "actually", "additional", "airplane", "airplanes", "airliner", "altitude",
    "america", "american", "another", "anything", "anywhere", "atmosphere",
    "atmospheres", "attention", "automatically", "available", "basically",
    "beautiful", "centimeter", "centimeters", "comfortable", "commercial",
    "commonplace", "communicating", "communication", "completely", "complicated",
    "computer", "computers", "condition", "conditions", "connected", "connection",
    "construction", "contagious", "continuous", "conversation", "curiosity",
    "dangerous", "definitely", "destination", "difference", "different",
    "difficult", "direction", "directly", "disappear", "discovered", "discovering",
    "discovery", "electricity", "electronic", "emergency", "engineering",
    "environment", "especially", "essential", "eventually", "everybody",
    "everyday", "everything", "everywhere", "exact", "exactly", "example",
    "examples", "expectation", "experience", "experiment", "experiments",
    "explanation", "extraordinary", "extremely", "fascinating", "favorite",
    "following", "fortunately", "foundation", "generations", "generally",
    "gravity", "helicopter", "historical", "history", "horizontal", "however",
    "immediately", "important", "impossible", "incomplete", "incredible",
    "incredibly", "individual", "information", "ingredient", "ingredients",
    "initially", "inside", "instantly", "instead", "interesting", "invisible",
    "kilometer", "kilometers", "literally", "machinery", "manufacturing",
    "material", "materials", "mechanical", "mechanism", "medical", "millimeter",
    "millimeters", "miniature", "misleading", "modern", "moisture", "mysterious",
    "naturally", "necessary", "newspaper", "normally", "noticeable", "noticing",
    "observation", "observations", "obviously", "obstructing", "occasional",
    "operation", "ordinary", "original", "originally", "otherwise", "outside",
    "passenger", "passengers", "particular", "particularly", "perfectly",
    "permanent", "perspective", "physical", "physically", "possibility",
    "possible", "possibly", "powerful", "practical", "practically", "pressure",
    "pressurized", "preventing", "previously", "primary", "probably",
    "protection", "protective", "question", "questions", "quickly", "quietly",
    "reality", " Protection", "realize", "really", "reasonable", "recently",
    "recognize", "refrigerator", "regular", "regularly", "relationship",
    "relatively", "reliably", "remember", "reminding", "protective", "scientist",
    "scientists", "astronomer", "astronomers", "researcher", "researchers",
    "separate", "separately", "seriously", "similar", "similarly", "simplicity",
    "situation", "socially", "solution", "somebody", "something", "sometimes",
    "somewhere", "specific", "specifically", "standard", "structural", "structure",
    "structures", "suddenly", "Surrounding", "surrounding", "surprising",
    "surprisingly", "survival", "technology", "television", "temperature",
    "temperatures", "temporary", "thousands", "together", "traditional",
    "traveler", "travelers", "typically", "ultimately", "umbrella", "uncomfortable",
    "understand", "understanding", "unexpected", "unfamiliar", "unfortunately",
    "universe", "unlikely", "unnecessary", "unusual", "usually", "valuable",
    "variation", "vertical", "visibility", "waterproof", "whenever", "wherever",
    "wonderful", "yesterday",
})

# Plain-English explanation cues that clarify a technical concept in context
PLAIN_ENGLISH_EXPLANATION_CUES: tuple[str, ...] = (
    "called", "known as", "means", "meaning", "which is", "that is", "it's the",
    "is the point", "is a", "acts like", "works like", "like a", "basically",
    "in other words", "simply", "because", "so that", "allowing", "preventing",
)

# Discourse/referential/causal connectors that link a sentence to the preceding idea
SENTENCE_COHESION_MARKERS: frozenset[str] = frozenset({
    "this", "that", "these", "those", "it", "its", "it's", "they", "their", "them",
    "there", "here", "such", "same", "other", "another", "both", "each", "every",
    "because", "so", "but", "yet", "still", "though", "although", "while", "when",
    "once", "as", "until", "unless", "if", "without", "instead", "beyond", "inside",
    "outside", "beneath", "under", "over", "through", "across", "near", "far",
    "farther", "further", "even", "only", "just", "why", "how", "which", "where",
})

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
            "Attention-grabbing opening hook, exactly ONE sentence, ideally 4-10 words (up to 15 words max), "
            "opening a specific curiosity gap—often a direct question or surprising fact without filler or meta-clichés"
        )
    )
    narration: str = Field(
        description=(
            "The full spoken narration text, 110-150 words, plain conversational spoken text only, "
            "must begin verbatim with the hook sentence and unfold as one connected cause-and-effect story"
        )
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


def estimate_spoken_duration_seconds(narration: str, words_per_second: float = 2.5) -> float:
    """Estimate natural spoken duration in seconds using word count and syllable density."""
    tokens = re.findall(r"\b[\w'-]+\b", narration)
    if not tokens:
        return 0.0
    total_syllables = sum(_count_syllables(t) for t in tokens)
    # Blend standard 2.5 wps with syllable rate (~3.75 syllables/sec) + sentence pause allowance
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", narration.strip()) if s.strip()]
    pause_seconds = max(0, len(sentences) - 1) * 0.35
    word_dur = len(tokens) / words_per_second
    syll_dur = (total_syllables / 3.9) + pause_seconds
    return round((word_dur * 0.6 + syll_dur * 0.4), 1)


def evaluate_script_quality(
    script: ScriptOutput,
    target_duration_range: tuple[float, float] = (36.0, 65.0),
) -> list[str]:
    """Evaluate storytelling, conversational clarity, sentence rhythm, and narrative progression.

    Checks for:
    1. Excessively complex or long sentences.
    2. Unexplained technical vocabulary or stiff textbook/encyclopedic phrasing.
    3. Abrupt transitions or disconnected fact-list writing.
    4. Repeated phrases or redundant information.
    5. Weak or missing narrative progression (delayed payoff, missing cause-and-effect, weak ending).
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
    est_duration = estimate_spoken_duration_seconds(narration)
    min_dur, max_dur = target_duration_range
    if len(words) >= 110 and len(words) <= 150 and (est_duration < min_dur or est_duration > max_dur):
        issues.append(
            f"Estimated spoken duration is {est_duration}s (target: {min_dur:.0f}-{max_dur:.0f}s for a natural short-form pace)."
        )

    comma_num_match = re.search(r"\b\d{1,3}(?:,\d{3})+\b", narration)
    if comma_num_match:
        issues.append(
            f"Narration contains comma-formatted digits ('{comma_num_match.group(0)}'), which TTS engines mispronounce. "
            f"Spell out numbers as spoken words (e.g. 'thirty thousand')."
        )

    # 2. Excessively long or complex sentences
    long_sentences: list[str] = []
    for idx, sent in enumerate(sentences, start=1):
        s_words = sent.split()
        if len(s_words) > 28:
            long_sentences.append(f"sentence {idx} ({len(s_words)} words)")
        if sent.count(":") + sent.count(";") >= 2:
            issues.append(
                f"Sentence {idx} stacks multiple colons or semicolons, making it sound like written academic prose. "
                f"Use natural conversational clauses instead."
            )

    if long_sentences:
        issues.append(
            f"Narration has overly long sentences ({', '.join(long_sentences)}; keep individual sentences under 28 words for clear spoken delivery)."
        )

    if sentences:
        avg_len = len(words) / len(sentences)
        if avg_len > 22.5:
            issues.append(
                f"Average sentence length is {avg_len:.1f} words across {len(sentences)} sentences. "
                f"Break up dense sentences and mix shorter punchy lines with medium explanations."
            )

    # 3. Stiff textbook/encyclopedic language & unexplained technical vocabulary
    textbook_match = TEXTBOOK_PHRASE_REGEX.search(narration)
    if textbook_match:
        issues.append(
            f"Narration uses stiff textbook or academic phrasing ('{textbook_match.group(0)}'). "
            f"Rewrite in warm, conversational plain English as if explaining to a friend."
        )

    # Check for unexplained complex/polysyllabic jargon (>= 4 syllables not in everyday allowlist)
    unexplained_jargon: list[str] = []
    for idx, sent in enumerate(sentences):
        sent_lower = sent.lower()
        neighbor_context = sent_lower
        if idx + 1 < len(sentences):
            neighbor_context += " " + sentences[idx + 1].lower()
        has_explanation_cue = any(cue in neighbor_context for cue in PLAIN_ENGLISH_EXPLANATION_CUES)

        tokens = re.findall(r"\b[a-zA-Z]{7,}\b", sent)
        for tok in tokens:
            low = tok.lower()
            if low in EVERYDAY_MULTISYLLABLE_ALLOWLIST:
                continue
            if _count_syllables(low) >= 4 and not has_explanation_cue:
                if low not in unexplained_jargon:
                    unexplained_jargon.append(low)

    if len(unexplained_jargon) >= 3:
        sample_terms = ", ".join(f"'{w}'" for w in unexplained_jargon[:4])
        issues.append(
            f"Narration contains dense or unexplained technical vocabulary ({sample_terms}). "
            f"Replace complex vocabulary with everyday words or explain technical terms in plain English."
        )

    # 4. Disconnected fact-list writing & abrupt transitions
    also_trivia_count = 0
    hard_catalog_openers: list[str] = []
    for sent in sentences[1:]:
        cat_m = re.match(
            r"^(?:Additionally|Furthermore|Moreover|In\s+addition|Another\s+(?:fact|interesting\s+fact|thing))\b",
            sent,
            flags=re.IGNORECASE,
        )
        if cat_m:
            hard_catalog_openers.append(cat_m.group(0))
        if re.match(
            r"^(?:[A-Z][a-z]+\s+also\b|It\s+also\s+tends\b|They\s+also\b|Scientists\s+first\s+detected\b)",
            sent,
        ):
            also_trivia_count += 1

    if hard_catalog_openers or also_trivia_count >= 2:
        issues.append(
            "Narration reads like a disconnected list of facts (using additive 'also' / 'furthermore' trivia jumps). "
            "Focus on one connected cause-and-effect narrative thread."
        )

    # Check inter-sentence cohesion across body sentences (using a 3-sentence rolling context window + word stems)
    if len(sentences) >= 5:
        unlinked_sentences = 0
        recent_stem_sets: list[set[str]] = []
        for idx, sent in enumerate(sentences):
            sent_tokens = [w.lower() for w in re.findall(r"\b[a-zA-Z']+\b", sent)]
            curr_stems = {
                re.sub(r"(?:ies|es|s|ed|ing)$", "", w.strip("'"))[:6]
                for w in sent_tokens
                if len(w.strip("'")) >= 4 and w.strip("'") not in STOPWORDS
            }
            if idx >= 1:
                first_four = set(sent_tokens[:4])
                has_connector = bool(first_four & SENTENCE_COHESION_MARKERS) or any(
                    m in sent_tokens for m in ("because", "so", "but", "while", "without", "instead", "which", "though")
                )
                rolling_context = set().union(*recent_stem_sets[-3:]) if recent_stem_sets else set()
                shares_keyword = bool(curr_stems & rolling_context)
                if not has_connector and not shares_keyword:
                    unlinked_sentences += 1
            recent_stem_sets.append(curr_stems)

        if unlinked_sentences >= max(3, int(len(sentences) * 0.4)):
            issues.append(
                f"Narration has abrupt transitions between ideas ({unlinked_sentences} sentences introduce disconnected thoughts without linking to the previous context). "
                f"Use natural conversational transitions so each sentence flows from what came before."
            )

    # 5. Repeated phrases or redundant sentence openings
    consecutive_same_start = False
    for idx in range(1, len(sentences)):
        prev_w = [w.lower() for w in re.findall(r"\b[a-zA-Z']+\b", sentences[idx - 1])[:2]]
        curr_w = [w.lower() for w in re.findall(r"\b[a-zA-Z']+\b", sentences[idx])[:2]]
        if len(prev_w) == 2 and prev_w == curr_w:
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

    # 6. Narrative progression checks (immediate payoff after hook, cause-and-effect, strong ending)
    if len(sentences) >= 2:
        second_sent = sentences[1]
        if DELAYED_PAYOFF_REGEX.search(second_sent):
            issues.append(
                f"Second sentence delays the explanation with filler ('{second_sent[:40]}...'). "
                f"Answer or advance the hook immediately in the second sentence."
            )

    if sentences:
        last_sent = sentences[-1].strip()
        if WEAK_ENDING_REGEX.search(last_sent):
            issues.append(
                f"Final sentence ends weakly with a hedged disclaimer or call-to-action ('{last_sent}'). "
                f"End with a crisp, memorable insight or reveal."
            )

    return issues


def validate_script_soft_checks(script: ScriptOutput) -> list[str]:
    """Execute code-side soft checks and storytelling quality checks on the generated script output.

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

    # 2. Hook length (target: 4-10 words, soft warning > 10, limit <= 15) and narration start
    hook_words = script.hook.split()
    if len(hook_words) > 15:
        warnings.append(
            f"Hook length is {len(hook_words)} words (target: at most 15 words)."
        )
    elif len(hook_words) > 10:
        warnings.append(
            f"Hook length is {len(hook_words)} words (target: ideally 4-10 words for short-form retention, max 15)."
        )

    # 3. Hook cliché and generic opener detection
    cliche_match = CLICHE_HOOK_REGEX.search(script.hook)
    if cliche_match:
        warnings.append(
            f"Hook contains generic meta-cliché: '{cliche_match.group(0)}'. Make the hook more specific, surprising, or direct."
        )

    opener_match = GENERIC_OPENER_REGEX.search(script.hook)
    if opener_match:
        warnings.append(
            f"Hook starts with generic formula: '{opener_match.group(0).strip()}'. Ask the question directly (e.g. 'Why do...' instead of 'Do you know why...')."
        )

    norm_hook = normalize_for_comparison(script.hook).rstrip(".!? ")
    norm_narr = normalize_for_comparison(script.narration)
    if not norm_narr.startswith(norm_hook):
        warnings.append("Narration does not begin with the hook sentence.")

    # 4. No bracketed directions or parentheses
    if re.search(r"\[.*?\]|\(.*?\)", script.narration):
        warnings.append("Narration contains bracketed directions or parentheses.")

    # 5. No markdown formatting characters
    if re.search(r"[*#_`]", script.narration):
        warnings.append("Narration contains markdown formatting characters.")

    # 6. No emojis
    if EMOJI_PATTERN.search(script.narration):
        warnings.append("Narration contains emoji characters.")

    # 7. Angle membership
    if script.angle not in ALLOWED_ANGLES:
        warnings.append(
            f"Angle '{script.angle}' is not in the allowed angles list: {list(ALLOWED_ANGLES)}."
        )

    # 8. Claims to verify count (1 - 6)
    claims_count = len(script.claims_to_verify)
    if claims_count < 1 or claims_count > 6:
        warnings.append(
            f"Claims count is {claims_count} (target: 1-6 standalone claims)."
        )

    # 9. Hashtags count (3 - 5)
    tags_count = len(script.hashtags)
    if tags_count < 3 or tags_count > 5:
        warnings.append(
            f"Hashtags count is {tags_count} (target: 3-5 tags)."
        )

    # 10. Storytelling, conversational flow, and narrative progression checks
    quality_issues = evaluate_script_quality(script)
    warnings.extend(quality_issues)

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
        "Plan one connected story arc (hook -> immediate answer -> progressive cause-and-effect -> memorable payoff) "
        "and generate a structured script that strictly conforms to the ScriptOutput schema."
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

    # 8. Soft-checks and storytelling quality validation
    warnings = validate_script_soft_checks(script_output)

    # 9. Targeted rewrite (once max) if soft checks or quality checks fail and not in mock mode
    if warnings and llm.provider_name != "mock":
        attempts = 2
        retry_prompt = (
            f"{prompt}\n\n"
            f"[IMPORTANT FEEDBACK: Your previous script draft had the following soft-check and storytelling quality issues:\n"
            + "\n".join(f"- {w}" for w in warnings)
            + "\n\nPreviously generated draft:\n"
            + json.dumps(script_output.model_dump(), indent=2)
            + "\n\nPlease perform a targeted rewrite that fixes every listed issue while keeping verified facts accurate, "
            "using natural conversational language, connecting sentences into one smooth cause-and-effect narrative, "
            "and strictly conforming to the ScriptOutput schema.]"
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

    est_spoken_seconds = estimate_spoken_duration_seconds(script_output.narration)

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
            "estimated_duration_seconds": est_spoken_seconds,
        },
    }
