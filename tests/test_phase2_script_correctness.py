"""Regression tests for Phase 2: Script & Narration Correctness.

Verifies:
1. Production instruction / meta-framing leakage detection (e.g. 'picture a fictional farmer')
2. Legitimate usage of words like 'fictional' or 'imagine' passes without false flags
3. Deflated hook-payoff dynamics (e.g. 'How could an ocean vanish? It didn't') is caught
4. Diverse genre hooks (history, science, nature, product) pass validation cleanly
5. Backstage production directives and protagonist archetypes are isolated in prompt building
6. Product explainer topics continue to be supported without narrow channel locking
7. Controlled retry behavior when storytelling quality issues occur
"""

import json
import uuid
from unittest.mock import MagicMock
import pytest

from app.llm.types import LLMResult
from app.steps.script import (
    ScriptOutput,
    build_script_prompt,
    evaluate_script_quality,
    generate_script_step,
    load_prompt_template,
    validate_script_soft_checks,
)


def _make_script(
    hook: str,
    body: str,
    angle: str = "myth vs reality",
    title: str = "Test Script",
) -> ScriptOutput:
    """Helper to assemble a ScriptOutput for validation tests."""
    if body.strip().startswith(hook.strip()):
        narration = body.strip()
    else:
        narration = f"{hook} {body}".strip()
    return ScriptOutput(
        title=title,
        hook=hook,
        narration=narration,
        claims_to_verify=[
            "Test verification claim one.",
            "Test verification claim two.",
        ],
    )


# ==============================================================================
# Task 1: Separate Production Instructions from Spoken Narration
# ==============================================================================

def test_validator_detects_production_instruction_leakage():
    """Verify soft-checks detect backstage instructions and meta-framing leaking into dialogue."""
    hook = "Why did temperatures drop across the globe in eighteen sixteen?"
    leaked_body = (
        "Picture a fictional farmer in New England waking up to frozen corn in July. "
        "Keep the farmer fictional and representative of ordinary people facing the disaster. "
        "The real culprit was thousands of miles away on an island in Indonesia. "
        "Mount Tambora had blown one hundred cubic kilometers of ash into the stratosphere. "
        "This dense volcanic shroud blocked solar radiation and chilled the northern hemisphere. "
        "Because of that global veil, eighteen sixteen became known as the year without a summer."
    )
    script = _make_script(hook=hook, body=leaked_body, angle="origin story")
    issues = evaluate_script_quality(script, target_word_range=(60, 110))

    assert any("leaks internal production instructions" in i.lower() for i in issues)
    assert any("fictional farmer" in i.lower() or "representative" in i.lower() for i in issues)


def test_validator_allows_legitimate_contextual_words():
    """Verify legitimate uses of words like 'fictional' or 'imagine' in valid narrative context pass."""
    hook = "Why does Sherlock Holmes still feel like a real person?"
    legitimate_body = (
        "Sherlock Holmes is a fictional detective created in the nineteenth century, but thousands "
        "of readers still write letters to his Baker Street address. Arthur Conan Doyle based Holmes "
        "on his medical professor who could diagnose strangers simply by observing their boots. "
        "By grounding keen forensic observation in real Victorian London, the stories blurred the line "
        "between imagination and historical fact. That realistic method made an invented character feel alive."
    )
    script = _make_script(hook=hook, body=legitimate_body, angle="origin story")
    issues = evaluate_script_quality(script, target_word_range=(60, 110))

    assert not any("leaks internal production instructions" in i.lower() for i in issues)


def test_build_script_prompt_isolates_backstage_directives():
    """Verify build_script_prompt isolates protagonist archetypes and creative constraints as backstage notes."""
    cd_data = {
        "format_archetype": "historical_chronicle",
        "storytelling_angle": "The Year Without a Summer",
        "central_thesis": "Mount Tambora's eruption caused global climate disruption.",
        "tone": "gripping and immersive",
        "protagonist_config": {
            "human_presence": "single",
            "archetype_summary": "An 1816 New England farmer witnessing crop frost in July (keep fictional and representative)",
        },
        "creative_constraints": [
            "Clarify that the ocean did not literally vanish",
            "Maintain strict 1816 historical authenticity",
        ],
    }

    prompt = build_script_prompt(
        rules_text="Base Rules",
        topic="The Ocean That Disappeared Overnight",
        description="Historical climate anomaly",
        creative_direction_data=cd_data,
    )

    # Narrative framing should be labeled
    assert "Approved Creative Direction (Story Narrative Framing)" in prompt
    # Backstage directives must be explicitly marked as internal and forbidden to narrate
    assert "Production Directives & Internal Guardrails (INTERNAL ONLY — DO NOT NARRATE)" in prompt
    assert "Character Staging / Protagonist Archetype (Internal Note)" in prompt
    assert "Factual & Creative Guardrails (Internal Notes)" in prompt
    assert "NEVER narrate these instructions or use phrases like 'fictional character'" in prompt


# ==============================================================================
# Task 2: Improve Hook and Payoff Quality
# ==============================================================================

def test_validator_detects_deflated_hook_negation():
    """Verify that deflated question negation regex has been removed under Phase 1D.7."""
    hook = "How could an entire ocean vanish without moving?"
    deflated_body = (
        "It didn't; Mount Tambora erupted thousands of miles away and darkened the sky. "
        "In the summer of eighteen sixteen, cold ash clouds blocked out global sunlight. "
        "Crops froze across North America and Europe, causing widespread food shortages. "
        "People thought the seas had dried up because rainfall dropped so drastically. "
        "The water never left, but an invisible volcanic winter changed the planet forever."
    )
    script = _make_script(hook=hook, body=deflated_body, angle="myth vs reality")
    issues = evaluate_script_quality(script, target_word_range=(55, 100))

    assert not any("immediately deflates the hook question" in i.lower() for i in issues)


def test_validator_passes_suspenseful_hook_and_narrative_reveal():
    """Verify a strong hook followed by an immediate tension-building reveal passes cleanly."""
    hook = "How could an entire ocean seem to vanish overnight?"
    engaging_body = (
        "In the summer of eighteen sixteen, harbor masters woke up to find docks marooned in mud. "
        "A colossal volcanic eruption on the other side of the world had blasted millions of tons of ash "
        "into the atmosphere. That dark blanket blocked solar heat, shutting down global weather systems and "
        "drying up coastal tributaries. While terrified locals feared the sea had drained away, the water "
        "was simply trapped by disrupted rains. One violent eruption temporarily rewrote the climate of the entire planet."
    )
    script = _make_script(hook=hook, body=engaging_body, angle="myth vs reality")
    issues = evaluate_script_quality(script, target_word_range=(65, 120))

    assert not any("deflates the hook question" in i.lower() for i in issues)
    assert not any("delays the explanation" in i.lower() for i in issues)


# ==============================================================================
# Task 3: Remove Overly Narrow Genre Assumptions
# ==============================================================================

def test_prompt_template_supports_diverse_genres():
    """Verify script.md prompt template supports history, science, nature, and engineering genres."""
    rules_text, prompt_version = load_prompt_template()

    # Channel-specific hardcoding must be removed from prompt template
    assert '("Hidden Facts About Everyday Things")' not in rules_text
    assert "High-Retention Short-Form Storytelling" in rules_text
    assert "history, science, mysteries, nature, engineering" in rules_text

    # Cross-genre angle guidance must be present
    assert "Engineering / Objects" in rules_text or "Engineering" in rules_text
    assert "Nature / Science" in rules_text or "Science" in rules_text
    assert "History / Science" in rules_text or "History" in rules_text

    # Backstage isolation rule must be present
    assert "Production & Directorial Isolation Rule" in rules_text


def test_product_explainer_genre_remains_fully_supported():
    """Verify everyday product explainers continue to pass validation seamlessly."""
    hook = "Why do airplane windows need a tiny hole?"
    product_body = (
        "That tiny bleed hole balances the immense pressure difference between the cabin and thin outside air. "
        "Airplane windows have three acrylic panes, and the middle pane holds this tiny opening. "
        "It lets pressure equalize against the heavy outer pane while venting moisture so the glass never fogs over. "
        "Without that simple opening, high altitude frost would completely blind the pilots. "
        "A two millimeter hole quietly keeps millions of high altitude travelers safe every day."
    )
    script = _make_script(hook=hook, body=product_body, angle="why it's designed this way")
    issues = evaluate_script_quality(script, target_word_range=(60, 105))

    assert issues == []


# ==============================================================================
# Task 4: Script Quality Validation & Controlled Retry
# ==============================================================================

def test_generate_script_step_retries_on_meta_instruction_leakage(monkeypatch):
    """Verify that a script draft with meta-framing leakage triggers a controlled rewrite."""
    hook = "Why did eighteen sixteen have no summer?"

    # Draft 1: Leaks production instructions
    bad_narr = (
        f"{hook} Picture a fictional farmer in eighteen sixteen witnessing snow in July. "
        "As an internal note, this character is representative of rural New England. "
        "Mount Tambora in Indonesia had erupted and blasted massive ash clouds into the upper sky. "
        "The dense sulfuric cloud reflected sunlight back into space, dropping temperatures worldwide. "
        "Crops withered in the fields, and the cold weather persisted for months. "
        "A single distant volcano caused a year without summer across the globe."
    )
    script_bad = _make_script(hook=hook, body=bad_narr, angle="origin story")

    # Draft 2: Immersive narration without meta-framing
    good_narr = (
        f"{hook} In July of that year, New England farmers woke up to thick ice covering their corn crops. "
        "Thousands of miles away in Indonesia, Mount Tambora had blasted millions of tons of ash into the stratosphere. "
        "That reflective aerosol cloud spread across the globe, blocking sunlight and causing temperatures to plummet. "
        "Frost struck repeatedly in mid-summer, destroying harvests across North America and Europe as food prices soared. "
        "A distant volcanic eruption had quietly triggered a worldwide climate catastrophe that lasted for over a year."
    )
    script_good = _make_script(hook=hook, body=good_narr, angle="origin story")

    mock_llm = MagicMock()
    mock_llm.provider_name = "anthropic"
    mock_llm.model = "claude-mock"

    res1 = LLMResult(
        output=script_bad,
        provider="anthropic",
        model="claude-mock",
        input_tokens=150,
        output_tokens=60,
        latency_seconds=1.2,
    )
    res2 = LLMResult(
        output=script_good,
        provider="anthropic",
        model="claude-mock",
        input_tokens=180,
        output_tokens=60,
        latency_seconds=1.1,
    )
    mock_llm.generate_structured.side_effect = [res1, res2]

    monkeypatch.setattr("app.steps.script.get_llm", lambda task: mock_llm)

    class DummyVideo:
        id = uuid.uuid4()
        topic = "The Year Without a Summer"
        description = "Historical climate anomaly in 1816"
        channel = None
        channel_id = None
        duration_mode = "manual"
        target_duration = 38.0

    class DummyStep:
        name = "script"
        user_prompt = None
        output = None

    result = generate_script_step(DummyVideo(), {}, DummyStep())

    assert mock_llm.generate_structured.call_count == 2
    retry_prompt = mock_llm.generate_structured.call_args_list[1].kwargs["prompt"]
    assert "Narration leaks internal production instructions" in retry_prompt
    assert result["meta"]["attempts"] == 2
    assert result["warnings"] == []
