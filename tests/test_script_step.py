"""Unit tests for the real script step (network-free)."""

import uuid
from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.db import Base
from app.models import Channel, User, Video, VideoStep
from app.steps.script import (
    ALLOWED_ANGLES,
    ScriptOutput,
    build_script_prompt,
    generate_script_step,
    get_recent_angles,
    load_prompt_template,
    normalize_for_comparison,
    validate_script_soft_checks,
)


def _make_sample_script(
    hook="Have a look at airplane windows.",
    narration=None,
    angle="why it's designed this way",
    claims=None,
    hashtags=None,
) -> ScriptOutput:
    """Helper to build a valid ScriptOutput with controlled word counts."""
    if narration is None:
        # Build ~124 words narration starting with the hook
        filler = (
            "Air travelers often notice a tiny hole in the passenger window. This small bleed hole balances "
            "the immense pressure difference between the pressurized cabin and the thin outside atmosphere. "
            "Aircraft windows consist of three separate acrylic panes. The outer pane bears the primary pressure load, "
            "while the inner pane prevents passengers from scratching the crucial middle layer. The bleed hole allows "
            "moisture to escape, preventing frost from obstructing the view. Without this simple engineering detail, "
            "windows could fail or fog completely. Aircraft designers solved a massive cabin pressure challenge with "
            "a miniature opening. Every commercial jet flying at high altitude relies on these tiny ports for passenger "
            "safety and clear visibility throughout the entire flight."
        )
        narration = f"{hook} {filler}"

    return ScriptOutput(
        title="Why Airplane Windows Have Tiny Holes",
        hook=hook,
        narration=narration,
        angle=angle,
        claims_to_verify=(
            claims
            if claims is not None
            else [
                "Airplane windows have three acrylic panes.",
                "The bleed hole balances cabin pressure.",
                "The hole vents moisture to stop frost.",
            ]
        ),
        hashtags=(
            hashtags
            if hashtags is not None
            else ["#aviation", "#engineering", "#travel", "#facts"]
        ),
    )


# ==============================================================================
# 1. Soft-Check Validator Tests
# ==============================================================================

def test_validator_passes_on_compliant_script():
    script = _make_sample_script()
    warnings = validate_script_soft_checks(script)
    assert warnings == []


def test_validator_detects_narration_too_short():
    short_narr = "Have a look at airplane windows. This is a very short text."
    script = _make_sample_script(narration=short_narr)
    warnings = validate_script_soft_checks(script)
    assert any("word count" in w.lower() for w in warnings)


def test_validator_detects_narration_too_long():
    long_narr = "Have a look at airplane windows. " + "word " * 160
    script = _make_sample_script(narration=long_narr)
    warnings = validate_script_soft_checks(script)
    assert any("word count" in w.lower() for w in warnings)


def test_validator_detects_hook_too_long():
    long_hook = "This is an extremely long opening hook sentence that comfortably exceeds the fifteen word maximum limit easily."
    narr = f"{long_hook} " + "word " * 115
    script = _make_sample_script(hook=long_hook, narration=narr)
    warnings = validate_script_soft_checks(script)
    assert any("hook length" in w.lower() for w in warnings)
    assert any("at most 15 words" in w.lower() for w in warnings)


def test_validator_flags_hook_exceeding_ten_words_soft_warning():
    hook_12_words = "Commercial airplane windows feature a tiny hole to balance immense cabin pressure."
    assert len(hook_12_words.split()) == 12
    narr = f"{hook_12_words} " + "word " * 115
    script = _make_sample_script(hook=hook_12_words, narration=narr)
    warnings = validate_script_soft_checks(script)
    assert any("target: ideally 4-10 words" in w.lower() for w in warnings)
    assert not any("at most 15 words" in w.lower() for w in warnings)


def test_validator_passes_on_concise_hook():
    hook_6_words = "These tiny rivets saved miners' pants."
    assert len(hook_6_words.split()) == 6
    narr = f"{hook_6_words} " + "word " * 115
    script = _make_sample_script(hook=hook_6_words, narration=narr)
    warnings = validate_script_soft_checks(script)
    assert not any("hook length" in w.lower() for w in warnings)
    assert not any("cliché" in w.lower() or "cliche" in w.lower() for w in warnings)
    assert not any("generic formula" in w for w in warnings)


def test_validator_detects_cliche_hooks():
    cliche_hooks = [
        "That tiny hole in an airplane window is doing more than you think.",
        "Those tiny metal rivets on jeans aren't just for decoration.",
        "Those tiny metal buttons are not just for decoration.",
        "Jeans rivets have a hidden purpose.",
        "There is an interesting reason for this hole.",
        "This opening is not what you think.",
        "That small metal piece actually serves a crucial role.",
    ]
    for hook in cliche_hooks:
        narr = f"{hook} " + "word " * 115
        script = _make_sample_script(hook=hook, narration=narr)
        warnings = validate_script_soft_checks(script)
        assert any("generic meta-cliché" in w for w in warnings), f"Failed to detect cliché in: {hook}"


def test_validator_detects_generic_openers():
    openers = [
        "Do you know why airplane windows have tiny holes?",
        "Have you ever wondered why jeans have tiny rivets?",
        "Ever wondered why your pockets have rivets?",
        "Did you know that windows have a tiny hole?",
        "Imagine if your airplane window had a hole.",
        "Here's why airplane windows have tiny holes.",
        "Here is why airplane windows have tiny holes.",
        "Here's a fact about airplane windows.",
        "Here is a fact about airplane windows.",
    ]
    for hook in openers:
        narr = f"{hook} " + "word " * 115
        script = _make_sample_script(hook=hook, narration=narr)
        warnings = validate_script_soft_checks(script)
        assert any("generic formula" in w for w in warnings), f"Failed to detect opener in: {hook}"


def test_validator_passes_on_curiosity_questions_and_statements():
    clean_hooks = [
        "Why do airplane windows need a tiny hole?",
        "Why is there a tiny hole in airplane windows?",
        "What happens if this tiny hole is blocked?",
        "Why do jeans have tiny metal rivets?",
        "What stops your jeans pockets from ripping?",
        "Why did miners need metal studs on their jeans?",
        "What does that tiny hole in airplane windows actually do?",
        "Airplane windows have tiny bleed holes to stop cracks.",
        "Without these tiny rivets, heavy pockets tore completely off.",
        "In 1872, a tailor added copper rivets to denim.",
        "Those copper studs on jeans were never meant for fashion.",
        "Thirty thousand feet creates five tons of pressure on windows.",
    ]
    for hook in clean_hooks:
        narr = f"{hook} " + "word " * 115
        script = _make_sample_script(hook=hook, narration=narr)
        warnings = validate_script_soft_checks(script)
        assert not any("generic" in w for w in warnings), f"False positive on: {hook}"
        assert not any("cliché" in w.lower() or "cliche" in w.lower() for w in warnings), f"False positive on: {hook}"


def test_validator_detects_narration_not_starting_with_hook():
    hook = "Look at this window right now."
    wrong_narr = "Here is a completely different opening sentence. " + "word " * 115
    script = _make_sample_script(hook=hook, narration=wrong_narr)
    warnings = validate_script_soft_checks(script)
    assert any("does not begin with the hook" in w.lower() for w in warnings)


def test_validator_normalizes_quotes_and_whitespace_for_hook_match():
    # Hook with straight quotes; narration with curly quotes and em-dash
    hook = "Don't ignore this window."
    curly_narr = "Don’t ignore this window—it balances air pressure. " + "word " * 115
    script = _make_sample_script(hook=hook, narration=curly_narr)
    warnings = validate_script_soft_checks(script)
    # Should not warn about hook mismatch
    assert not any("does not begin with the hook" in w.lower() for w in warnings)


def test_validator_detects_bracketed_directions_and_parentheses():
    script = _make_sample_script()
    script.narration = script.narration + " [pause for effect]"
    warnings = validate_script_soft_checks(script)
    assert any("bracketed directions" in w.lower() for w in warnings)

    script2 = _make_sample_script()
    script2.narration = script2.narration + " (whispering softly)"
    warnings2 = validate_script_soft_checks(script2)
    assert any("bracketed directions" in w.lower() for w2 in warnings2 for w in [w2])


def test_validator_detects_markdown_formatting():
    script = _make_sample_script()
    script.narration = script.narration + " This is **bold** text."
    warnings = validate_script_soft_checks(script)
    assert any("markdown" in w.lower() for w in warnings)


def test_validator_detects_emojis():
    script = _make_sample_script()
    script.narration = script.narration + " Look at the sky ✈️."
    warnings = validate_script_soft_checks(script)
    assert any("emoji" in w.lower() for w in warnings)


def test_validator_detects_invalid_claims_count():
    # 0 claims
    script_empty = _make_sample_script(claims=[])
    warnings_empty = validate_script_soft_checks(script_empty)
    assert any("claims count" in w.lower() for w in warnings_empty)

    # 7 claims (exceeds max of 6)
    script_many = _make_sample_script(claims=[f"Claim {i}" for i in range(1, 8)])
    warnings_many = validate_script_soft_checks(script_many)
    assert any("claims count" in w.lower() for w in warnings_many)


def test_validator_detects_invalid_hashtags_count():
    # 2 tags (fewer than 3)
    script_few = _make_sample_script(hashtags=["#one", "#two"])
    warnings_few = validate_script_soft_checks(script_few)
    assert any("hashtags count" in w.lower() for w in warnings_few)

    # 6 tags (more than 5)
    script_many = _make_sample_script(hashtags=[f"#tag{i}" for i in range(1, 7)])
    warnings_many = validate_script_soft_checks(script_many)
    assert any("hashtags count" in w.lower() for w in warnings_many)


# ==============================================================================
# 2. Recent Angles Lookup Tests
# ==============================================================================

def test_recent_angles_returns_empty_when_no_session():
    angles = get_recent_angles(db=None, channel_id=uuid.uuid4(), current_video_id=uuid.uuid4())
    assert angles == []


def test_recent_angles_query_with_in_memory_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)

    with Session(engine) as session:
        user = User(email="test@example.com", password_hash="secret")
        session.add(user)
        session.flush()

        channel = Channel(user_id=user.id, name="Facts Channel", niche="Engineering")
        other_channel = Channel(user_id=user.id, name="Other Channel", niche="Nature")
        session.add_all([channel, other_channel])
        session.flush()

        # Video 1: Done with "origin story"
        v1 = Video(channel_id=channel.id, topic="Topic 1")
        session.add(v1)
        session.flush()
        s1 = VideoStep(
            video_id=v1.id,
            name="script",
            position=0,
            status="done",
            output={"script": {"angle": "origin story", "title": "T1"}},
            finished_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )

        # Video 2: Done with "why it's designed this way" (more recent)
        v2 = Video(channel_id=channel.id, topic="Topic 2")
        session.add(v2)
        session.flush()
        s2 = VideoStep(
            video_id=v2.id,
            name="script",
            position=0,
            status="done",
            output={"script": {"angle": "why it's designed this way", "title": "T2"}},
            finished_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        )

        # Video 3: Other channel (should be ignored)
        v3 = Video(channel_id=other_channel.id, topic="Topic 3")
        session.add(v3)
        session.flush()
        s3 = VideoStep(
            video_id=v3.id,
            name="script",
            position=0,
            status="done",
            output={"script": {"angle": "myth vs reality", "title": "T3"}},
            finished_at=datetime(2026, 1, 3, tzinfo=timezone.utc),
        )

        # Current video
        current_v = Video(channel_id=channel.id, topic="Current Topic")
        session.add(current_v)
        session.add_all([s1, s2, s3])
        session.commit()

        angles = get_recent_angles(db=session, channel_id=channel.id, current_video_id=current_v.id)
        # Should return v2's angle then v1's angle
        assert angles == ["why it's designed this way", "origin story"]


# ==============================================================================
# 3. Prompt Assembly & Versioning Tests
# ==============================================================================

def test_load_prompt_template_reads_version_and_hash():
    text, version_str = load_prompt_template()
    assert len(text) > 100
    assert "script-v1" in version_str
    assert ":" in version_str
    # Verify hash portion is 8 characters
    hash_part = version_str.split(":", 1)[1]
    assert len(hash_part) == 8


def test_build_script_prompt_fresh_mode():
    rules = "# Test Rules"
    prompt = build_script_prompt(
        rules_text=rules,
        topic="Submarine fiber optic cables",
        description="Carrying global internet traffic",
        channel_name="Everyday Mysteries",
        channel_niche="Tech History",
        channel_settings={"tone": "curious"},
        recent_angles=["myth vs reality"],
    )

    assert "Submarine fiber optic cables" in prompt
    assert "Carrying global internet traffic" in prompt
    assert "Everyday Mysteries" in prompt
    assert "myth vs reality" in prompt
    assert "REVISION MODE" not in prompt


def test_build_script_prompt_revision_mode():
    rules = "# Test Rules"
    prev_script = {
        "title": "Subsea Cables",
        "hook": "Deep beneath the ocean...",
        "narration": "Full narration text...",
        "angle": "origin story",
    }
    prompt = build_script_prompt(
        rules_text=rules,
        topic="Submarine fiber optic cables",
        description=None,
        previous_script=prev_script,
        revision_instruction="Make the ending punchier and mention landing stations.",
    )

    assert "REVISION MODE" in prompt
    assert "Make the ending punchier and mention landing stations." in prompt
    assert "Deep beneath the ocean..." in prompt


# ==============================================================================
# 4. Full Step Execution (Mock Provider)
# ==============================================================================

def test_full_script_step_with_mock_provider(monkeypatch):
    """Test full execution of generate_script_step with the mock LLM provider."""
    from app.config import settings
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_provider_script", "mock")

    class DummyChannel:
        id = uuid.uuid4()
        name = "Engineering Shorts"
        niche = "Tech"
        settings = {}

    class DummyVideo:
        id = uuid.uuid4()
        topic = "How barcodes work"
        description = "Laser scanning and numbers"
        channel = DummyChannel()
        channel_id = DummyChannel.id

    class DummyStep:
        name = "script"
        user_prompt = None
        output = None

    result = generate_script_step(
        video=DummyVideo(),
        prev={},
        step=DummyStep(),
    )

    assert "script" in result
    assert "warnings" in result
    assert "meta" in result

    script = result["script"]
    assert script["angle"] in ALLOWED_ANGLES
    assert "title" in script
    assert "hook" in script
    assert "narration" in script

    meta = result["meta"]
    assert meta["provider"] == "mock"
    assert meta["model"] == "mock-model"
    assert meta["task"] == "script"
    assert meta["attempts"] == 1
    assert meta["revision"] is False
    assert "script-v1" in meta["prompt_version"]


def test_full_script_step_revision_mode(monkeypatch):
    """Test that setting step.user_prompt and prior step.output triggers revision mode."""
    from app.config import settings
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_provider_script", "mock")

    class DummyVideo:
        id = uuid.uuid4()
        topic = "How barcodes work"
        description = None
        channel = None
        channel_id = None

    class DummyStep:
        name = "script"
        user_prompt = "Change angle to myth vs reality"
        output = {
            "script": {
                "title": "Previous Barcode Title",
                "hook": "Original opening hook.",
                "narration": "Original narration body.",
                "angle": "origin story",
                "claims_to_verify": ["Barcodes were invented in 1952."],
                "hashtags": ["#tech", "#retail", "#history"],
            }
        }

    result = generate_script_step(
        video=DummyVideo(),
        prev={},
        step=DummyStep(),
    )

    assert result["meta"]["revision"] is True


def test_generate_script_step_retries_on_cliche_warning(monkeypatch):
    """Verify that an initial cliché warning triggers a retry attempt with corrective feedback."""
    from unittest.mock import MagicMock
    from app.llm.types import LLMResult
    from app.steps.script import ScriptOutput, generate_script_step

    cliche_script = _make_sample_script(
        hook="That tiny hole is doing more than you think.",
        narration="That tiny hole is doing more than you think. " + "word " * 115,
    )
    clean_script = _make_sample_script(
        hook="Airplane windows have tiny holes to prevent cracks.",
        narration="Airplane windows have tiny holes to prevent cracks. " + "word " * 115,
    )

    mock_llm = MagicMock()
    mock_llm.provider_name = "openai"
    mock_llm.model = "gpt-mock"

    res1 = LLMResult(
        output=cliche_script,
        provider="openai",
        model="gpt-mock",
        input_tokens=100,
        output_tokens=50,
        latency_seconds=1.0,
    )
    res2 = LLMResult(
        output=clean_script,
        provider="openai",
        model="gpt-mock",
        input_tokens=120,
        output_tokens=50,
        latency_seconds=1.2,
    )
    mock_llm.generate_structured.side_effect = [res1, res2]

    monkeypatch.setattr("app.steps.script.get_llm", lambda task: mock_llm)

    class DummyVideo:
        id = uuid.uuid4()
        topic = "Airplane windows"
        description = None
        channel = None
        channel_id = None

    class DummyStep:
        name = "script"
        user_prompt = None
        output = None

    result = generate_script_step(DummyVideo(), {}, DummyStep())

    assert mock_llm.generate_structured.call_count == 2
    retry_prompt = mock_llm.generate_structured.call_args_list[1].kwargs["prompt"]
    assert "generic meta-cliché" in retry_prompt
    assert result["meta"]["attempts"] == 2
    assert result["script"]["hook"] == "Airplane windows have tiny holes to prevent cracks."
    assert result["warnings"] == []


def test_generate_script_step_retries_on_generic_opener_warning(monkeypatch):
    """Verify that a generic question formula (e.g. 'Do you know why...') triggers a retry."""
    from unittest.mock import MagicMock
    from app.llm.types import LLMResult
    from app.steps.script import ScriptOutput, generate_script_step

    formula_script = _make_sample_script(
        hook="Do you know why airplane windows have tiny holes?",
        narration="Do you know why airplane windows have tiny holes? " + "word " * 115,
    )
    direct_script = _make_sample_script(
        hook="Why do airplane windows need a tiny hole?",
        narration="Why do airplane windows need a tiny hole? " + "word " * 115,
    )

    mock_llm = MagicMock()
    mock_llm.provider_name = "openai"
    mock_llm.model = "gpt-mock"

    res1 = LLMResult(
        output=formula_script,
        provider="openai",
        model="gpt-mock",
        input_tokens=100,
        output_tokens=50,
        latency_seconds=1.0,
    )
    res2 = LLMResult(
        output=direct_script,
        provider="openai",
        model="gpt-mock",
        input_tokens=120,
        output_tokens=50,
        latency_seconds=1.1,
    )
    mock_llm.generate_structured.side_effect = [res1, res2]

    monkeypatch.setattr("app.steps.script.get_llm", lambda task: mock_llm)

    class DummyVideo:
        id = uuid.uuid4()
        topic = "Airplane windows"
        description = None
        channel = None
        channel_id = None

    class DummyStep:
        name = "script"
        user_prompt = None
        output = None

    result = generate_script_step(DummyVideo(), {}, DummyStep())

    assert mock_llm.generate_structured.call_count == 2
    retry_prompt = mock_llm.generate_structured.call_args_list[1].kwargs["prompt"]
    assert "generic formula" in retry_prompt
    assert result["meta"]["attempts"] == 2
    assert result["script"]["hook"] == "Why do airplane windows need a tiny hole?"
    assert result["warnings"] == []


# ==============================================================================
# 8. Script Quality & Storytelling Checks (Phase 5)
# ==============================================================================

def test_evaluate_script_quality_detects_long_sentences_and_textbook_phrasing():
    from app.steps.script import evaluate_script_quality

    hook = "Why do people yawn when someone else yawns?"
    # Includes a 32-word sentence, academic hedging ("no single explanation is settled"), and comma digits
    narration = (
        f"{hook} Seeing someone yawn can make your own jaw stretch within seconds, even across a room of 30,000 people. "
        "Researchers link this contagious reflex to automatic social mirroring inside human brain networks that track the faces and emotions of people around us every single day without conscious effort or deliberate thought. "
        "However, studies haven't shown that oxygen levels control it, and no single explanation is settled. "
        "Your brain quietly mirrors the people around you before you even notice."
    )
    script = _make_sample_script(hook=hook, narration=narration)
    issues = evaluate_script_quality(script)

    assert any("overly long sentences" in i.lower() for i in issues)
    assert any("textbook or academic phrasing" in i.lower() for i in issues)
    assert any("comma-formatted digits" in i.lower() for i in issues)


def test_evaluate_script_quality_detects_disconnected_fact_list_and_weak_ending():
    from app.steps.script import evaluate_script_quality

    hook = "What happens near a black hole's edge?"
    narration = (
        f"{hook} A black hole packs enormous mass into a tiny space, creating gravity so strong that light cannot escape. "
        "Time also passes more slowly near a black hole than it does far away. "
        "Furthermore, its edge is called the event horizon, and crossing it means never returning. "
        "In addition, scientists first detected ripples in spacetime from colliding black holes. "
        "Follow for more space facts."
    )
    script = _make_sample_script(hook=hook, narration=narration)
    issues = evaluate_script_quality(script)

    assert any("disconnected list of facts" in i.lower() for i in issues)
    assert any("final sentence ends weakly" in i.lower() for i in issues)


def test_evaluate_script_quality_detects_unexplained_jargon_and_repeated_phrases():
    from app.steps.script import evaluate_script_quality

    hook = "How do tardigrades survive extreme radiation?"
    narration = (
        f"{hook} Microscopic tardigrades endure extreme dehydration through cryptobiosis, anhydrobiosis, and osmobiosis in harsh environments. "
        "Extreme radiation damages cellular structures, yet extreme radiation fails to destroy their resilient biological shields. "
        "Extreme radiation triggers specialized proteins that bind directly to their genetic material and prevent shattering. "
        "That microscopic shield keeps them alive even in the vacuum of space."
    )
    script = _make_sample_script(hook=hook, narration=narration)
    issues = evaluate_script_quality(script)

    assert any("unexplained technical vocabulary" in i.lower() for i in issues)
    assert any("repeats the phrase" in i.lower() or "start with the identical phrase" in i.lower() for i in issues)


def test_estimate_spoken_duration_seconds():
    from app.steps.script import estimate_spoken_duration_seconds

    script = _make_sample_script()
    dur = estimate_spoken_duration_seconds(script.narration)
    assert 40.0 <= dur <= 60.0
    assert estimate_spoken_duration_seconds("") == 0.0



