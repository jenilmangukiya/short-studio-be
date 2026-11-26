"""Tests for balanced character and subject selection framework in Creative Direction."""

import pytest
from app.schemas import CreativeDirectionOutput, EmotionalBeat, ProtagonistConfig
from app.steps.creative_direction import (
    HUMAN_SENSORY_TOPIC_REGEX,
    build_creative_direction_prompt,
    load_prompt_template,
    validate_creative_direction,
)


def _make_sample_cd(
    human_presence="single",
    archetype_summary="A relatable adult sleeper",
    topic_summary="Explaining the hypnic jerk during sleep onset",
) -> CreativeDirectionOutput:
    return CreativeDirectionOutput(
        format_archetype="explainer_myth_bust",
        storytelling_angle="Contrasting the fear of falling with the brain's sleep onset reflex",
        central_thesis=topic_summary,
        tone="Curious, reassuring, and engaging",
        visual_genre="styled_3d",
        protagonist_config=ProtagonistConfig(
            human_presence=human_presence,
            archetype_summary=archetype_summary if human_presence != "none" else None,
        ),
        emotional_progression=[
            EmotionalBeat(phase="hook", target_emotion="Curiosity and recognition", visual_intent="Show the sudden jolt"),
            EmotionalBeat(phase="setup", target_emotion="Calmness", visual_intent="Show settling in bed"),
            EmotionalBeat(phase="complication", target_emotion="Sudden surprise", visual_intent="Show falling sensation"),
            EmotionalBeat(phase="reveal", target_emotion="Clarity", visual_intent="Show brain mechanism"),
            EmotionalBeat(phase="resolution", target_emotion="Reassurance", visual_intent="Show peaceful sleeper"),
        ],
        creative_constraints=["Keep consistent bedroom lighting", "No supernatural imagery"],
    )


# ==============================================================================
# 1. Advisory Soft-Check Warnings
# ==============================================================================

def test_advisory_warning_triggers_on_human_sensory_topic_with_no_character():
    """When a topic involves human sleep/sensory reflexes but human_presence='none', emit advisory warning."""
    sensory_topics = [
        "Why Do We Sometimes Feel Like We're Falling Just Before Falling Asleep?",
        "What Causes a Hypnic Jerk?",
        "Why Do People Yawn When Someone Else Yawns?",
        "Why do we dream every night?",
        "What causes sleep paralysis?",
        "Why do you get an intense brain freeze from ice cream?",
        "Why do we get goosebumps when we are cold or scared?",
    ]

    cd_none = _make_sample_cd(human_presence="none")

    for topic in sensory_topics:
        warnings = validate_creative_direction(cd_none, topic=topic)
        assert any("human physical or sensory experience" in w.lower() for w in warnings), (
            f"Expected advisory warning for topic '{topic}' with human_presence='none'"
        )


def test_advisory_warning_does_not_trigger_when_character_is_provided():
    """When human_presence is 'single' with archetype summary, no advisory warning is produced."""
    topic = "Why Do We Sometimes Feel Like We're Falling Just Before Falling Asleep?"
    cd_single = _make_sample_cd(human_presence="single", archetype_summary="A relatable adult sleeper in simple sleepwear")

    warnings = validate_creative_direction(cd_single, topic=topic)
    assert not any("human physical or sensory experience" in w.lower() for w in warnings)


def test_advisory_warning_does_not_trigger_on_non_sensory_topics_with_no_character():
    """Mechanical, cosmic, chemical, and animal topics with human_presence='none' must NOT trigger sensory warnings."""
    non_sensory_topics = [
        "Why Do Airplane Windows Have a Tiny Hole?",
        "5 Mind Blowing Facts About Black Holes",
        "Why Honey Never Spoils in Ancient Tombs",
        "How a Mechanical Watch Escapement Works",
        "Why Ice Is Actually Slippery",
        "Why Do Cats Always Land on Their Feet?",
    ]

    for topic in non_sensory_topics:
        cd_none = _make_sample_cd(
            human_presence="none",
            topic_summary=f"Scientific principles behind {topic}",
        )
        warnings = validate_creative_direction(cd_none, topic=topic)
        assert not any("human physical or sensory experience" in w.lower() for w in warnings), (
            f"False positive advisory warning on non-sensory topic: '{topic}'"
        )


# ==============================================================================
# 2. Balanced Selection Prompt Guidance & Faceless Video Clarification
# ==============================================================================

def test_creative_direction_prompt_template_contains_balanced_framework():
    """Verify creative_direction.md contains the positive guidelines and faceless video clarification."""
    rules_text, _ = load_prompt_template()

    # 1. Balanced character selection guidelines
    assert "Human Physical, Sensory & Psychological Experiences" in rules_text
    assert "Strongly consider `single` or `ensemble`" in rules_text
    assert "Animal & Wildlife Topics" in rules_text
    assert "Inanimate, Technical, Cosmic & Abstract Topics" in rules_text

    # 2. Faceless video clarification
    assert "Clarification on \"Faceless Video\"" in rules_text
    assert "does not appear on camera as a live" in rules_text
    assert "does NOT prohibit human beings, characters, or animated subjects" in rules_text


def test_build_creative_direction_prompt_injects_balanced_framework():
    """Verify build_creative_direction_prompt injects the balanced guidelines in dynamic instructions."""
    rules_text, _ = load_prompt_template()
    prompt = build_creative_direction_prompt(
        rules_text=rules_text,
        topic="Why Do We Yawn When We See Others Yawn?",
        description="A faceless video about contagious yawning.",
    )

    assert "Apply the balanced character and subject selection guidelines" in prompt
    assert "strongly consider defining a relatable human character/subject" in prompt
    assert "consider an appropriate animal subject" in prompt
    assert "Do NOT force a character if the topic is purely mechanical, cosmic, chemical, or abstract" in prompt


# ==============================================================================
# 3. Topic Category Representation Tests
# ==============================================================================

@pytest.mark.parametrize(
    "topic,expected_char_count,expected_archetype",
    [
        (
            "Why Do We Sometimes Feel Like We're Falling Just Before Falling Asleep?",
            "single",
            "A relatable adult sleeper",
        ),
        (
            "Why Do Cats Always Land on Their Feet?",
            "single",
            "An agile domestic cat demonstrating the righting reflex",
        ),
        (
            "Why Do Airplane Windows Have a Tiny Hole?",
            "none",
            None,
        ),
        (
            "5 Mind Blowing Facts About Black Holes",
            "none",
            None,
        ),
        (
            "Why Honey Never Spoils in Ancient Tombs",
            "none",
            None,
        ),
    ],
)
def test_topic_category_model_conformance(topic, expected_char_count, expected_archetype):
    """Verify that CreativeDirectionOutput parses and validates cleanly across all topic archetypes."""
    cd = _make_sample_cd(
        human_presence=expected_char_count,
        archetype_summary=expected_archetype,
        topic_summary=f"Core explanation for: {topic}",
    )
    assert cd.protagonist_config.human_presence == expected_char_count
    if expected_char_count != "none":
        assert cd.protagonist_config.archetype_summary == expected_archetype
    else:
        assert cd.protagonist_config.archetype_summary is None

    warnings = validate_creative_direction(cd, topic=topic)
    assert len(warnings) == 0
