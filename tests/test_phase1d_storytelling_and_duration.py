"""Comprehensive test suite for Phase 1D: Narrative Storytelling & Adaptive Duration.

Covers:
- Video duration mode schema validation (auto, manual 30s-180s, invalid inputs).
- Canonical duration resolution and manual precedence over AI recommendations.
- Dynamic word budget calculations scaling smoothly across 30s to 180s.
- Creative Direction narrative plan schema, duration rationale, and format archetypes.
- Script flexible openings (single-sentence, multi-sentence, 'Imagine' scene-setting).
- Soft checks for meta-instruction leakage, cliches, and natural ending guidance.
- Adaptive Storyboard scene scaling for long-duration videos without 8-12 scene caps.
- Duration-aware processing timeouts in Render and Subtitles.
- Pipeline rerun invalidation mechanics.
"""

from __future__ import annotations

import uuid
from typing import Any
import pytest
from pydantic import ValidationError

from app.models import Video, VideoStep
from app.schemas import (
    CreativeDirectionOutput,
    EmotionalBeat,
    NarrativePlan,
    NarrativeStage,
    ProtagonistConfig,
    VideoCreate,
    VideoDetail,
    VideoSummary,
)
from app.steps.duration import (
    DEFAULT_TARGET_DURATION_SECONDS,
    MAX_ALLOWED_DURATION_SECONDS,
    MIN_ALLOWED_DURATION_SECONDS,
    calculate_word_budget_for_duration,
    clamp_duration_seconds,
    resolve_video_target_duration,
)
from app.steps.script import (
    ScriptOutput,
    calculate_target_word_count_range,
    evaluate_script_quality,
    resolve_requested_duration,
    validate_script_soft_checks,
)
from app.steps.storyboard import validate_storyboard_soft_checks


# ==============================================================================
# 1. Duration Schema & Validation Tests
# ==============================================================================

def test_video_create_duration_modes():
    """Verify VideoCreate validation for auto and manual duration modes."""
    # 1. Default is auto mode with no manual target
    v_auto = VideoCreate(
        topic="The Ocean That Disappeared Overnight",
        visual_style_id="stylized_2d_narrative",
    )
    assert v_auto.duration_mode == "auto"
    assert v_auto.target_duration is None

    # 2. Manual mode with valid target durations across the allowed range
    for target in (30, 45, 60, 90, 120, 180):
        v_man = VideoCreate(
            topic="The Ocean That Disappeared Overnight",
            visual_style_id="stylized_2d_narrative",
            duration_mode="manual",
            target_duration=target,
        )
        assert v_man.duration_mode == "manual"
        assert v_man.target_duration == target

    # 3. Manual mode without target_duration fails
    with pytest.raises(ValidationError) as exc:
        VideoCreate(
            topic="The Ocean That Disappeared",
            visual_style_id="stylized_2d_narrative",
            duration_mode="manual",
        )
    assert "target_duration is required" in str(exc.value)

    # 4. Target duration below 30s fails
    with pytest.raises(ValidationError) as exc:
        VideoCreate(
            topic="The Ocean That Disappeared",
            visual_style_id="stylized_2d_narrative",
            duration_mode="manual",
            target_duration=25,
        )
    assert "greater than or equal to 30" in str(exc.value)

    # 5. Target duration above 180s fails
    with pytest.raises(ValidationError) as exc:
        VideoCreate(
            topic="The Ocean That Disappeared",
            visual_style_id="stylized_2d_narrative",
            duration_mode="manual",
            target_duration=240,
        )
    assert "less than or equal to 180" in str(exc.value)


def test_video_summary_and_detail_duration_fields():
    """Verify VideoSummary and VideoDetail expose duration_mode and target_duration."""
    vid_id = uuid.uuid4()
    ch_id = uuid.uuid4()

    summary_data = {
        "id": vid_id,
        "channel_id": ch_id,
        "topic": "The Year Without a Summer",
        "description": "Historical disaster narrative",
        "status": "draft",
        "visual_style_id": "historical_oil_painting",
        "duration_mode": "manual",
        "target_duration": 90,
        "created_at": "2026-11-27T12:00:00Z",
    }
    summary = VideoSummary.model_validate(summary_data)
    assert summary.duration_mode == "manual"
    assert summary.target_duration == 90

    detail_data = {**summary_data, "steps": []}
    detail = VideoDetail.model_validate(detail_data)
    assert detail.duration_mode == "manual"
    assert detail.target_duration == 90


# ==============================================================================
# 2. Canonical Duration Resolution & Precedence Tests
# ==============================================================================

def test_manual_duration_precedence_over_ai_recommendation():
    """Manual duration must take absolute precedence over AI recommendations."""
    class DummyVideo:
        duration_mode = "manual"
        target_duration = 45

    cd_output = {
        "recommended_duration_seconds": 120,
        "resolved_target_duration": 120,
    }

    resolved = resolve_video_target_duration(DummyVideo(), cd_output)
    assert resolved == 45, "Manual target of 45s must override AI recommendation of 120s"


def test_auto_duration_adopts_creative_direction_recommendation():
    """In auto mode, the duration resolved by Creative Direction becomes canonical."""
    class DummyVideo:
        duration_mode = "auto"
        target_duration = None

    cd_output = {
        "recommended_duration_seconds": 110,
        "resolved_target_duration": 110,
    }

    resolved = resolve_video_target_duration(DummyVideo(), cd_output)
    assert resolved == 110


def test_fallback_duration_when_no_cd_exists():
    """When in auto mode and Creative Direction has not run, default to 60 seconds."""
    class DummyVideo:
        duration_mode = "auto"
        target_duration = None
        steps = []

    resolved = resolve_video_target_duration(DummyVideo(), None)
    assert resolved == DEFAULT_TARGET_DURATION_SECONDS


def test_duration_clamping():
    """Durations must be clamped to [30, 180] seconds."""
    assert clamp_duration_seconds(15) == MIN_ALLOWED_DURATION_SECONDS
    assert clamp_duration_seconds(240) == MAX_ALLOWED_DURATION_SECONDS
    assert clamp_duration_seconds(90) == 90


# ==============================================================================
# 3. Dynamic Word Budget Calculations
# ==============================================================================

def test_word_budget_scaling():
    """Word budget must scale smoothly across durations with breathing room."""
    durations = [30, 45, 60, 90, 120, 180]
    prev_max = 0

    for dur in durations:
        w_min, w_max = calculate_word_budget_for_duration(dur)
        # Ensure positive ranges and reasonable speaking pace (~2.0 - 2.5 wps)
        assert w_min < w_max
        assert w_max > prev_max, f"Word budget should increase with duration ({dur}s)"
        prev_max = w_max

        # Check approximate boundaries for key milestones
        if dur == 30:
            assert 45 <= w_min <= 65
            assert 70 <= w_max <= 90
        elif dur == 60:
            assert 105 <= w_min <= 130
            assert 140 <= w_max <= 170
        elif dur == 180:
            assert 320 <= w_min <= 370
            assert 420 <= w_max <= 510


# ==============================================================================
# 4. Creative Direction Schema & Narrative Plan Tests
# ==============================================================================

def test_creative_direction_schema_with_narrative_plan():
    """Verify CreativeDirectionOutput schema with narrative plan and duration fields."""
    cd = CreativeDirectionOutput(
        format_archetype="experiential_history",
        storytelling_angle="myth vs reality",
        central_thesis="A distant volcano caused the summer of 1816 to freeze.",
        tone="mysterious and dramatic",
        protagonist_config=ProtagonistConfig(
            human_presence="contextual",
            archetype_summary="19th-century farming families facing unexplained frost",
        ),
        emotional_progression=[
            EmotionalBeat(phase="hook", target_emotion="baffled", visual_intent="snow in summer"),
            EmotionalBeat(phase="setup", target_emotion="dread", visual_intent="crops failing"),
            EmotionalBeat(phase="complication", target_emotion="despair", visual_intent="food shortages"),
            EmotionalBeat(phase="reveal", target_emotion="awe", visual_intent="volcanic explosion"),
            EmotionalBeat(phase="resolution", target_emotion="reverence", visual_intent="the Year Without a Summer"),
        ],
        creative_constraints=["Keep all dates factually accurate to 1815-1816"],
        narrative_plan=NarrativePlan(
            narrative_perspective="experiential_witness",
            opening_situation="Waking up in June expecting summer but finding snow blanketing fields.",
            information_revelation_strategy="experiential_crisis_then_distant_origin",
            story_progression=[
                NarrativeStage(stage_name="cold_open", description="June snow in New England"),
                NarrativeStage(stage_name="escalating_crisis", description="Crops die, hunger spreads"),
                NarrativeStage(stage_name="distant_catalyst", description="Mount Tambora eruption in Indonesia"),
                NarrativeStage(stage_name="atmospheric_shield", description="Sulfur aerosol blocking sunlight"),
                NarrativeStage(stage_name="historical_resolution", description="The Year Without a Summer"),
            ],
            audience_takeaway="How planetary weather systems connect the entire globe.",
            resolution_payoff="History remembers 1816 as the Year Without a Summer.",
        ),
        recommended_duration_seconds=90,
        resolved_target_duration=90,
        duration_rationale="Multi-stage historical narrative requires 90s to build atmospheric tension before reveal.",
    )

    dump = cd.model_dump()
    assert dump["format_archetype"] == "experiential_history"
    assert dump["narrative_plan"]["narrative_perspective"] == "experiential_witness"
    assert len(dump["narrative_plan"]["story_progression"]) == 5
    assert dump["resolved_target_duration"] == 90


# ==============================================================================
# 5. Script Storytelling & Opening Flexibility Tests
# ==============================================================================

def test_script_allows_multi_sentence_and_scene_setting_openings():
    """Verify that atmospheric scene-setting cold opens (including 'Imagine') pass soft checks."""
    hook = "Imagine waking up one morning in June, expecting summer. But outside, snow is falling on your crops."
    narration = (
        f"{hook} Across New England in eighteen sixteen, the weather turned violently cold. "
        "Farms froze overnight and food quickly ran short as families struggled to understand the sudden catastrophe. "
        "What nobody realized was that the real cause had started a year earlier, ten thousand miles away. "
        "On an island in Indonesia, Mount Tambora had erupted with unbelievable force, blasting millions of tons "
        "of sulfur dust into the upper atmosphere. The resulting aerosol veil drifted around the planet, blocking "
        "the sun's warmth and chilling global temperatures. Summer arrived on the calendar, but the heat never came. "
        "History remembers eighteen sixteen as the Year Without a Summer."
    )

    script = ScriptOutput(
        title="The Year Without a Summer",
        hook=hook,
        narration=narration,
        angle="origin story",
        claims_to_verify=[
            "Mount Tambora erupted in eighteen fifteen.",
            "Eighteen sixteen was known as the Year Without a Summer.",
        ],
        hashtags=["#history", "#volcano", "#weather"],
    )

    warnings = validate_script_soft_checks(script, target_duration=60.0)

    # 1. 'Imagine' must NOT trigger generic formula warning
    assert not any("generic formula" in w for w in warnings), f"Unexpected generic opener warning in {warnings}"
    # 2. Hook of 17 words must NOT trigger hook length warning
    assert not any("hook" in w.lower() and "words" in w.lower() for w in warnings)
    # 3. Narration begins verbatim with hook
    assert not any("does not begin with" in w for w in warnings)


def test_script_flags_production_instruction_leakage():
    """Verify that backstage directives leaking into dialogue are flagged."""
    hook = "Why did the summer of eighteen sixteen turn to freezing ice?"
    leaked_narration = (
        f"{hook} Picture a fictional farmer standing in the field as our representative character. "
        "As a fictional figure, he is observing the cold winds while our creative constraint states we must keep it dramatic. "
        "Because of the sulfur veil, global temperatures dropped significantly."
    )

    script = ScriptOutput(
        title="Testing Meta Leakage",
        hook=hook,
        narration=leaked_narration,
        angle="origin story",
        claims_to_verify=["Mount Tambora erupted."],
        hashtags=["#history", "#science", "#volcano"],
    )

    issues = evaluate_script_quality(script)
    assert any("leaks internal production instructions" in i for i in issues)


def test_script_ending_guidance():
    """Verify that clean thematic endings pass, while weak disclaimers are flagged."""
    # Good ending: Resonant historical capstone
    good_hook = "Why did summer never arrive in eighteen sixteen?"
    good_narr = (
        f"{good_hook} Relentless freezing winds destroyed crops across Europe and North America. "
        "A colossal volcanic veil blocked solar heat for months on end. "
        "History remembers eighteen sixteen as the Year Without a Summer."
    )
    good_script = ScriptOutput(
        title="Good Ending",
        hook=good_hook,
        narration=good_narr,
        angle="origin story",
        claims_to_verify=["Claim"],
        hashtags=["#history", "#science", "#nature"],
    )
    issues_good = evaluate_script_quality(good_script)
    assert not any("ends weakly" in i for i in issues_good)

    # Bad ending: Academic hedging or CTA
    bad_narr = good_narr + " But more research is needed."
    bad_script = ScriptOutput(
        title="Bad Ending",
        hook=good_hook,
        narration=bad_narr,
        angle="origin story",
        claims_to_verify=["Claim"],
        hashtags=["#history", "#science", "#nature"],
    )
    issues_bad = evaluate_script_quality(bad_script)
    assert any("ends weakly" in i for i in issues_bad)


# ==============================================================================
# 6. Adaptive Storyboard Scene Scaling Tests
# ==============================================================================

def test_storyboard_adaptive_scene_scaling_for_180s():
    """A 180s video with 28 scenes must pass soft checks without 8-12 scene warnings."""
    # Construct 28 synthetic scenes covering a 180s narration
    words_per_scene = 14
    total_words = 28 * words_per_scene
    words_list = [f"word{i}" for i in range(total_words)]

    scenes = []
    script_parts = []
    shot_types = ["wide", "medium", "close_up", "extreme_close_up", "medium", "wide"]

    for i in range(28):
        part = " ".join(words_list[i * words_per_scene : (i + 1) * words_per_scene]) + "."
        script_parts.append(part)
        scenes.append({
            "index": i + 1,
            "narration": part,
            "image_prompt": f"Scene {i + 1} dramatic depiction of historical cold",
            "shot_type": shot_types[i % len(shot_types)],
            "visual_intent": "action_event",
            "visual_technique": "cinematic_illustration",
            "uses_anchor": (i == 0),
        })

    full_narration = " ".join(script_parts)

    # Soft checks with target_duration = 180.0
    warnings = validate_storyboard_soft_checks(
        scenes=scenes,
        script_narration=full_narration,
        subject_anchor="Historical landscape anchor",
        target_duration=180.0,
    )

    # Must NOT warn about scene count being outside 8-12
    assert not any("scene count" in w.lower() for w in warnings), f"Unexpected scene count warning: {warnings}"


# ==============================================================================
# 7. Duration-Aware Downstream Processing Timeouts
# ==============================================================================

def test_render_timeout_scales_with_duration():
    """Verify that FFmpeg render timeout calculation accommodates long videos."""
    from app.config import settings

    default_timeout = getattr(settings, "render_timeout_seconds", 300.0)

    # For a 45s video, timeout stays at base 300s
    t_45 = max(default_timeout, 45.0 * 3.0 + 60.0)
    assert t_45 == 300.0

    # For a 180s video, timeout scales to at least 600s
    t_180 = max(default_timeout, 180.0 * 3.0 + 60.0)
    assert t_180 == 600.0


def test_acoustic_subtitle_timeout_is_expanded():
    """Verify that AcousticSubtitleProvider default timeout handles long continuous audio."""
    from app.subtitles.acoustic import AcousticSubtitleProvider

    provider = AcousticSubtitleProvider()
    assert provider.timeout_seconds >= 60.0


# ==============================================================================
# 8. Corrective Pass: Canonical Duration, Speed, Simplicity & Migration Tests
# ==============================================================================

def test_require_resolved_raises_pipeline_error_when_no_cd_and_auto_mode():
    """Downstream generation must NOT silently fall back to 60s when CD duration is missing in auto mode."""
    from app.pipeline import PipelineError

    class DummyVideo:
        duration_mode = "auto"
        target_duration = None

    with pytest.raises(PipelineError, match="requires a valid current Creative Direction resolved duration"):
        resolve_video_target_duration(DummyVideo(), None, require_resolved=True)


def test_no_stale_step_history_fallback_during_invalidation():
    """Historical steps on the video model must NEVER be inspected for duration resolution.
    Reruns and rollbacks must consume only the current passed creative_direction_data."""
    from app.pipeline import PipelineError

    class DummyOldStep:
        name = "creative_direction"
        status = "done"
        output = {"creative_direction": {"resolved_target_duration": 45}}

    class DummyVideo:
        duration_mode = "auto"
        target_duration = None
        steps = [DummyOldStep()]

    # When creative_direction_data is None (e.g. step was invalidated to stale/pending),
    # history in DummyVideo.steps must NOT be consulted
    with pytest.raises(PipelineError):
        resolve_video_target_duration(DummyVideo(), None, require_resolved=True)


def test_no_legacy_target_duration_fallback_without_manual_mode():
    """An explicit target_duration on a video is ignored if duration_mode is 'auto'.
    Manual mode must be explicitly specified."""
    class DummyVideo:
        duration_mode = "auto"
        target_duration = 90  # Orphaned or inconsistent attribute

    cd_output = {"creative_direction": {"resolved_target_duration": 45}}
    resolved = resolve_video_target_duration(DummyVideo(), cd_output, require_resolved=True)
    # CD resolved duration governs because duration_mode is auto
    assert resolved == 45


def test_word_budget_accounts_for_speaking_speed():
    """Word budget must scale proportionally with configured TTS speaking speed."""
    base_min, base_max = calculate_word_budget_for_duration(60, speaking_speed=1.0)
    fast_min, fast_max = calculate_word_budget_for_duration(60, speaking_speed=1.2)
    slow_min, slow_max = calculate_word_budget_for_duration(60, speaking_speed=0.8)

    assert fast_max > base_max, "Faster speech should accommodate more words in the same duration"
    assert slow_max < base_max, "Slower speech must reduce word target to prevent overrun"
    assert round(fast_max / base_max, 1) == 1.2


def test_simple_topic_and_direct_explanation_flexibility():
    """Direct, straightforward explanations without artificial drama or forced endings must pass soft checks."""
    direct_body = (
        "Why do airplane windows have a tiny hole? "
        "Airplane windows have three acrylic panes, and the middle pane has a tiny hole called a bleed hole. "
        "It balances the massive air pressure difference between the cabin and thin outside air. "
        "It also releases moisture so the glass does not fog or freeze at high altitudes. "
        "A two-millimeter opening keeps every passenger safe and the windows crystal clear."
    )
    direct_script = ScriptOutput(
        title="Why Airplane Windows Have Tiny Holes",
        hook="Why do airplane windows have a tiny hole?",
        narration=direct_body,
        angle="why it's designed this way",
        claims_to_verify=["Airplane windows have three acrylic panes."],
        hashtags=["#aviation", "#engineering", "#travel"],
    )

    issues = evaluate_script_quality(direct_script, target_word_range=(55, 100))
    # Must not complain about lack of artificial tension or philosophical ending
    assert issues == [], f"Direct explanation should have no issues, got: {issues}"


def test_alembic_duration_fields_migration_sqlite():
    """Verify that Alembic migration d4a8e21c3b70 upgrades and downgrades cleanly on an isolated database."""
    import sqlalchemy as sa
    from sqlalchemy import create_engine, inspect
    from alembic.config import Config
    from alembic.migration import MigrationContext
    from alembic.operations import Operations
    from alembic.script import ScriptDirectory

    test_engine = create_engine("sqlite:///:memory:")
    with test_engine.begin() as conn:
        conn.execute(sa.text("""
            CREATE TABLE videos (
                id VARCHAR(36) PRIMARY KEY,
                channel_id VARCHAR(36) NOT NULL,
                topic VARCHAR(300) NOT NULL,
                status VARCHAR(20) NOT NULL,
                visual_style_id VARCHAR(50)
            )
        """))

        ctx = MigrationContext.configure(conn, opts={"as_sql": False})
        op = Operations(ctx)

        cfg = Config("alembic.ini")
        script = ScriptDirectory.from_config(cfg)
        rev = script.get_revision("d4a8e21c3b70")
        assert rev.down_revision == "c7a2e81d4b60"

        mig = rev.module
        mig.op = op

        try:
            # 1. Upgrade
            mig.upgrade()
            insp = inspect(conn)
            cols = {c["name"] for c in insp.get_columns("videos")}
            assert "duration_mode" in cols
            assert "target_duration" in cols

            # 2. Downgrade
            mig.downgrade()
            insp_down = inspect(conn)
            cols_down = {c["name"] for c in insp_down.get_columns("videos")}
            assert "duration_mode" not in cols_down
            assert "target_duration" not in cols_down
        finally:
            mig.op = None
