"""Phase 4A Tests — Storyboard & Prompt Flow Integration for Visual Style System.

Verifies:
1. Authoritative snapshot resolution (snapshot -> style_id -> legacy fallback).
2. Storyboard LLM prompt building respects active preset rules and anchors.
3. Prompt composition integration with entity adapters (stickman, stylized 2D, legacy Clean 3D).
4. Semantic storyboard data preservation (narration, sequence, timing, actions, bindings).
5. Deterministic prompt recomposition via recompute_final_prompts (zero LLM calls).
6. Preserved 100% legacy parity for NULL-style and legacy_default videos.
"""

from __future__ import annotations

import copy
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from app.config import settings
from app.schemas import (
    CameraDirection,
    CharacterState,
    EntityBindings,
    StoryboardBeat,
)
from app.steps.prompt_compose import (
    COMPOSITION_RULES,
    compose_final_prompt,
    recompute_final_prompts,
)
from app.steps.storyboard import (
    build_storyboard_prompt,
    generate_storyboard_step,
    load_default_visual_style,
    load_prompt_template,
)
from app.styles.adapter import (
    adapt_character,
    adapt_concept,
    adapt_environment,
    adapt_object,
)
from app.styles.registry import VisualStyleResolutionError, get_style, list_styles
from app.styles.schemas import VisualStylePreset


# ==============================================================================
# Helpers & Fixtures
# ==============================================================================

@pytest.fixture(autouse=True)
def mock_llm_settings(monkeypatch):
    monkeypatch.setattr(settings, "llm_provider", "mock")
    monkeypatch.setattr(settings, "llm_provider_storyboard", "mock")


def _make_dummy_video(
    visual_style_id: str | None = None,
    visual_style_snapshot: dict[str, Any] | None = None,
    visual_style_version: int | None = -1,
    channel_default_style_id: str | None = None,
    channel_settings: dict[str, Any] | None = None,
) -> Any:
    resolved_version = (1 if visual_style_id else None) if visual_style_version == -1 else visual_style_version

    class DummyChannel:
        def __init__(self):
            self.id = "chan_001"
            self.name = "Science Explained"
            self.niche = "Science & Tech"
            self.default_visual_style_id = channel_default_style_id
            self.settings = channel_settings or {}

    class DummyVideo:
        def __init__(self):
            self.id = "vid_001"
            self.topic = "Why We Sleep"
            self.title = "The Science of Sleep"
            self.visual_style_id = visual_style_id
            self.visual_style_snapshot = visual_style_snapshot
            self.visual_style_version = resolved_version
            self.channel = DummyChannel()
            self.duration_mode = "manual"
            self.target_duration = 60

    return DummyVideo()


def _make_sample_script_data() -> dict[str, Any]:
    return {
        "title": "Why We Sleep",
        "hook": "Ever wondered why your body forces you to sleep every night?",
        "angle": "Biological exploration of sleep architecture",
        "narration": (
            "Ever wondered why your body forces you to sleep every night? "
            "Deep inside your brain, neurons clear out toxic waste. "
            "Meanwhile, memories are consolidated into long-term storage."
        ),
    }


def _make_sample_entities_and_bible() -> dict[str, Any]:
    return {
        "entities": {
            "characters": [
                {
                    "id": "char_sleeper",
                    "name": "Dr. Alex",
                    "role": "Neuroscientist",
                    "visual_anchor": "slender researcher with round spectacles and silver hair",
                    "apparel_anchor": "crisp white laboratory coat over dark teal sweater",
                    "forbidden_mutations": ["no cartoon animal ears", "no neon hair"],
                }
            ],
            "environments": [
                {
                    "id": "env_lab",
                    "name": "Sleep Research Lab",
                    "spatial_layout": "clean clinical chamber with EEG monitors and dim observation window",
                    "lighting_setup": "subdued cool blue ambient lighting",
                    "time_of_day": "night",
                    "persistent_props": ["EEG workstation", "polygraph monitor"],
                }
            ],
            "objects_and_props": [
                {
                    "id": "obj_eeg_cap",
                    "name": "EEG Sensor Headset",
                    "visual_details": "mesh cap with gold-plated electrodes and braided ribbon cables",
                    "material_and_finish": "matte black polymer with polished brass terminals",
                }
            ],
            "concepts_and_diagrams": [
                {
                    "id": "cpt_glymphatic",
                    "name": "Glymphatic Clearance Flow",
                    "concept_explained": "cerebrospinal fluid washing metabolic waste between glial cells",
                    "visual_analogy": "pulsing micro-currents through narrow cellular pathways",
                    "graphical_elements": ["directional flow arrows", "waste particle icons"],
                }
            ],
        },
        "art_direction": {
            "style_family": "cinematic",
            "visual_genre": "photorealistic_cinematic",
            "rendering_texture": "smooth matte surfaces, subtle micro-contrast",
            "lighting_philosophy": "subdued high-contrast laboratory illumination",
            "primary_palette_hex": ["#0F172A", "#1E293B", "#38BDF8"],
            "negative_anchors": ["no oversaturated neon", "no muddy grain"],
        },
    }


# ==============================================================================
# 1. Authoritative Snapshot Resolution Tests
# ==============================================================================

def _make_dummy_step():
    class DummyStep:
        name = "storyboard"
        user_prompt = None
        output = None
    return DummyStep()


def test_storyboard_resolves_authoritative_snapshot():
    """Video.visual_style_snapshot is authoritative over registry and channel."""
    stickman_preset = get_style("stickman_tactical_explainer")
    custom_snapshot = copy.deepcopy(stickman_preset.model_dump())
    custom_snapshot["display_name"] = "Custom Stickman Explainer V2"

    video = _make_dummy_video(
        visual_style_id="stylized_2d_narrative",  # Registry mismatch on purpose
        visual_style_snapshot=custom_snapshot,  # Snapshot must win
        channel_default_style_id="legacy_default",
    )

    prev = {"script": {"script": _make_sample_script_data()}}
    result = generate_storyboard_step(video, prev, _make_dummy_step())

    sb = result["storyboard"]
    assert sb["style_source"] == "preset"
    assert sb["visual_style_id"] == "stickman_tactical_explainer"
    assert sb["visual_style_snapshot"]["display_name"] == "Custom Stickman Explainer V2"
    assert "Custom Stickman Explainer V2" in sb["visual_style"]


def test_storyboard_fallback_to_video_style_id_safe_version():
    """When snapshot is absent, falls back to video.visual_style_id with matching version."""
    video = _make_dummy_video(
        visual_style_id="stylized_2d_narrative",
        visual_style_version=1,
        visual_style_snapshot=None,
        channel_default_style_id="legacy_default",
    )

    prev = {"script": {"script": _make_sample_script_data()}}
    result = generate_storyboard_step(video, prev, _make_dummy_step())

    sb = result["storyboard"]
    assert sb["style_source"] == "preset"
    assert sb["visual_style_id"] == "stylized_2d_narrative"
    assert sb["visual_style_snapshot"] is not None
    assert "Stylized 2D Narrative" in sb["visual_style"]


def test_legacy_video_null_fields_channel_default_stickman_preserves_legacy():
    """Existing legacy video with NULL snapshot and NULL style ID must NOT inherit channel Stickman default."""
    video = _make_dummy_video(
        visual_style_id=None,
        visual_style_snapshot=None,
        visual_style_version=None,
        channel_default_style_id="stickman_tactical_explainer",
    )

    prev = {"script": {"script": _make_sample_script_data()}}
    result = generate_storyboard_step(video, prev, _make_dummy_step())

    sb = result["storyboard"]
    assert sb["style_source"] == "default"
    assert sb["visual_style"] == load_default_visual_style()
    assert sb.get("visual_style_id") is None
    assert sb.get("visual_style_snapshot") is None
    assert "Stickman" not in sb["visual_style"]
    # Verify generated prompts use legacy Clean 3D path
    for sc in sb["scenes"]:
        assert "Clean 3D render" in sc["final_prompt"]
        assert "stickman" not in sc["final_prompt"].lower()


def test_legacy_video_null_fields_channel_default_stylized_2d_preserves_legacy():
    """Existing legacy video with NULL style fields must NOT inherit channel Stylized 2D default."""
    video = _make_dummy_video(
        visual_style_id=None,
        visual_style_snapshot=None,
        visual_style_version=None,
        channel_default_style_id="stylized_2d_narrative",
    )

    prev = {"script": {"script": _make_sample_script_data()}}
    result = generate_storyboard_step(video, prev, _make_dummy_step())

    sb = result["storyboard"]
    assert sb["style_source"] == "default"
    assert sb["visual_style"] == load_default_visual_style()
    assert sb.get("visual_style_id") is None
    assert sb.get("visual_style_snapshot") is None
    assert "Stylized 2D" not in sb["visual_style"]
    for sc in sb["scenes"]:
        assert "Clean 3D render" in sc["final_prompt"]
        assert "stylized 2d" not in sc["final_prompt"].lower()


def test_storyboard_incompatible_historical_version_raises_error():
    """When video has an incompatible preset version without snapshot, fail-closed raises VisualStyleResolutionError."""
    video = _make_dummy_video(
        visual_style_id="stickman_tactical_explainer",
        visual_style_version=99,  # Incompatible historical version (registry has version 1)
        visual_style_snapshot=None,
        channel_default_style_id="stickman_tactical_explainer",
    )

    prev = {"script": {"script": _make_sample_script_data()}}
    with pytest.raises(VisualStyleResolutionError) as exc_info:
        generate_storyboard_step(video, prev, _make_dummy_step())
    assert "Incompatible visual style version 99" in str(exc_info.value)


def test_storyboard_unknown_style_id_raises_error():
    """When video has an unknown style ID without snapshot, fail-closed raises VisualStyleResolutionError."""
    video = _make_dummy_video(
        visual_style_id="nonexistent_style_preset_xyz",
        visual_style_version=1,
        visual_style_snapshot=None,
    )

    prev = {"script": {"script": _make_sample_script_data()}}
    with pytest.raises(VisualStyleResolutionError) as exc_info:
        generate_storyboard_step(video, prev, _make_dummy_step())
    assert "Unknown visual style ID 'nonexistent_style_preset_xyz'" in str(exc_info.value)


def test_storyboard_malformed_snapshot_raises_error():
    """When video has a malformed or invalid snapshot, fail-closed raises VisualStyleResolutionError."""
    video = _make_dummy_video(
        visual_style_id="stickman_tactical_explainer",
        visual_style_snapshot={"corrupted": "missing_required_fields"},
    )

    prev = {"script": {"script": _make_sample_script_data()}}
    with pytest.raises(VisualStyleResolutionError) as exc_info:
        generate_storyboard_step(video, prev, _make_dummy_step())
    assert "Malformed or invalid visual style snapshot" in str(exc_info.value)


def test_storyboard_legacy_null_behavior_preserved():
    """When all style fields are None, preserves existing Clean 3D studio default."""
    video = _make_dummy_video(
        visual_style_id=None,
        visual_style_snapshot=None,
        channel_default_style_id=None,
    )

    prev = {"script": {"script": _make_sample_script_data()}}
    result = generate_storyboard_step(video, prev, _make_dummy_step())

    sb = result["storyboard"]
    assert sb["style_source"] == "default"
    assert sb["visual_style"] == load_default_visual_style()
    assert sb.get("visual_style_id") is None
    assert sb.get("visual_style_snapshot") is None


# ==============================================================================
# 2. Storyboard LLM Prompt Building with Active Presets
# ==============================================================================

def test_build_storyboard_prompt_includes_active_preset_rules():
    """Prompt builder injects active preset name, anchors, and drawing rules."""
    rules_text, _ = load_prompt_template()
    script = _make_sample_script_data()
    stickman = get_style("stickman_tactical_explainer")
    anchors_str = ", ".join(stickman.permanent_prompt_anchors)
    rules_str = (
        f"Linework: {stickman.rules.linework_and_outlines}. "
        f"Shading: {stickman.rules.color_and_shading}."
    )
    style_str = f"{stickman.display_name}: {anchors_str}. {rules_str}"

    prompt = build_storyboard_prompt(
        rules_text=rules_text,
        topic="Sleep Science",
        script_data=script,
        visual_style=style_str,
    )

    assert "Target Visual Style" in prompt
    assert "Stickman Tactical Explainer" in prompt
    assert "clean 2D animated stickman illustration style" in prompt


# ==============================================================================
# 3. Prompt Composition & Entity Adapter Integration Tests
# ==============================================================================

def test_compose_final_prompt_stickman_preset_with_video_bible():
    """Stickman preset applies adapt_character, adapt_object, and stickman anchors."""
    preset = get_style("stickman_tactical_explainer")
    bible = _make_sample_entities_and_bible()

    scene = {
        "visual_description": "Dr. Alex inspecting the EEG cap workstation.",
        "entity_bindings": {
            "character_id": "char_sleeper",
            "environment_id": "env_lab",
            "object_ids": ["obj_eeg_cap"],
        },
        "camera": {
            "shot_type": "medium_close_up",
            "camera_angle": "low_angle",
            "focal_point_description": "the glowing electrode cap",
        },
        "character_state": {
            "pose_description": "pointing finger at the screen",
            "expression_description": "focused neutral",
            "action_in_progress": "analyzing real-time waveforms",
        },
    }

    prompt = compose_final_prompt(
        scene=scene,
        video_bible=bible,
        preset=preset,
    )

    # 1. Preset permanent anchors present
    for anchor in preset.permanent_prompt_anchors:
        assert anchor in prompt

    # 2. Lighting rule injected
    assert f"Lighting: {preset.rules.lighting_style}" in prompt

    # 3. Character adapted via adapt_character
    assert "2D stickman illustration style: Dr. Alex (Neuroscientist)" in prompt
    assert "clean uniform black outlines, simplified stick figure anatomy" in prompt
    assert "pose: pointing finger at the screen" in prompt

    # 4. Object adapted via adapt_object
    assert "tactical 2D line-art prop: EEG Sensor Headset" in prompt

    # 5. Camera framing preserved
    assert "medium close up shot" in prompt
    assert "low angle" in prompt
    assert "Primary focal point: the glowing electrode cap" in prompt

    # 6. Negative constraints injected
    assert "Avoid:" in prompt
    for neg in preset.negative_prompt_anchors:
        assert neg in prompt
    assert "no cartoon animal ears" in prompt  # character forbidden mutation

    # 7. Standard 9:16 vertical composition rules
    assert COMPOSITION_RULES in prompt


def test_compose_final_prompt_stylized_2d_narrative_with_video_bible():
    """Stylized 2D preset applies adapt_character, adapt_environment, and 2D narrative anchors."""
    preset = get_style("stylized_2d_narrative")
    bible = _make_sample_entities_and_bible()

    scene = {
        "visual_description": "Dr. Alex sitting by the observation window at night.",
        "entity_bindings": {
            "character_id": "char_sleeper",
            "environment_id": "env_lab",
        },
        "camera": {
            "shot_type": "wide",
            "camera_angle": "eye_level",
            "focal_point_description": "silhouette against observation window",
        },
    }

    prompt = compose_final_prompt(
        scene=scene,
        video_bible=bible,
        preset=preset,
    )

    # 1. Preset permanent anchors present
    for anchor in preset.permanent_prompt_anchors:
        assert anchor in prompt

    # 2. Character adapted via adapt_character for 2D narrative
    assert "stylized 2D graphic illustration style: Dr. Alex (Neuroscientist)" in prompt
    assert "expressive ink linework, 2-tone cel-shading" in prompt

    # 3. Environment adapted via adapt_environment
    assert "stylized 2D narrative background art: Sleep Research Lab" in prompt

    # 4. Negative anchors injected
    assert "Avoid:" in prompt
    for neg in preset.negative_prompt_anchors:
        assert neg in prompt


def test_compose_final_prompt_concept_diagram_adaptation():
    """Concept beat adapts concept diagrams via adapt_concept according to active preset."""
    preset_stickman = get_style("stickman_tactical_explainer")
    preset_2d = get_style("stylized_2d_narrative")
    bible = _make_sample_entities_and_bible()

    concept_scene = {
        "visual_description": "Schematic view of cerebrospinal fluid channels.",
        "entity_bindings": {
            "concept_id": "cpt_glymphatic",
        },
        "camera": {
            "shot_type": "diagram",
            "focal_point_description": "intercellular pathways",
        },
    }

    prompt_stickman = compose_final_prompt(
        scene=concept_scene,
        video_bible=bible,
        preset=preset_stickman,
    )
    assert "Tactical schematic visual diagram: Glymphatic Clearance Flow" in prompt_stickman
    assert "Simplified cross-section with clean vector arrows" in prompt_stickman

    prompt_2d = compose_final_prompt(
        scene=concept_scene,
        video_bible=bible,
        preset=preset_2d,
    )
    assert "Illustrated conceptual diagram: Glymphatic Clearance Flow" in prompt_2d
    assert "Narrative visual metaphors with clean cel-shaded graphical flows" in prompt_2d


def test_compose_final_prompt_legacy_default_parity():
    """When preset is legacy_default or None, produces identical Clean 3D studio output."""
    bible = _make_sample_entities_and_bible()
    scene = {
        "visual_description": "Dr. Alex looking through notes in the laboratory.",
        "entity_bindings": {
            "character_id": "char_sleeper",
            "environment_id": "env_lab",
        },
        "camera": {
            "shot_type": "medium",
            "camera_angle": "eye-level",
            "focal_point": "Dr. Alex",
        },
    }

    # Calling with preset=None
    prompt_none = compose_final_prompt(
        scene=scene,
        video_bible=bible,
        preset=None,
    )

    # Calling with preset=get_style("legacy_default")
    legacy_preset = get_style("legacy_default")
    prompt_legacy = compose_final_prompt(
        scene=scene,
        video_bible=bible,
        preset=legacy_preset,
    )

    # Both must match character-for-character
    assert prompt_none == prompt_legacy
    assert "Photorealistic cinematic visual style" in prompt_none
    assert "Subject: Dr. Alex, slender researcher" in prompt_none
    assert "Environment setting: Sleep Research Lab" in prompt_none


# ==============================================================================
# 4. Semantic Storyboard Data Preservation & Deterministic Recomposition
# ==============================================================================

def test_recompute_final_prompts_preserves_storyboard_semantics():
    """Recomputing prompts when changing visual style preserves 100% of semantic data."""
    stickman = get_style("stickman_tactical_explainer")
    narrative_2d = get_style("stylized_2d_narrative")
    bible = _make_sample_entities_and_bible()

    # Initial storyboard under Stickman preset
    initial_scenes = [
        {
            "index": 1,
            "narration": "Deep inside your brain, neurons clear out toxic waste.",
            "est_seconds": 3.5,
            "shot_type": "medium_close_up",
            "camera_angle": "low_angle",
            "focal_point": "the brain scan display",
            "visual_purpose": "Explain neuronal waste clearance during deep sleep",
            "visual_description": "Dr. Alex pointing at the glowing neuro-imaging monitor.",
            "entity_bindings": {"character_id": "char_sleeper", "environment_id": "env_lab"},
            "character_state": {"pose_description": "standing tall", "action_in_progress": "pointing"},
            "visual_beats": [
                {
                    "id": "beat_1_1",
                    "order": 1,
                    "narration_text": "Deep inside your brain...",
                    "visual_description": "Dr. Alex pointing at the glowing neuro-imaging monitor.",
                    "entity_bindings": {"character_id": "char_sleeper"},
                    "final_prompt": "old prompt",
                }
            ],
            "final_prompt": "old prompt",
        }
    ]

    storyboard = {
        "visual_style_id": stickman.id,
        "visual_style_snapshot": stickman.model_dump(),
        "visual_style": stickman.display_name,
        "scenes": copy.deepcopy(initial_scenes),
        "video_bible": bible,
    }

    # Recompute to Stylized 2D Narrative
    recomputed = recompute_final_prompts(
        storyboard=storyboard,
        preset=narrative_2d,
    )

    # Verify style metadata was updated
    assert recomputed["visual_style_id"] == "stylized_2d_narrative"
    assert recomputed["visual_style_snapshot"]["id"] == "stylized_2d_narrative"
    assert "Stylized 2D Narrative" in recomputed["visual_style"]

    # Verify prompts were recompiled with 2D narrative anchors
    new_scene_prompt = recomputed["scenes"][0]["final_prompt"]
    assert "stylized 2D graphic illustration style" in new_scene_prompt
    assert "flat 2D vector stickman" not in new_scene_prompt

    new_beat_prompt = recomputed["scenes"][0]["visual_beats"][0]["final_prompt"]
    assert "stylized 2D graphic illustration style" in new_beat_prompt

    # VERIFY SEMANTIC DATA IS 100% UNCHANGED:
    sc = recomputed["scenes"][0]
    orig_sc = initial_scenes[0]
    assert sc["narration"] == orig_sc["narration"]
    assert sc["est_seconds"] == orig_sc["est_seconds"]
    assert sc["shot_type"] == orig_sc["shot_type"]
    assert sc["camera_angle"] == orig_sc["camera_angle"]
    assert sc["focal_point"] == orig_sc["focal_point"]
    assert sc["visual_purpose"] == orig_sc["visual_purpose"]
    assert sc["visual_description"] == orig_sc["visual_description"]
    assert sc["entity_bindings"] == orig_sc["entity_bindings"]
    assert sc["character_state"] == orig_sc["character_state"]
    assert len(sc["visual_beats"]) == len(orig_sc["visual_beats"])
    assert sc["visual_beats"][0]["order"] == orig_sc["visual_beats"][0]["order"]
    assert sc["visual_beats"][0]["narration_text"] == orig_sc["visual_beats"][0]["narration_text"]


def test_recompute_final_prompts_zero_llm_calls():
    """recompute_final_prompts is purely deterministic code and makes zero LLM calls."""
    stickman = get_style("stickman_tactical_explainer")
    bible = _make_sample_entities_and_bible()

    storyboard = {
        "visual_style_id": "stickman_tactical_explainer",
        "visual_style_snapshot": stickman.model_dump(),
        "scenes": [
            {
                "index": 1,
                "narration": "Neurons clear toxic waste.",
                "visual_description": "Dr. Alex in the lab.",
                "entity_bindings": {"character_id": "char_sleeper"},
                "final_prompt": "",
            }
        ],
        "video_bible": bible,
    }

    # Patch get_llm to ensure it is never called
    with patch("app.llm.get_llm") as mock_get_llm:
        res = recompute_final_prompts(storyboard=storyboard, preset=stickman)
        mock_get_llm.assert_not_called()
        assert "clean 2D animated stickman illustration style" in res["scenes"][0]["final_prompt"]


def test_recompute_final_prompts_legacy_null_storyboard_preserves_clean3d():
    """Recomputing a legacy storyboard with NULL style fields preserves legacy Clean 3D prompt path."""
    bible = _make_sample_entities_and_bible()
    legacy_storyboard = {
        "visual_style": load_default_visual_style(),
        "visual_style_id": None,
        "visual_style_snapshot": None,
        "scenes": [
            {
                "index": 1,
                "narration": "Neurons clear toxic waste during deep sleep.",
                "visual_description": "Dr. Alex inspecting the laboratory workstation.",
                "entity_bindings": {"character_id": "char_sleeper", "environment_id": "env_lab"},
                "camera": {"shot_type": "medium", "camera_angle": "eye-level"},
                "final_prompt": "",
            }
        ],
        "video_bible": bible,
    }

    recomputed = recompute_final_prompts(storyboard=legacy_storyboard)
    assert recomputed.get("visual_style_id") is None
    assert recomputed.get("visual_style_snapshot") is None
    sc_prompt = recomputed["scenes"][0]["final_prompt"]
    assert "Photorealistic cinematic visual style" in sc_prompt
    assert "Subject: Dr. Alex" in sc_prompt
    assert "Environment setting: Sleep Research Lab" in sc_prompt
    assert "stickman" not in sc_prompt.lower()
    assert "stylized 2d" not in sc_prompt.lower()


def test_resolve_preset_unit_scenarios():
    """Test _resolve_preset version-awareness, snapshot authority, and fail-closed errors."""
    from app.steps.prompt_compose import _resolve_preset

    stickman = get_style("stickman_tactical_explainer")

    # 1. Authoritative snapshot wins over mismatched ID
    snap_context = {
        "visual_style_snapshot": stickman.model_dump(),
        "visual_style_id": "stylized_2d_narrative",
    }
    resolved = _resolve_preset(context=snap_context)
    assert resolved is not None
    assert resolved.id == "stickman_tactical_explainer"

    # 2. Matching version resolves successfully
    matching_context = {
        "visual_style_id": "stickman_tactical_explainer",
        "visual_style_version": 1,
    }
    resolved_matching = _resolve_preset(context=matching_context)
    assert resolved_matching is not None
    assert resolved_matching.id == "stickman_tactical_explainer"

    # 3. Incompatible version raises VisualStyleResolutionError (fail-closed)
    incompat_context = {
        "visual_style_id": "stickman_tactical_explainer",
        "visual_style_version": 99,
    }
    with pytest.raises(VisualStyleResolutionError):
        _resolve_preset(context=incompat_context)

    # 4. Unknown style ID raises VisualStyleResolutionError (fail-closed)
    unknown_context = {
        "visual_style_id": "unknown_future_preset_xyz",
    }
    with pytest.raises(VisualStyleResolutionError):
        _resolve_preset(context=unknown_context)

    # 5. Malformed snapshot raises VisualStyleResolutionError (fail-closed)
    malformed_context = {
        "visual_style_snapshot": {"broken": "data"},
    }
    with pytest.raises(VisualStyleResolutionError):
        _resolve_preset(context=malformed_context)

    # 6. NULL style fields return None (historical Clean 3D path)
    null_context = {
        "visual_style_id": None,
        "visual_style_snapshot": None,
        "visual_style_version": None,
    }
    resolved_null = _resolve_preset(context=null_context)
    assert resolved_null is None


def test_compose_final_prompt_malformed_snapshot_raises_error():
    """compose_final_prompt fail-closed raises VisualStyleResolutionError when passed a malformed snapshot."""
    scene = {
        "visual_description": "A character walking.",
        "visual_style_snapshot": {"invalid": True},
    }
    with pytest.raises(VisualStyleResolutionError):
        compose_final_prompt(scene=scene)
