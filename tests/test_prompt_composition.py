"""Tests for Phase 2: Image prompt composition, context integration, and adaptive visual style."""

import re
from app.steps.prompt_compose import (
    COMPOSITION_RULES,
    MAX_FINAL_PROMPT_LENGTH,
    adapt_visual_style,
    compose_final_prompt,
    recompute_final_prompts,
)


def test_all_metadata_fields_incorporated():
    """Verify that shot_type, camera_angle, focal_point, visual_technique, environment,

    and visual_purpose are not dropped when composing the final prompt.
    """
    scene = {
        "image_prompt": "An advanced silicon processor chip mounted on a circuit board.",
        "uses_anchor": False,
        "shot_type": "close-up",
        "camera_angle": "45-degree angle",
        "focal_point": "intricate copper heat pipes and microscopic circuit traces",
        "visual_technique": "macro-photo",
        "visual_purpose": "Explain high-performance computing hardware requirements",
        "narration": "Because the answer takes powerful computer chips,",
        "environment": "inside an open enterprise server blade chassis",
    }
    style = "Clean 3D render, smooth matte materials"

    prompt = compose_final_prompt(scene, visual_style=style)

    # 1. Subject description present
    assert "An advanced silicon processor chip mounted on a circuit board." in prompt

    # 2. Environment present and not dropped
    assert "Environment: inside an open enterprise server blade chassis." in prompt

    # 3. Visual purpose / concept present
    assert "Visual concept: Explain high-performance computing hardware requirements." in prompt

    # 4. Camera direction with framing and angle present
    assert "Camera direction: close-up shot, 45-degree angle." in prompt

    # 5. Primary focal point present
    assert "Primary focal point: intricate copper heat pipes and microscopic circuit traces." in prompt

    # 6. Visual technique present
    assert "Visual technique: macro-photo." in prompt

    # 7. Style present
    assert "Clean 3D render, smooth matte materials." in prompt

    # 8. Safe vertical framing rules present
    assert COMPOSITION_RULES in prompt

    # 9. Length within budget
    assert len(prompt) <= MAX_FINAL_PROMPT_LENGTH


def test_adaptive_visual_style_removes_pastel_for_real_environment():
    """Verify that generic pastel gradient background is stripped when a real environment is specified."""
    default_style = (
        "Clean 3D render, soft studio lighting, smooth matte materials, "
        "simple pastel gradient background, shallow depth of field"
    )

    # Data center scene
    adapted = adapt_visual_style(
        visual_style=default_style,
        environment="modern data center server hall with blue LED racks",
        shot_type="wide",
        visual_technique="cinematic-photo",
    )

    assert "pastel gradient background" not in adapted
    assert "shallow depth of field" not in adapted
    assert "Clean 3D render" in adapted
    assert "smooth matte materials" in adapted


def test_adaptive_visual_style_preserves_studio_for_abstract_scene():
    """Verify that studio lighting and pastel backdrop are preserved for abstract standalone objects."""
    default_style = (
        "Clean 3D render, soft studio lighting, smooth matte materials, "
        "simple pastel gradient background, shallow depth of field"
    )

    adapted = adapt_visual_style(
        visual_style=default_style,
        environment="",
        shot_type="close-up",
        visual_technique="cinematic-photo",
    )

    assert "simple pastel gradient background" in adapted
    assert "shallow depth of field" in adapted
    assert "Clean 3D render" in adapted


def test_environment_anchor_used_when_scene_environment_is_empty():
    """Verify that environment_anchor provides fallback background context."""
    scene = {
        "image_prompt": "Rows of glowing server blades.",
        "uses_anchor": False,
        "shot_type": "medium",
        "environment": "",  # Empty on scene
    }
    env_anchor = "High-density enterprise data center with cool blue ambient glow."
    style = "Clean 3D render, smooth matte materials"

    prompt = compose_final_prompt(
        scene,
        visual_style=style,
        environment_anchor=env_anchor,
    )

    assert "Environment setting: High-density enterprise data center with cool blue ambient glow." in prompt


def test_recompute_final_prompts_integrates_environment_anchor():
    """Verify recompute_final_prompts updates scenes in-place with environment_anchor."""
    storyboard = {
        "storyboard": {
            "visual_style": "Clean 3D render",
            "subject_anchor": "A glowing quantum processor.",
            "environment_anchor": "Futuristic subterranean laboratory facility.",
            "scenes": [
                {
                    "index": 1,
                    "image_prompt": "Processor resting on testing pedestal.",
                    "uses_anchor": True,
                    "shot_type": "medium",
                    "camera_angle": "eye-level",
                    "focal_point": "quantum processor chip",
                    "visual_technique": "cinematic-photo",
                    "environment": "",
                },
                {
                    "index": 2,
                    "image_prompt": "Overview of lab control consoles.",
                    "uses_anchor": False,
                    "shot_type": "wide",
                    "camera_angle": "high-angle",
                    "focal_point": "monitoring screens",
                    "visual_technique": "cinematic-photo",
                    "environment": "laboratory control room",
                },
            ],
        }
    }

    updated = recompute_final_prompts(storyboard)
    scenes = updated["storyboard"]["scenes"]

    # Scene 1 uses environment_anchor
    assert "Environment setting: Futuristic subterranean laboratory facility." in scenes[0]["final_prompt"]
    assert "A glowing quantum processor." in scenes[0]["final_prompt"]

    # Scene 2 uses explicit environment
    assert "Environment: laboratory control room." in scenes[1]["final_prompt"]
    assert "A glowing quantum processor." not in scenes[1]["final_prompt"]


def test_three_scenes_from_ai_cost_storyboard():
    """Demonstrate actual final prompts produced for three distinct scenes from the AI question cost project."""
    style = "Clean 3D render, soft studio lighting, smooth matte materials, simple pastel gradient background, shallow depth of field"
    env_anchor = "Modern enterprise data center infrastructure with dark metallic surfaces and cool LED status glow."
    subj_anchor = "High-performance enterprise AI accelerator processor with exposed silicon die and copper heat spreader."

    # Scene 1: Hook - AI Query Interface
    scene_1 = {
        "index": 1,
        "narration": "Why does answering one AI question cost so much?",
        "visual_purpose": "Establish curiosity gap with luminous query interface in dark computing space",
        "shot_type": "medium",
        "camera_angle": "slightly low angle",
        "focal_point": "luminous digital prompt query box",
        "visual_technique": "cinematic-photo",
        "environment": "dark modern computing environment with soft blue status illumination",
        "image_prompt": "A glowing holographic query interface floating in a sleek minimalist tech space with subtle computational light pulses",
        "uses_anchor": False,
    }

    prompt_1 = compose_final_prompt(
        scene_1,
        subject_anchor=subj_anchor,
        visual_style=style,
        environment_anchor=env_anchor,
        video_topic="Why AI Questions Cost So Much",
    )

    # Scene 2: Payoff - Powerful Computer Chips
    scene_2 = {
        "index": 2,
        "narration": "Because the answer takes powerful computer chips,",
        "visual_purpose": "Reveal dedicated silicon processor hardware executing massive matrix calculations",
        "shot_type": "close-up",
        "camera_angle": "45-degree angle",
        "focal_point": "intricate copper heat spreader and silicon core",
        "visual_technique": "macro-photo",
        "environment": "open server blade chassis with dense circuit board traces",
        "image_prompt": "An advanced silicon processor chip mounted on a multi-layer server motherboard with copper heat pipes",
        "uses_anchor": True,
    }

    prompt_2 = compose_final_prompt(
        scene_2,
        subject_anchor=subj_anchor,
        visual_style=style,
        environment_anchor=env_anchor,
        video_topic="Why AI Questions Cost So Much",
    )

    # Scene 4: Infrastructure Scale - Data Centers
    scene_4 = {
        "index": 4,
        "narration": "That’s only the start. Companies also pay to build data centers,",
        "visual_purpose": "Show vast physical scale and investment of enterprise data center halls",
        "shot_type": "wide",
        "camera_angle": "low-angle wide perspective",
        "focal_point": "towering server cabinets receding into infinity",
        "visual_technique": "cinematic-photo",
        "environment": "vast enterprise data center facility with endless server rack aisles and overhead cable trays",
        "image_prompt": "Wide architectural view of an expansive data center hall with rows of illuminated server racks and fiber optic cable trays",
        "uses_anchor": False,
    }

    prompt_4 = compose_final_prompt(
        scene_4,
        subject_anchor=subj_anchor,
        visual_style=style,
        environment_anchor=env_anchor,
        video_topic="Why AI Questions Cost So Much",
    )

    # Verifications for Scene 1
    assert "pastel gradient background" not in prompt_1
    assert "Environment: dark modern computing environment with soft blue status illumination." in prompt_1
    assert "Visual concept: Establish curiosity gap with luminous query interface in dark computing space." in prompt_1
    assert "Camera direction: medium shot, slightly low angle." in prompt_1
    assert "Primary focal point: luminous digital prompt query box." in prompt_1
    assert subj_anchor not in prompt_1  # uses_anchor is False

    # Verifications for Scene 2
    assert "pastel gradient background" not in prompt_2
    assert subj_anchor in prompt_2  # uses_anchor is True
    assert "Environment: open server blade chassis with dense circuit board traces." in prompt_2
    assert "Camera direction: close-up shot, 45-degree angle." in prompt_2
    assert "Visual technique: macro-photo." in prompt_2

    # Verifications for Scene 4
    assert "pastel gradient background" not in prompt_4
    assert "shallow depth of field" not in prompt_4  # Wide shot must have deep focus
    assert "Environment: vast enterprise data center facility with endless server rack aisles and overhead cable trays." in prompt_4
    assert "Camera direction: wide shot, low-angle wide perspective." in prompt_4
    assert "Primary focal point: towering server cabinets receding into infinity." in prompt_4
    assert COMPOSITION_RULES in prompt_4
