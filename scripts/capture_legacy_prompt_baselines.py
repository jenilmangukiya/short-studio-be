"""Script to capture genuine baseline outputs from current unmodified prompt composition.

Executes compose_final_prompt across canonical test cases and saves exact outputs to
tests/fixtures/legacy_prompts_golden.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.steps.prompt_compose import compose_final_prompt

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
GOLDEN_PATH = FIXTURES_DIR / "legacy_prompts_golden.json"

LEGACY_STYLE_TEXT = (
    "Clean 3D render, soft studio lighting, smooth matte materials, "
    "simple pastel gradient background, shallow depth of field"
)


def capture_golden_baselines() -> dict[str, dict]:
    baselines: dict[str, dict] = {}

    # Case 1: Studio Close-up (no environment) -> retains pastel gradient and shallow DOF
    scene_1 = {
        "image_prompt": "A modern smartphone showing an incoming alert",
        "shot_type": "close-up",
    }
    prompt_1 = compose_final_prompt(scene=scene_1, visual_style=LEGACY_STYLE_TEXT)
    baselines["studio_closeup"] = {
        "scene": scene_1,
        "visual_style": LEGACY_STYLE_TEXT,
        "expected_prompt": prompt_1,
    }

    # Case 2: Wide shot in specific environment -> strips pastel gradient and shallow DOF
    scene_2 = {
        "image_prompt": "Rows of enterprise server racks with flashing blue LED indicators",
        "shot_type": "wide",
        "environment": "data center server hall",
    }
    prompt_2 = compose_final_prompt(scene=scene_2, visual_style=LEGACY_STYLE_TEXT)
    baselines["wide_environment"] = {
        "scene": scene_2,
        "visual_style": LEGACY_STYLE_TEXT,
        "expected_prompt": prompt_2,
    }

    # Case 3: Cutaway diagram -> diagram technique, strips shallow DOF
    scene_3 = {
        "image_prompt": "Cross-section cutaway showing internal fan blades",
        "shot_type": "cutaway",
        "visual_technique": "diagram",
    }
    prompt_3 = compose_final_prompt(scene=scene_3, visual_style=LEGACY_STYLE_TEXT)
    baselines["cutaway_diagram"] = {
        "scene": scene_3,
        "visual_style": LEGACY_STYLE_TEXT,
        "expected_prompt": prompt_3,
    }

    # Case 4: Aerial Vista -> aerial shot, strips shallow DOF and pastel
    scene_4 = {
        "image_prompt": "Aerial view of a sprawling metropolis with highway interchanges at night",
        "shot_type": "aerial",
        "environment": "urban highway landscape",
    }
    prompt_4 = compose_final_prompt(scene=scene_4, visual_style=LEGACY_STYLE_TEXT)
    baselines["aerial_cityscape"] = {
        "scene": scene_4,
        "visual_style": LEGACY_STYLE_TEXT,
        "expected_prompt": prompt_4,
    }

    # Case 5: Character with subject anchor
    scene_5 = {
        "image_prompt": "An airline captain inspecting the flight deck instrument panel",
        "shot_type": "medium",
        "uses_anchor": True,
    }
    subject_anchor_5 = "Captain Miller, a tall middle-aged pilot with aviator sunglasses and navy uniform"
    prompt_5 = compose_final_prompt(
        scene=scene_5,
        subject_anchor=subject_anchor_5,
        visual_style=LEGACY_STYLE_TEXT,
    )
    baselines["character_with_anchor"] = {
        "scene": scene_5,
        "subject_anchor": subject_anchor_5,
        "visual_style": LEGACY_STYLE_TEXT,
        "expected_prompt": prompt_5,
    }

    # Case 6: Connected Video Bible beat
    scene_6 = {
        "entity_bindings": {"character_id": "c1", "environment_id": "e1"},
        "camera": {"shot_type": "medium", "camera_angle": "eye_level", "focal_point_description": "the pilot"},
        "visual_description": "Checking flight instruments",
    }
    vb_6 = {
        "art_direction": {
            "style_family": "cinematic",
            "visual_genre": "styled_3d",
            "rendering_texture": "smooth matte geometry",
            "lighting_philosophy": "soft directional key light",
            "negative_anchors": ["no 2D sketches", "no photorealism"],
        },
        "entities": {
            "characters": [{"id": "c1", "name": "Pilot", "visual_anchor": "Stylized 3D pilot", "apparel_anchor": "navy uniform"}],
            "environments": [{"id": "e1", "name": "Cockpit", "spatial_layout": "Instrument panel controls"}],
        },
    }
    prompt_6 = compose_final_prompt(scene=scene_6, video_bible=vb_6)
    baselines["connected_video_bible_beat"] = {
        "scene": scene_6,
        "video_bible": vb_6,
        "expected_prompt": prompt_6,
    }

    # Case 7: Standalone conceptual diagram beat (sanitizes bedroom/sleeper references)
    scene_7 = {
        "entity_bindings": {"concept_id": "cp1"},
        "camera": {"shot_type": "medium", "focal_point_description": "signal waveform"},
        "visual_description": "Brain wave frequencies shifting from alpha to theta",
    }
    vb_7 = {
        "art_direction": {
            "style_family": "cinematic",
            "visual_genre": "styled_3d",
            "rendering_texture": "smooth matte geometry, cotton weave bedding",
            "negative_anchors": ["keep character in bedroom"],
        },
        "entities": {
            "concepts": [{
                "id": "cp1",
                "name": "Sleep Onset",
                "concept_explained": "As the sleeper drifts off and muscles relax, brainwaves slow down",
                "visual_analogy": "The sleeper gently settles into the bed while soundwaves ripple into rest",
            }],
        },
    }
    prompt_7 = compose_final_prompt(scene=scene_7, video_bible=vb_7)
    baselines["conceptual_diagram_beat"] = {
        "scene": scene_7,
        "video_bible": vb_7,
        "expected_prompt": prompt_7,
    }

    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump(baselines, f, indent=2)

    print(f"Captured {len(baselines)} golden baselines to {GOLDEN_PATH}")
    return baselines


if __name__ == "__main__":
    capture_golden_baselines()
