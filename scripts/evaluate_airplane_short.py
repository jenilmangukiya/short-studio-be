"""Generate fresh storyboard and image test set for airplane window evaluation."""

import json
import time
import uuid
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from app.config import settings
from app.images import get_image_provider
from app.steps.prompt_compose import compose_final_prompt
from app.steps.storyboard import generate_storyboard_step

EVAL_DIR = Path("eval_airplane_window")
EVAL_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR = EVAL_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

SCRIPT_DATA = {
    "title": "Why airplane windows have a tiny hole",
    "hook": "That tiny hole in an airplane window is doing more than you think.",
    "angle": "why it's designed this way",
    "narration": (
        "That tiny hole in an airplane window is doing more than you think. "
        "A typical passenger window isn't one thick sheet; it uses several clear plastic panes stacked in a frame. "
        "The cabin-side pane is a scratch cover, and the tiny hole in it lets cabin air pass into the space behind it. "
        "As the plane climbs, outside pressure falls, but cabin pressure is kept higher for passengers. "
        "Air moving through the hole helps the space between panes stay close to cabin pressure. "
        "That reduces the pressure load on the inner cover and leaves the outer pane to handle most of the difference between cabin and outside air. "
        "The hole doesn't lead straight outdoors: the intact outer pane still seals the cabin. "
        "So that little opening isn't damage or a leak; it's part of how the layered window manages pressure."
    ),
    "claims_to_verify": [
        "Typical passenger airplane windows use several clear plastic panes stacked in a frame.",
        "The cabin-side window pane acts as a scratch cover and has a small hole that lets cabin air pass behind it.",
        "As an airplane climbs, outside air pressure falls while cabin pressure is kept higher.",
        "The hole helps the space between window panes stay close to cabin pressure.",
        "The outer pane handles most of the pressure difference and seals the cabin from outside air.",
    ],
}


class DummyChannel:
    name = "Curious Engineering"
    niche = "Aviation Engineering"
    settings = {
        "visual_style": "Clean cinematic photography, realistic aviation lighting, crisp details"
    }


class DummyVideo:
    id = uuid.uuid4()
    topic = "Why is there a tiny hole in an airplane window?"
    channel = DummyChannel()


class DummyStep:
    name = "storyboard"
    user_prompt = None
    output = None


def step1_generate_storyboard():
    print("--- 1. Generating Fresh Storyboard via current LLM ---")
    prev = {"script": {"script": SCRIPT_DATA}}
    result = generate_storyboard_step(DummyVideo(), prev, DummyStep())

    sb_path = EVAL_DIR / "storyboard.json"
    sb_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Storyboard saved to: {sb_path}")
    print(f"Warnings: {result.get('warnings', [])}")
    print(f"Attempts: {result.get('meta', {}).get('attempts')}")
    print(f"Prompt version: {result.get('meta', {}).get('prompt_version')}")

    scenes = result["storyboard"]["scenes"]
    print(f"\nGenerated {len(scenes)} scenes:")
    for s in scenes:
        print(f"\n[Scene {s['index']}] ({s['est_seconds']}s) | Shot: {s['shot_type']} | Anchor: {s['uses_anchor']}")
        print(f"  Visual Purpose: {s.get('visual_purpose')}")
        print(f"  Angle/POV:      {s.get('camera_angle')}")
        print(f"  Focal Point:    {s.get('focal_point')}")
        print(f"  Technique:      {s.get('visual_technique')}")
        print(f"  Narration:      \"{s['narration']}\"")
        print(f"  Image Prompt:   \"{s['image_prompt']}\"")
        print(f"  Final Prompt:   \"{s['final_prompt']}\"")

    return result


if __name__ == "__main__":
    step1_generate_storyboard()
