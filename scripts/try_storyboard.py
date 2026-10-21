#!/usr/bin/env python3
"""CLI utility to test and inspect the real storyboard generation step without database interaction."""

import argparse
import json
import sys
import uuid
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from app.config import settings
from app.steps.storyboard import generate_storyboard_step


class SimpleNamespace:
    """Lightweight dummy object to simulate SQLAlchemy models without DB connection."""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, name):
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run Shorts Studio storyboard step logic without touching the database."
    )
    parser.add_argument(
        "--script-file",
        help="Path to a text file with script narration, or a JSON file with script output.",
    )
    parser.add_argument(
        "--script",
        help="Inline script narration text or inline JSON script output.",
    )
    parser.add_argument(
        "--visual-style",
        help="Optional visual style override.",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "gemini", "openai"],
        help="Override LLM provider for the storyboard task.",
    )
    parser.add_argument(
        "--model",
        help="Optional model override when setting provider.",
    )
    parser.add_argument(
        "--revise",
        help="Revision instruction to test revision mode (requires previous storyboard).",
    )
    parser.add_argument(
        "--prev-storyboard",
        help="Path to a JSON file or inline JSON containing previous storyboard output for revision mode.",
    )
    parser.add_argument(
        "--topic",
        default="Why airplane windows have tiny holes",
        help="Video topic (default: 'Why airplane windows have tiny holes').",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full raw result as JSON.",
    )

    args = parser.parse_args()

    # Apply task override if specified
    if args.provider:
        settings.llm_provider_storyboard = args.provider
    if args.model:
        settings.llm_model_storyboard = args.model

    # Determine script content
    script_data = None
    if args.script_file:
        p = Path(args.script_file)
        if not p.exists():
            print(f"Error: Script file not found: {args.script_file}", file=sys.stderr)
            sys.exit(1)
        raw = p.read_text(encoding="utf-8").strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "script" in parsed:
                script_data = parsed["script"] if isinstance(parsed["script"], dict) else parsed
            else:
                script_data = parsed
        except json.JSONDecodeError:
            script_data = {"narration": raw, "title": args.topic, "hook": raw.split(".")[0] + "."}
    elif args.script:
        raw = args.script.strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "script" in parsed:
                script_data = parsed["script"] if isinstance(parsed["script"], dict) else parsed
            else:
                script_data = parsed
        except json.JSONDecodeError:
            script_data = {"narration": raw, "title": args.topic, "hook": raw.split(".")[0] + "."}
    else:
        # Default sample narration for quick test
        script_data = {
            "title": "Why Airplane Windows Have Tiny Holes",
            "hook": "Have a look at airplane windows.",
            "angle": "why it's designed this way",
            "narration": (
                "Have a look at airplane windows. Air travelers often notice a tiny hole in the passenger window. "
                "This small bleed hole balances the immense pressure difference between the pressurized cabin and the thin outside atmosphere. "
                "Aircraft windows consist of three separate acrylic panes. The outer pane bears the primary pressure load, "
                "while the inner pane prevents passengers from scratching the crucial middle layer. The bleed hole allows "
                "moisture to escape, preventing frost from obstructing the view. Without this simple engineering detail, "
                "windows could fail or fog completely. Aircraft designers solved a massive cabin pressure challenge with "
                "a miniature opening. Every commercial jet flying at high altitude relies on these tiny ports for passenger "
                "safety and clear visibility throughout the entire flight."
            ),
        }

    # Dummy channel settings
    channel_settings = {}
    if args.visual_style:
        channel_settings["visual_style"] = args.visual_style

    dummy_channel = SimpleNamespace(
        id=uuid.uuid4(),
        name="Hidden Facts About Everyday Things",
        niche="Engineering and everyday mysteries",
        settings=channel_settings,
    )
    dummy_video = SimpleNamespace(
        id=uuid.uuid4(),
        topic=args.topic,
        channel=dummy_channel,
        channel_id=dummy_channel.id,
    )

    # Prepare previous storyboard if revising
    prev_sb_dict = None
    if args.prev_storyboard:
        p = Path(args.prev_storyboard)
        if p.exists():
            parsed = json.loads(p.read_text(encoding="utf-8"))
        else:
            parsed = json.loads(args.prev_storyboard)
        if isinstance(parsed, dict) and "storyboard" in parsed:
            prev_sb_dict = parsed["storyboard"]
        else:
            prev_sb_dict = parsed

    dummy_step = SimpleNamespace(
        name="storyboard",
        user_prompt=args.revise,
        output={"storyboard": prev_sb_dict} if prev_sb_dict else None,
    )

    prev_inputs = {
        "script": {"script": script_data}
    }

    try:
        result = generate_storyboard_step(
            video=dummy_video,
            prev=prev_inputs,
            step=dummy_step,
        )

        if args.json:
            print(json.dumps(result, indent=2))
            return

        sb = result["storyboard"]
        scenes = sb.get("scenes", [])
        warnings = result.get("warnings", [])
        meta = result.get("meta", {})

        print("\n" + "=" * 60)
        print("STORYBOARD GENERATION RESULT")
        print("=" * 60)
        print(f"Visual Style ({sb.get('style_source', 'default')}): {sb.get('visual_style')}")
        print(f"Total Scenes: {len(scenes)} | Total Duration: {sb.get('total_seconds')}s\n")

        print("SCENES:")
        for s in scenes:
            print(f"  [{s['index']}] ({s['est_seconds']}s | {s['shot_type']})")
            print(f"      Image Prompt: {s['image_prompt']}")
            print(f"      Narration:    {s['narration']}")
            print()

        if warnings:
            print("-" * 60)
            print("WARNINGS:")
            for w in warnings:
                print(f"  * {w}")

        print("-" * 60)
        print("METADATA:")
        for k, v in meta.items():
            print(f"  {k}: {v}")
        print("=" * 60 + "\n")

    except Exception as exc:
        print(f"\nExecution failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
