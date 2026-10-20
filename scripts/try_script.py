#!/usr/bin/env python3
"""CLI utility to test and inspect the real script generation step without database interaction."""

import argparse
import json
import sys
import uuid
from pathlib import Path

# Ensure project root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from app.config import settings
from app.steps.script import generate_script_step


class SimpleNamespace:
    """Lightweight dummy object to simulate SQLAlchemy models without DB connection."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, name):
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Run Shorts Studio script step logic without touching the database."
    )
    parser.add_argument(
        "--topic",
        required=True,
        help="Video topic (e.g. 'Why airplane windows have tiny holes').",
    )
    parser.add_argument(
        "--description",
        help="Optional extra context or background for the video.",
    )
    parser.add_argument(
        "--channel-name",
        default="Hidden Facts About Everyday Things",
        help="Channel name (default: 'Hidden Facts About Everyday Things').",
    )
    parser.add_argument(
        "--channel-niche",
        default="Engineering and everyday mysteries",
        help="Channel niche.",
    )
    parser.add_argument(
        "--provider",
        choices=["mock", "gemini", "openai"],
        help="Override LLM provider for the script task.",
    )
    parser.add_argument(
        "--model",
        help="Optional model override when setting provider.",
    )
    parser.add_argument(
        "--revise",
        help="Revision instruction to test revision mode (requires --prev-script or previous run).",
    )
    parser.add_argument(
        "--prev-script",
        help="Optional path to a JSON file or inline JSON containing previous script output for revision mode.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output full raw result as JSON.",
    )

    args = parser.parse_args()

    # Apply task override if specified
    if args.provider:
        settings.llm_provider_script = args.provider
    if args.model:
        settings.llm_model_script = args.model

    # Construct dummy channel and video objects
    dummy_channel = SimpleNamespace(
        id=uuid.uuid4(),
        name=args.channel_name,
        niche=args.channel_niche,
        settings={"tone": "conversational", "pace": "fast"},
    )
    dummy_video = SimpleNamespace(
        id=uuid.uuid4(),
        topic=args.topic,
        description=args.description,
        channel=dummy_channel,
        channel_id=dummy_channel.id,
    )

    # Prepare previous script if revising
    prev_script_dict = None
    if args.prev_script:
        p = Path(args.prev_script)
        if p.exists():
            prev_script_dict = json.loads(p.read_text(encoding="utf-8"))
        else:
            prev_script_dict = json.loads(args.prev_script)
        # Unwrap nested shape if needed
        if isinstance(prev_script_dict, dict) and "script" in prev_script_dict:
            prev_script_dict = prev_script_dict["script"]

    step_output = {"script": prev_script_dict} if prev_script_dict else None
    dummy_step = SimpleNamespace(
        name="script",
        user_prompt=args.revise,
        output=step_output,
    )

    try:
        result = generate_script_step(
            video=dummy_video,
            prev={},
            step=dummy_step,
        )

        if args.json:
            print(json.dumps(result, indent=2))
            return

        script = result["script"]
        warnings = result.get("warnings", [])
        meta = result.get("meta", {})
        narration_words = len(script.get("narration", "").split())
        hook_words = len(script.get("hook", "").split())

        print("\n" + "=" * 65)
        print("Shorts Studio - Script Step Output")
        print("=" * 65)
        print(f"Title:         {script.get('title')}")
        print(f"Angle:         {script.get('angle')}")
        print(f"Hook:          {script.get('hook')} ({hook_words} words)")
        print("-" * 65)
        print("Narration:")
        print(script.get("narration"))
        print(f"Word Count:    {narration_words} words (target: 110-150 words)")
        print("-" * 65)
        print("Claims to Verify:")
        for idx, claim in enumerate(script.get("claims_to_verify", []), 1):
            print(f"  {idx}. {claim}")
        print(f"Hashtags:      {' '.join(script.get('hashtags', []))}")
        print("-" * 65)

        if warnings:
            print(f"Warnings ({len(warnings)}):")
            for w in warnings:
                print(f"  [!] {w}")
        else:
            print("Warnings:      None (All soft checks passed)")

        print("-" * 65)
        print("Execution Metadata:")
        print(f"  Provider:       {meta.get('provider')}")
        print(f"  Model:          {meta.get('model')}")
        print(f"  Prompt Version: {meta.get('prompt_version')}")
        print(f"  Latency:        {meta.get('latency_seconds')}s")
        print(f"  Attempts:       {meta.get('attempts')}")
        print(f"  Revision Mode:  {meta.get('revision')}")
        print(f"  Input Tokens:   {meta.get('input_tokens')}")
        print(f"  Output Tokens:  {meta.get('output_tokens')}")
        print(f"  Reasoning:      {meta.get('reasoning_tokens') if meta.get('reasoning_tokens') is not None else 'N/A'}")
        print("=" * 65 + "\n")

    except Exception as exc:
        print(f"\n[Error] Script step failed: {exc}\n", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
