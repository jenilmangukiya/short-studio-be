"""Standalone CLI script to test image generation without a database.

Supports generating an image from a direct prompt or extracting and composing
a scene prompt from an exported storyboard JSON file.

Usage:
    # Single prompt with mock provider (default)
    python scripts/try_image.py --prompt "A sleek retro robot walking down neon street"

    # From a storyboard JSON file
    python scripts/try_image.py --storyboard-file path/to/sb.json --scene 1

    # Using OpenAI (makes exactly ONE real API call)
    python scripts/try_image.py --prompt "A vintage camera" --provider openai
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

# Add project root to sys.path so app modules import cleanly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.images import get_image_provider
from app.steps.prompt_compose import compose_final_prompt


def parse_args():
    parser = argparse.ArgumentParser(description="Shorts Studio - Standalone Image Generation Tester")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--prompt", type=str, help="Direct prompt text to generate an image for")
    group.add_argument("--storyboard-file", type=str, help="Path to JSON file containing storyboard output")

    parser.add_argument("--scene", type=int, default=1, help="Scene index (1-based) to pick from storyboard-file")
    parser.add_argument("--provider", type=str, choices=["mock", "openai"], default=None, help="Image provider (default: from settings or mock)")
    parser.add_argument("--size", type=str, default=None, help="Image size (e.g. 1024x1536, 864x1536, or 1024x1024)")
    parser.add_argument("--quality", type=str, default=None, help="Image quality (e.g. low, medium, high)")
    parser.add_argument("--out-dir", type=str, default="./tmp_images", help="Output directory to save generated PNG (default: ./tmp_images)")

    return parser.parse_args()


def load_scene_prompt_from_storyboard(filepath: str, scene_index: int) -> tuple[str, int]:
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Storyboard file not found: {filepath}")

    data = json.loads(path.read_text(encoding="utf-8"))
    sb = data.get("storyboard") if isinstance(data, dict) and "storyboard" in data else data
    if isinstance(sb, dict) and isinstance(sb.get("storyboard"), dict):
        sb = sb.get("storyboard")

    scenes = sb.get("scenes") if isinstance(sb, dict) else None
    if not scenes or not isinstance(scenes, list):
        raise ValueError("Could not find scenes array in storyboard JSON")

    target_scene = next((s for s in scenes if s.get("index") == scene_index), None)
    if not target_scene:
        raise ValueError(f"Scene with index {scene_index} not found in storyboard")

    visual_style = sb.get("visual_style")
    subject_anchor = sb.get("subject_anchor")

    final_prompt = compose_final_prompt(
        scene=target_scene,
        subject_anchor=subject_anchor,
        visual_style=visual_style,
    )
    return final_prompt, scene_index


def main():
    args = parse_args()

    scene_idx = 1
    if args.prompt:
        prompt_text = args.prompt.strip()
    else:
        prompt_text, scene_idx = load_scene_prompt_from_storyboard(args.storyboard_file, args.scene)

    provider_name = args.provider or settings.image_provider or "mock"
    size = args.size or settings.openai_image_size or "1024x1536"
    quality = args.quality or settings.openai_image_quality or "low"

    provider = get_image_provider(provider=provider_name)

    if provider.provider_name == "openai":
        print(f"\n[NOTICE] About to make exactly 1 real API call to OpenAI Images API with model: '{provider.model}'.\n")

    full_hash = hashlib.sha256(f"{prompt_text}|{provider.model}|{size}|{quality}".encode("utf-8")).hexdigest()
    hash8 = full_hash[:8]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_filename = f"scene_{scene_idx:02d}_{hash8}.png"
    out_path = out_dir / out_filename

    print(f"--- Generating Image ---")
    print(f"Provider:    {provider.provider_name}")
    print(f"Model:       {provider.model}")
    print(f"Size:        {size}")
    print(f"Quality:     {quality}")
    print(f"Prompt:      {prompt_text}")
    print(f"Prompt Hash: {full_hash}")

    start_time = time.perf_counter()
    result = provider.generate(prompt=prompt_text, size=size, quality=quality)
    elapsed = time.perf_counter() - start_time

    out_path.write_bytes(result.image_bytes)

    print("\n--- Result ---")
    print(f"Saved To:     {out_path.resolve()} ({len(result.image_bytes):,} bytes)")
    print(f"Latency:      {result.latency_seconds:.3f}s (wall time: {elapsed:.3f}s)")
    if result.est_cost_usd is not None:
        print(f"Est Cost:     ${result.est_cost_usd:.4f}")
    else:
        print(f"Est Cost:     null (no cost configured)")
    if result.input_tokens is not None or result.output_tokens is not None:
        print(f"Tokens:       in={result.input_tokens}, out={result.output_tokens}, total={result.total_tokens}")
    print("Done.")


if __name__ == "__main__":
    main()
