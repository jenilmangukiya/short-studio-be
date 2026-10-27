"""Generate test images for the fresh airplane-window storyboard and build a contact sheet."""

import json
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

from app.config import settings
from app.images import get_image_provider
from app.steps.prompt_compose import compose_final_prompt

EVAL_DIR = Path("eval_airplane_window")
IMAGES_DIR = EVAL_DIR / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
CONTACT_SHEET_PATH = EVAL_DIR / "contact_sheet.png"


def load_storyboard():
    sb_file = EVAL_DIR / "storyboard.json"
    if not sb_file.exists():
        raise FileNotFoundError(f"Storyboard file not found at {sb_file}")
    return json.loads(sb_file.read_text(encoding="utf-8"))


def generate_images():
    data = load_storyboard()
    sb = data["storyboard"]
    scenes = sb["scenes"]
    subject_anchor = sb.get("subject_anchor")
    visual_style = sb.get("visual_style")

    provider = get_image_provider()
    size = settings.openai_image_size or "1024x1536"
    quality = settings.openai_image_quality or "low"

    print(f"=== Starting Evaluation Image Generation ===")
    print(f"Provider: {provider.provider_name} | Model: {provider.model}")
    print(f"Size: {size} | Quality: {quality}")
    print(f"Total scenes: {len(scenes)}")
    print(f"Saving to: {IMAGES_DIR.resolve()}\n")

    generated_files = []

    for s in scenes:
        idx = s["index"]
        out_img_path = IMAGES_DIR / f"scene_{idx:02d}.png"
        out_txt_path = IMAGES_DIR / f"scene_{idx:02d}_prompt.txt"

        final_prompt = compose_final_prompt(
            scene=s,
            subject_anchor=subject_anchor,
            visual_style=visual_style,
        )
        out_txt_path.write_text(final_prompt, encoding="utf-8")

        if out_img_path.exists() and out_img_path.stat().st_size > 1000:
            print(f"[Scene {idx:02d}/{len(scenes):02d}] Already exists: {out_img_path.name}")
            generated_files.append((s, out_img_path))
            continue

        print(f"\n[Scene {idx:02d}/{len(scenes):02d}] Generating image...")
        print(f"  Shot:      {s.get('shot_type')} | Technique: {s.get('visual_technique')}")
        print(f"  Purpose:   {s.get('visual_purpose')}")
        print(f"  Prompt:    {final_prompt[:120]}...")

        t0 = time.perf_counter()
        result = provider.generate(prompt=final_prompt, size=size, quality=quality)
        latency = time.perf_counter() - t0

        out_img_path.write_bytes(result.image_bytes)
        print(f"  Saved:     {out_img_path.name} ({len(result.image_bytes):,} bytes, {latency:.2f}s)")
        generated_files.append((s, out_img_path))

    return generated_files


def create_contact_sheet(generated_files):
    print("\n=== Generating Contact Sheet ===")
    total = len(generated_files)
    if total == 0:
        print("No images to assemble.")
        return

    # Layout: 4 columns
    cols = 4
    rows = (total + cols - 1) // cols

    thumb_w = 340
    thumb_h = 510  # 2:3 aspect ratio
    card_padding = 16
    caption_h = 130
    cell_w = thumb_w + card_padding * 2
    cell_h = thumb_h + caption_h + card_padding * 2

    header_h = 100
    canvas_w = cell_w * cols + 40
    canvas_h = header_h + cell_h * rows + 40

    sheet = Image.new("RGB", (canvas_w, canvas_h), color=(15, 17, 23))
    draw = ImageDraw.Draw(sheet)

    # Fonts (fallback to default)
    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        font_sub = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_badge = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 15)
        font_body = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except Exception:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_badge = ImageFont.load_default()
        font_body = ImageFont.load_default()

    # Draw Header
    draw.text((30, 24), "Shorts Studio - Visual Storytelling Evaluation", fill=(255, 255, 255), font=font_title)
    draw.text((30, 64), "Topic: \"Why is there a tiny hole in an airplane window?\" | Model: gpt-image-2 | Fresh Storyboard Beats", fill=(160, 170, 190), font=font_sub)

    for i, (scene, img_path) in enumerate(generated_files):
        c = i % cols
        r = i // cols

        x = 20 + c * cell_w
        y = header_h + r * cell_h

        # Card background
        card_rect = [x + 6, y + 6, x + cell_w - 6, y + cell_h - 6]
        draw.rounded_rectangle(card_rect, radius=12, fill=(26, 30, 42), outline=(45, 52, 72), width=1)

        # Thumbnail
        try:
            with Image.open(img_path) as im:
                thumb = im.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                sheet.paste(thumb, (x + card_padding, y + card_padding))
        except Exception as exc:
            print(f"Error loading image {img_path}: {exc}")
            draw.rectangle([x + card_padding, y + card_padding, x + card_padding + thumb_w, y + card_padding + thumb_h], fill=(50, 50, 50))

        # Caption text
        idx = scene["index"]
        shot = scene.get("shot_type", "shot")
        technique = scene.get("visual_technique", "photo")
        purpose = scene.get("visual_purpose", "")
        narration = scene.get("narration", "")

        tx = x + card_padding
        ty = y + card_padding + thumb_h + 10

        # Badge: Scene Number & Shot Type
        badge_text = f"Scene {idx}  |  {shot.upper()}  |  {technique}"
        draw.rounded_rectangle([tx, ty, tx + thumb_w, ty + 24], radius=4, fill=(40, 50, 75))
        draw.text((tx + 8, ty + 4), badge_text, fill=(100, 200, 255), font=font_badge)

        # Purpose line
        ty += 30
        draw.text((tx, ty), f"Purpose: {purpose[:45]}", fill=(240, 240, 240), font=font_body)

        # Narration snippet
        ty += 20
        draw.text((tx, ty), f"Narration: \"{narration[:45]}...\"", fill=(160, 170, 185), font=font_body)

        # Focal Point
        ty += 20
        fp = scene.get("focal_point", "")
        draw.text((tx, ty), f"Focus: {fp[:45]}", fill=(200, 180, 120), font=font_body)

    sheet.save(CONTACT_SHEET_PATH, quality=95)
    print(f"\n[SUCCESS] Contact sheet generated and saved to: {CONTACT_SHEET_PATH.resolve()}")


def main():
    generated_files = generate_images()
    create_contact_sheet(generated_files)


if __name__ == "__main__":
    main()
