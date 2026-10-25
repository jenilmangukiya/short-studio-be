"""Download Kokoro-82M ONNX model weights and voicepack assets.

This script fetches the official Kokoro ONNX model files and places them
into the designated storage models directory.

Usage:
    python scripts/download_kokoro_model.py
"""

import sys
import urllib.request
from pathlib import Path

# URLs for Kokoro v1.0 ONNX model files
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

DEFAULT_TARGET_DIR = Path(__file__).resolve().parent.parent / "storage" / "models" / "kokoro"


def _download_file(url: str, dest_path: Path):
    if dest_path.is_file() and dest_path.stat().st_size > 0:
        print(f"File already exists: {dest_path} ({dest_path.stat().st_size / (1024 * 1024):.1f} MB). Skipping.")
        return

    print(f"Downloading {url} to {dest_path}...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(".tmp")

    def _progress(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            downloaded_mb = (count * block_size) / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  Progress: {percent}% ({downloaded_mb:.1f}/{total_mb:.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, temp_path, reporthook=_progress)
        temp_path.rename(dest_path)
        print(f"\nSaved: {dest_path} ({dest_path.stat().st_size / (1024 * 1024):.1f} MB)")
    except Exception as exc:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {exc}") from exc


def main():
    target_dir = DEFAULT_TARGET_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Kokoro-82M Asset Downloader ===")
    print(f"Target Directory: {target_dir}\n")

    model_dest = target_dir / "kokoro-v1.0.onnx"
    voices_dest = target_dir / "voices-v1.0.bin"

    _download_file(MODEL_URL, model_dest)
    _download_file(VOICES_URL, voices_dest)

    print("\nKokoro-82M model assets are ready for local voiceover generation.")


if __name__ == "__main__":
    main()
