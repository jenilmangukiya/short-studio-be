"""Smoke test script to verify real Kokoro voiceover generation on local hardware."""

from pathlib import Path
import sys
import time
import wave

# Add project root to sys.path so app modules import cleanly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.voiceover import get_voiceover_provider


def main():
    narration = "Welcome to Shorts Studio. Today, we are exploring something interesting."
    voice = "af_heart"
    output_path = Path("storage/kokoro-smoke-test.wav")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== Kokoro-82M Voiceover Smoke Test ===")
    print(f"Narration: \"{narration}\"")
    print(f"Voice:     {voice}\n")

    # Measure total wall-clock time including model loading and inference
    total_start = time.perf_counter()

    init_start = time.perf_counter()
    provider = get_voiceover_provider(provider="kokoro")
    init_time = time.perf_counter() - init_start

    print(f"Initialized provider '{provider.provider_name}' with model '{provider.model}'.")
    print("Generating speech (loading ONNX model & running local inference)...")

    result = provider.generate(text=narration, voice=voice)
    total_elapsed = time.perf_counter() - total_start

    # Save generated WAV to disk
    output_path.write_bytes(result.audio_bytes)

    # Validate resulting WAV header and frames
    if not output_path.is_file():
        raise RuntimeError(f"Output file was not written to {output_path}")

    file_size = output_path.stat().st_size
    if file_size <= 44:
        raise RuntimeError(f"Output file size ({file_size} bytes) is too small to contain audio")

    with wave.open(str(output_path), "rb") as wf:
        n_channels = wf.getnchannels()
        samp_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()

    if n_frames == 0:
        raise RuntimeError("WAV file contains zero audio frames")

    verified_duration = round(n_frames / float(framerate), 3)

    print("\n--- Smoke Test Results ---")
    print(f"Status:               SUCCESS")
    print(f"Output Path:          {output_path.resolve()}")
    print(f"File Size:            {file_size:,} bytes")
    print(f"Sample Rate:          {result.sample_rate} Hz (verified: {framerate} Hz)")
    print(f"Channels:             {n_channels} ({'Mono' if n_channels == 1 else 'Stereo'})")
    print(f"Bit Depth:            {samp_width * 8}-bit PCM")
    print(f"Total Audio Frames:   {n_frames:,}")
    print(f"Verified Duration:    {verified_duration:.3f} s")
    print(f"Reported Duration:    {result.duration_seconds:.3f} s")
    print(f"Inference Latency:    {result.latency_seconds:.3f} s")
    print(f"Total Elapsed Time:   {total_elapsed:.3f} s (including model loading)")
    print(f"Real-Time Factor:     {verified_duration / result.latency_seconds:.2f}x faster than real-time")
    print("\nThe generated WAV file is valid and ready to play.")


if __name__ == "__main__":
    main()
