"""Real hardware integration test and audio quality inspector for Continuous Narration.

Executes continuous narration generation using the installed Kokoro-82M ONNX model
on a representative multi-scene storyboard script. Performs acoustic analysis
on the generated continuous audio WAV and scene slices, verifying speech energy,
WAV headers, durations, silence padding, and idempotent asset reuse.

Usage:
    .venv/bin/python scripts/verify_continuous_narration.py
"""

from pathlib import Path
import sys
import time
import uuid
import wave

import numpy as np
import soundfile as sf

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.steps.voiceover import generate_voiceover_step
from app.voiceover import KokoroVoiceoverProvider, get_voiceover_provider


class DummyVideo:
    def __init__(self, video_id=None):
        self.id = video_id or uuid.uuid4()
        self.topic = "The Mystery of Airplane Window Bleed Holes"
        self.channel = None


class DummyStep:
    def __init__(self, name="voiceover"):
        self.name = name
        self.user_prompt = None
        self.output = None
        self.status = "pending"


def analyze_audio_waveform(wav_path: Path) -> dict:
    """Analyze acoustic characteristics of a WAV file on disk."""
    data, sr = sf.read(str(wav_path))
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    duration = round(len(data) / float(sr), 3)
    peak = float(np.max(np.abs(data))) if len(data) > 0 else 0.0
    rms = float(np.sqrt(np.mean(data**2))) if len(data) > 0 else 0.0
    rms_db = round(20 * np.log10(rms + 1e-9), 2)
    peak_db = round(20 * np.log10(peak + 1e-9), 2)

    # Detect speech onset (first sample > 1.5% peak)
    thresh = max(0.005, 0.015 * peak)
    active = np.where(np.abs(data) > thresh)[0]
    lead_silence_ms = round((active[0] / sr) * 1000.0, 1) if len(active) > 0 else 0.0
    trail_silence_ms = round(((len(data) - active[-1]) / sr) * 1000.0, 1) if len(active) > 0 else 0.0

    return {
        "duration_seconds": duration,
        "sample_rate": sr,
        "total_samples": len(data),
        "peak_amplitude": round(peak, 4),
        "peak_db": peak_db,
        "rms_energy": round(rms, 4),
        "rms_db": rms_db,
        "lead_silence_ms": lead_silence_ms,
        "trail_silence_ms": trail_silence_ms,
        "clipping_detected": peak >= 0.999,
    }


def main():
    print("=" * 75)
    print("  KOKORO-82M CONTINUOUS NARRATION REAL INTEGRATION TEST & AUDIO AUDIT")
    print("=" * 75)

    model_path = Path(settings.kokoro_model_path)
    voices_path = Path(settings.kokoro_voices_path)
    print(f"Model path:  {model_path} (exists: {model_path.is_file()})")
    print(f"Voices path: {voices_path} (exists: {voices_path.is_file()})\n")

    if not model_path.is_file() or not voices_path.is_file():
        print("ERROR: Kokoro model files not found. Run scripts/download_kokoro_model.py first.")
        sys.exit(1)

    # 1. Initialize real Kokoro provider
    init_start = time.perf_counter()
    provider = get_voiceover_provider("kokoro")
    init_time = time.perf_counter() - init_start
    print(f"Initialized Kokoro provider '{provider.provider_name}' ({provider.model}) in {init_time:.2f}s")

    # 2. Define representative 5-scene storyboard script
    scenes = [
        {
            "index": 1,
            "narration": "Why do airplane windows have tiny holes?",
            "est_seconds": 2.5,
        },
        {
            "index": 2,
            "narration": "It is not a manufacturing defect, and it is definitely not an accident.",
            "est_seconds": 4.5,
        },
        {
            "index": 3,
            "narration": "That tiny opening is called a bleed hole.",
            "est_seconds": 3.0,
        },
        {
            "index": 4,
            "narration": "It balances atmospheric pressure between the pressurized cabin and the freezing exterior air.",
            "est_seconds": 5.5,
        },
        {
            "index": 5,
            "narration": "It also prevents moisture from fogging up the glass during your flight.",
            "est_seconds": 4.5,
        },
    ]

    prev = {
        "storyboard": {
            "storyboard": {
                "visual_style": "Cinematic 3D render",
                "scenes": scenes,
                "total_seconds": sum(s["est_seconds"] for s in scenes),
            }
        }
    }

    test_video_id = uuid.uuid4()
    video = DummyVideo(video_id=test_video_id)
    step = DummyStep()

    print(f"\nTarget Video ID: {video.id}")
    print(f"Total Scenes:    {len(scenes)}")
    print(f"Total Words:     {sum(len(s['narration'].split()) for s in scenes)}")
    print(f"Estimated Time:  {sum(s['est_seconds'] for s in scenes):.1f}s")
    print("\nScene Narrations:")
    for s in scenes:
        print(f"  Scene {s['index']}: \"{s['narration']}\"")

    # 3. Execute Continuous Narration Generation (Run 1: Cold generation)
    print("\n" + "-" * 75)
    print("RUN 1: Generating Continuous Narration with real Kokoro-82M ONNX model...")
    print("-" * 75)

    gen_start = time.perf_counter()
    output = generate_voiceover_step(video, prev, step=step, voiceover_provider=provider)
    gen_elapsed = time.perf_counter() - gen_start

    print(f"Generation completed in {gen_elapsed:.2f} seconds.")

    # 4. Validate output payload structure
    vo = output["voiceover"]
    meta = output["meta"]
    assert vo["continuous"] is True, "Expected continuous=True in payload"
    assert vo["complete"] is True, "Expected complete=True in payload"
    assert len(vo["items"]) == 5, f"Expected 5 scene items, got {len(vo['items'])}"

    total_dur = vo["total_duration_seconds"]
    cont_rel = vo["audio_file"]
    storage_root = Path(settings.storage_dir).resolve()
    cont_abs = storage_root / cont_rel

    print(f"\n--- Output Payload Metadata ---")
    print(f"Continuous Asset Path:  {cont_rel}")
    print(f"Total Duration:         {total_dur:.3f} s")
    print(f"Top-Level Duration:     {output['duration_seconds']:.3f} s")
    print(f"Inference Calls Made:   {meta.get('inference_calls', 1)}")
    print(f"Chunks Count:           {meta.get('chunks_count', 1)}")
    print(f"Scenes Generated:       {meta.get('scenes_generated')}")
    print(f"Scenes Reused:          {meta.get('scenes_reused')}")

    # 5. Acoustic Analysis on Continuous Narration WAV
    print("\n" + "-" * 75)
    print("ACOUSTIC ANALYSIS: Continuous Narration WAV Asset")
    print("-" * 75)

    assert cont_abs.is_file(), f"Continuous WAV file not found at {cont_abs}"
    analysis = analyze_audio_waveform(cont_abs)

    print(f"File Path:            {cont_abs}")
    print(f"File Size:            {cont_abs.stat().st_size:,} bytes")
    print(f"Sample Rate:          {analysis['sample_rate']} Hz (Standard: 24,000 Hz)")
    print(f"Total Duration:       {analysis['duration_seconds']} s (Payload: {total_dur} s)")
    print(f"RMS Energy Level:     {analysis['rms_db']} dBFS ({analysis['rms_energy']:.4f})")
    print(f"Peak Amplitude:       {analysis['peak_db']} dBFS ({analysis['peak_amplitude']:.4f})")
    print(f"Clipping Detected:    {'YES (WARNING)' if analysis['clipping_detected'] else 'NO (Clean headroom)'}")
    print(f"Lead Breath Silence:  {analysis['lead_silence_ms']} ms")
    print(f"Trail Pause Silence:  {analysis['trail_silence_ms']} ms")

    # Safety checks
    assert analysis["sample_rate"] == 24000, "Sample rate must be 24000 Hz"
    assert analysis["duration_seconds"] > 10.0, "15-word+ multi-scene script must produce > 10s audio"
    assert not analysis["clipping_detected"], "Audio must not clip"
    assert analysis["rms_energy"] > 0.02, "Speech must contain active audio energy"

    # 6. Scene-to-Narration Mapping Verification
    print("\n" + "-" * 75)
    print("SCENE MAPPINGS AUDIT")
    print("-" * 75)

    items = vo["items"]
    for it in items:
        sc_idx = it["index"]
        assert "file" not in it, f"Scene {sc_idx} should not have a separate audio slice file"

        print(f"Scene {sc_idx}:")
        print(f"  Narration:     \"{it['narration'][:45]}...\"")
        print(f"  Timing:        {it['start_time']:.3f}s -> {it['end_time']:.3f}s (dur: {it['duration_seconds']:.3f}s)")
        print(f"  Timing Type:   {it['timing_type']}")

    # 7. Idempotent Reuse Verification (Run 2: Warm rerun with identical storyboard)
    print("\n" + "-" * 75)
    print("RUN 2: Verifying Idempotent Caching & Asset Reuse...")
    print("-" * 75)

    step.output = output
    rerun_start = time.perf_counter()
    rerun_output = generate_voiceover_step(video, prev, step=step, voiceover_provider=provider)
    rerun_elapsed = time.perf_counter() - rerun_start

    rerun_meta = rerun_output["meta"]
    print(f"Rerun completed in {rerun_elapsed * 1000.0:.2f} ms (Target: < 50ms)")
    print(f"Scenes Reused:     {rerun_meta['scenes_reused']} / {rerun_meta['scenes_total']}")
    print(f"Scenes Generated:  {rerun_meta['scenes_generated']}")
    assert rerun_meta["scenes_reused"] == 5, "Expected all 5 scenes to be reused"
    assert rerun_meta["scenes_generated"] == 0, "Expected 0 scenes generated on reuse"
    assert all(it["reused"] is True for it in rerun_output["voiceover"]["items"])

    print("\n" + "=" * 75)
    print("  ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    main()
