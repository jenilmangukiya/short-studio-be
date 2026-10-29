"""Verification script for Continuous Narration Video Rendering Integration.

Performs:
1. End-to-end rendering in an isolated temporary test directory.
2. Uses continuous narration WAV audio with AI-directed camera animation across 3 scenes.
3. Probes the rendered video using ffprobe to verify video dimensions, frame rate, codecs,
   and exact audio/video duration alignment.
4. Executes the subtitles step to verify subtitles burning on top of the continuous-audio video.
5. Verifies audio stream parameters and that no drift or truncation occurred.

Usage:
    .venv/bin/python scripts/verify_continuous_rendering.py
"""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

import numpy as np
import soundfile as sf

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings
from app.images.mock import create_solid_png
from app.models import Video, VideoStep
from app.pipeline import h_render, h_subtitles
from app.steps.render import generate_render_step
from app.voiceover.mock import create_mock_wav


class DummyVideo:
    def __init__(self, video_id=None):
        self.id = video_id or uuid.uuid4()
        self.topic = "The Secrets of Airplane Bleed Holes"
        self.output_path = None
        self.channel = None


def run_ffprobe(file_path: Path) -> dict:
    """Run ffprobe on an MP4 file and return parsed JSON."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=codec_name,codec_type,width,height,r_frame_rate,channels,sample_rate,pix_fmt,duration,nb_frames",
        "-show_entries", "format=duration,size,bit_rate",
        "-of", "json",
        str(file_path),
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
    return json.loads(res.stdout)


def main():
    print("=" * 70)
    print("Shorts Studio — Continuous Narration Video Rendering Verification")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir).resolve()
        settings.storage_dir = str(tmp_path)
        settings.render_fps = 30

        video = DummyVideo()
        video_dir = tmp_path / str(video.id)
        video_dir.mkdir(parents=True, exist_ok=True)
        audio_dir = video_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[1] Preparing test assets in isolated path: {video_dir}")

        # 3 scenes with distinct vertical compositions
        scenes_def = [
            {"index": 1, "shot_type": "wide", "narration": "Have you ever noticed that tiny hole in airplane windows?"},
            {"index": 2, "shot_type": "close-up", "narration": "It is called a bleed hole, and it balances the extreme air pressure."},
            {"index": 3, "shot_type": "medium", "narration": "Without it, the inner pane could crack at cruising altitude."},
        ]

        img_items = []
        for sc in scenes_def:
            idx = sc["index"]
            img_rel = f"{video.id}/scene_{idx:02d}.png"
            img_abs = tmp_path / img_rel
            color = (30 * idx, 50 + 20 * idx, 100 + 30 * idx)
            img_abs.write_bytes(create_solid_png(1024, 1536, *color))
            img_items.append({"index": idx, "status": "ok", "file": img_rel})

        # Generate realistic continuous audio WAV: 4.8 seconds total
        total_audio_duration = 4.8
        sample_rate = 24000
        cont_audio_rel = f"{video.id}/audio/narration_4a8b1c2d.wav"
        cont_audio_abs = tmp_path / cont_audio_rel
        cont_audio_abs.write_bytes(create_mock_wav(total_audio_duration, sample_rate))

        # Scene timing metadata for continuous narration
        # Scene 1: 1.5s, Scene 2: 1.8s, Scene 3: 1.5s (Sum: 4.8s)
        slice_durs = [1.5, 1.8, 1.5]
        vo_items = []
        cur_t = 0.0
        for idx, (sc, d) in enumerate(zip(scenes_def, slice_durs), 1):
            vo_items.append({
                "index": idx,
                "scene_id": idx,
                "status": "ok",
                "narration": sc["narration"],
                "start_time": round(cur_t, 3),
                "end_time": round(cur_t + d, 3),
                "duration_seconds": d,
                "timing_type": "estimated",
            })
            cur_t += d

        prev = {
            "storyboard": {
                "storyboard": {
                    "scenes": scenes_def,
                }
            },
            "images": {
                "images": {
                    "items": img_items,
                }
            },
            "voiceover": {
                "continuous": True,
                "audio_file": cont_audio_rel,
                "duration_seconds": total_audio_duration,
                "voiceover": {
                    "continuous": True,
                    "audio_file": cont_audio_rel,
                    "items": vo_items,
                },
            },
        }

        print("[2] Executing generate_render_step with continuous audio...")
        t0 = time.perf_counter()
        render_res = generate_render_step(video, prev, None)
        render_latency = round(time.perf_counter() - t0, 3)
        print(f"    Render completed in {render_latency}s")

        final_mp4 = Path(render_res["video_path"])
        assert final_mp4.is_file(), f"Rendered video does not exist: {final_mp4}"
        print(f"    Rendered MP4: {final_mp4} ({final_mp4.stat().st_size / 1024:.1f} KB)")
        print(f"    Audio Source Reported: {render_res['render']['audio_source']}")
        print(f"    Audio File Used: {render_res['render']['audio_file']}")

        # 3. FFprobe deep inspection
        print("\n[3] Deep FFprobe Stream & Format Analysis:")
        probe = run_ffprobe(final_mp4)
        streams = probe["streams"]
        v_stream = next((s for s in streams if s["codec_type"] == "video"), None)
        a_stream = next((s for s in streams if s["codec_type"] == "audio"), None)
        fmt = probe["format"]

        assert v_stream is not None, "Missing video stream!"
        assert a_stream is not None, "Missing audio stream!"

        v_codec = v_stream.get("codec_name")
        v_pix_fmt = v_stream.get("pix_fmt")
        v_w = v_stream.get("width")
        v_h = v_stream.get("height")
        v_fps = v_stream.get("r_frame_rate")
        v_dur = float(v_stream.get("duration", 0.0))
        v_frames = int(v_stream.get("nb_frames", 0))

        a_codec = a_stream.get("codec_name")
        a_channels = a_stream.get("channels")
        a_sr = a_stream.get("sample_rate")
        a_dur = float(a_stream.get("duration", 0.0))
        fmt_dur = float(fmt.get("duration", 0.0))

        print(f"    Video Stream: codec={v_codec}, pix_fmt={v_pix_fmt}, res={v_w}x{v_h}, fps={v_fps}")
        print(f"                  duration={v_dur:.4f}s, nb_frames={v_frames}")
        print(f"    Audio Stream: codec={a_codec}, channels={a_channels}, rate={a_sr} Hz")
        print(f"                  duration={a_dur:.4f}s")
        print(f"    Format Duration: {fmt_dur:.4f}s (Audio input was {total_audio_duration:.4f}s)")

        # Invariants verification
        assert v_codec == "h264", f"Unexpected video codec: {v_codec}"
        assert v_pix_fmt == "yuv420p", f"Unexpected pixel format: {v_pix_fmt}"
        assert v_w == 1080 and v_h == 1920, f"Unexpected resolution: {v_w}x{v_h}"
        assert a_codec == "aac", f"Unexpected audio codec: {a_codec}"
        assert a_channels == 2, f"Unexpected audio channels: {a_channels}"
        assert a_sr == "44100", f"Unexpected audio sample rate: {a_sr}"

        # Verification of audio/video synchronization & no drift
        drift = abs(a_dur - total_audio_duration)
        assert drift < 0.05, f"Audio duration drifted by {drift:.4f}s (expected ~{total_audio_duration}s, got {a_dur}s)"
        v_a_diff = abs(v_dur - a_dur)
        assert v_a_diff < 0.1, f"Video/Audio stream duration mismatch: {v_a_diff:.4f}s"
        print(f"    Synchronization Drift: {drift:.4f}s (Pass - within <50ms frame tolerance)")

        # Verify Ken Burns animations preserved
        anims = render_res["render"]["animations"]
        print(f"\n[4] Camera Animation Validation: {len(anims)} scene motions generated")
        for a in anims:
            print(f"    Scene {a['scene_index']}: {a['animation_type']} (focal: {a.get('focal_coords')})")
        assert len(anims) == 3

        # 4. Subtitles pipeline step execution with continuous narration video
        print("\n[5] Executing Subtitles Step with Rendered Video...")
        prev["render"] = render_res["render"]
        video.output_path = str(final_mp4)

        t_sub0 = time.perf_counter()
        sub_res = h_subtitles(video, prev, None)
        sub_lat = round(time.perf_counter() - t_sub0, 3)
        print(f"    Subtitles completed in {sub_lat}s")
        print(f"    Captioned video: {sub_res['subtitles']['captioned_video_path']}")
        captioned_mp4 = Path(sub_res["subtitles"]["captioned_video_path"])
        assert captioned_mp4.is_file(), "Captioned video was not generated!"
        print(f"    Captioned video size: {captioned_mp4.stat().st_size / 1024:.1f} KB")

        # Probe captioned MP4
        cap_probe = run_ffprobe(captioned_mp4)
        cap_fmt_dur = float(cap_probe["format"]["duration"])
        print(f"    Captioned video duration: {cap_fmt_dur:.4f}s (Audio input was {total_audio_duration:.4f}s)")
        assert abs(cap_fmt_dur - total_audio_duration) < 0.1

        print("\n" + "=" * 70)
        print("ALL VERIFICATION CHECKS PASSED SUCCESSFULLY!")
        print("=" * 70)


if __name__ == "__main__":
    main()
