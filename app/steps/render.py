"""Video rendering step implementation for Shorts Studio.

Composes vertical short videos (1080x1920, 30 fps, H.264 / AAC) by synchronizing
each storyboard scene's vertical image with its exact Kokoro narration audio
using FFmpeg via standard-library subprocess.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any
import uuid

from sqlalchemy.orm import object_session

from app.config import settings
from app.models import Video, VideoStep


class PipelineError(Exception):
    """Raised when preconditions or render pipeline execution fails."""


def _probe_video_file(file_path: Path) -> dict[str, Any]:
    """Inspect a rendered MP4 video using ffprobe and return parsed metadata."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate,channels,sample_rate,pix_fmt",
        "-show_entries", "format=duration,size,bit_rate",
        "-of", "json",
        str(file_path),
    ]

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=30.0,
        )
        return json.loads(proc.stdout)
    except subprocess.CalledProcessError as exc:
        raise PipelineError(f"ffprobe inspection failed for '{file_path}': {exc.stderr.strip()}") from exc
    except Exception as exc:
        raise PipelineError(f"Failed to inspect rendered video with ffprobe: {exc}") from exc


def generate_render_step(
    video: Video,
    prev: dict[str, Any],
    step: VideoStep | None = None,
) -> dict[str, Any]:
    """Execute the video rendering step using FFmpeg.

    Args:
        video: Video ORM model instance.
        prev: Dictionary containing outputs of preceding pipeline steps.
        step: Current VideoStep instance.

    Returns:
        Structured dictionary conforming to the pipeline handler interface.
    """
    render_start = time.perf_counter()

    # 1. Validate images step
    img_data = prev.get("images", {})
    if isinstance(img_data, dict) and isinstance(img_data.get("images"), dict):
        img_items = img_data["images"].get("items", [])
    elif isinstance(img_data, dict) and isinstance(img_data.get("items"), list):
        img_items = img_data["items"]
    else:
        img_items = []

    valid_images = {
        it["index"]: it
        for it in img_items
        if isinstance(it, dict) and it.get("index") and it.get("status") == "ok" and it.get("file")
    }
    if not valid_images:
        raise PipelineError("Images step must be completed with valid generated images before rendering")

    # 2. Validate voiceover step
    vo_data = prev.get("voiceover", {})
    if isinstance(vo_data, dict) and isinstance(vo_data.get("voiceover"), dict):
        vo_items = vo_data["voiceover"].get("items", [])
    elif isinstance(vo_data, dict) and isinstance(vo_data.get("items"), list):
        vo_items = vo_data["items"]
    else:
        vo_items = []

    valid_audio = {
        it["index"]: it
        for it in vo_items
        if isinstance(it, dict) and it.get("index") and it.get("status") == "ok" and it.get("file")
    }
    if not valid_audio:
        raise PipelineError("Voiceover step must be completed with valid generated audio before rendering")

    # 3. Check scene alignment between images and audio
    all_indices = sorted(set(valid_images.keys()) | set(valid_audio.keys()))
    missing_images = [idx for idx in all_indices if idx not in valid_images]
    missing_audio = [idx for idx in all_indices if idx not in valid_audio]

    if missing_images or missing_audio:
        err_parts = []
        if missing_images:
            err_parts.append(f"scenes missing images: {missing_images}")
        if missing_audio:
            err_parts.append(f"scenes missing audio: {missing_audio}")
        raise PipelineError(f"Cannot render: {', '.join(err_parts)}")

    # 4. Resolve storage paths
    storage_root = Path(settings.storage_dir).resolve()
    video_dir = storage_root / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)

    final_mp4_path = video_dir / "final_video.mp4"
    temp_dir = video_dir / f"tmp_render_{uuid.uuid4().hex[:8]}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_final_mp4 = temp_dir / "temp_final.mp4"

    # Render configurations
    target_width = getattr(settings, "render_width", 1080)
    target_height = getattr(settings, "render_height", 1920)
    target_fps = getattr(settings, "render_fps", 30)
    timeout_sec = getattr(settings, "render_timeout_seconds", 300.0)

    try:
        # 5. Render individual scene segments into temp_dir
        segment_paths: list[Path] = []
        expected_total_duration = 0.0

        # Extract storyboard scene metadata if available to guide camera motion
        sb_outer = prev.get("storyboard") or {}
        sb_inner = (
            sb_outer.get("storyboard")
            if isinstance(sb_outer, dict) and isinstance(sb_outer.get("storyboard"), dict)
            else (sb_outer if isinstance(sb_outer, dict) else {})
        )
        sb_scenes = sb_inner.get("scenes") if isinstance(sb_inner.get("scenes"), list) else []
        scenes_by_idx = {
            s["index"]: s
            for s in sb_scenes
            if isinstance(s, dict) and s.get("index")
        }

        for idx in all_indices:
            img_rel = valid_images[idx]["file"]
            vo_rel = valid_audio[idx]["file"]

            img_abs = (storage_root / img_rel).resolve()
            vo_abs = (storage_root / vo_rel).resolve()

            if not img_abs.is_file():
                raise PipelineError(f"Image file for scene {idx} does not exist: {img_abs}")
            if not vo_abs.is_file():
                raise PipelineError(f"Audio file for scene {idx} does not exist: {vo_abs}")

            scene_dur = float(valid_audio[idx].get("duration_seconds", 0.0))
            if scene_dur <= 0.0:
                try:
                    import soundfile as sf
                    scene_dur = float(sf.info(str(vo_abs)).duration)
                except Exception:
                    scene_dur = 2.5
            expected_total_duration += scene_dur

            seg_path = temp_dir / f"scene_{idx:02d}.mp4"

            # Determine motion direction based on scene composition / shot type
            sc_meta = scenes_by_idx.get(idx, {})
            shot_type = str(sc_meta.get("shot_type") or "medium").lower()

            # Restrained 4.5% - 5.5% zoom delta to keep subjects visible and avoid edge crop
            zoom_delta = 0.05
            total_frames = max(1, int(round(scene_dur * target_fps)))

            if shot_type in ("extreme-close-up", "close-up", "macro"):
                # Close/macro shots: gentle pull-out so tight details are never cropped
                z_expr = f"({1.0 + zoom_delta:.3f}-{zoom_delta:.3f}*(on/{total_frames}))"
            elif shot_type in ("diagram", "cutaway") and idx % 2 == 0:
                # Alternating diagram/cutaway shots: gentle pull-out for variety
                z_expr = f"({1.0 + zoom_delta:.3f}-{zoom_delta:.3f}*(on/{total_frames}))"
            else:
                # Wide, medium, in-context, establishing shots: gentle slow push-in
                z_expr = f"(1.0+{zoom_delta:.3f}*(on/{total_frames}))"

            # Anchor motion around subject in upper-middle of 9:16 frame (y ~ 42% height)
            x_expr = "iw/2-(iw/zoom/2)"
            y_expr = "(ih*0.42)-((ih/zoom)*0.42)"

            # Pre-scale & crop to exact 9:16 at 1.5x resolution to eliminate sub-pixel jitter
            intermediate_w = int(target_width * 1.5)
            intermediate_h = int(target_height * 1.5)

            filter_chain = (
                f"scale={intermediate_w}:{intermediate_h}:force_original_aspect_ratio=increase,"
                f"crop={intermediate_w}:{intermediate_h},"
                f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':"
                f"d={total_frames + 2}:s={target_width}x{target_height}:fps={target_fps}"
            )

            seg_cmd = [
                "ffmpeg",
                "-y",
                "-i", str(img_abs),
                "-i", str(vo_abs),
                "-filter_complex", f"[0:v]{filter_chain}[v]",
                "-map", "[v]",
                "-map", "1:a",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "19",
                "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                "-b:a", "192k",
                "-ar", "44100",
                "-ac", "2",
                "-shortest",
                str(seg_path),
            ]

            try:
                proc = subprocess.run(
                    seg_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=True,
                    timeout=timeout_sec,
                )
            except subprocess.CalledProcessError as exc:
                err_tail = exc.stderr[-500:] if exc.stderr else "Unknown error"
                raise PipelineError(f"FFmpeg failed rendering scene {idx}: {err_tail.strip()}") from exc
            except subprocess.TimeoutExpired as exc:
                raise PipelineError(f"FFmpeg timed out rendering scene {idx} after {timeout_sec}s") from exc

            if not seg_path.is_file() or seg_path.stat().st_size == 0:
                raise PipelineError(f"FFmpeg produced empty segment file for scene {idx}")

            segment_paths.append(seg_path)

        # 6. Concatenate all segments with -c copy and +faststart
        concat_file = temp_dir / "concat_list.txt"
        concat_lines = [f"file '{p.resolve()}'" for p in segment_paths]
        concat_file.write_text("\n".join(concat_lines), encoding="utf-8")

        concat_cmd = [
            "ffmpeg",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            "-movflags", "+faststart",
            str(temp_final_mp4),
        ]

        try:
            subprocess.run(
                concat_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
                timeout=timeout_sec,
            )
        except subprocess.CalledProcessError as exc:
            err_tail = exc.stderr[-500:] if exc.stderr else "Unknown error"
            raise PipelineError(f"FFmpeg concatenation failed: {err_tail.strip()}") from exc
        except subprocess.TimeoutExpired as exc:
            raise PipelineError(f"FFmpeg timed out concatenating video segments after {timeout_sec}s") from exc

        if not temp_final_mp4.is_file() or temp_final_mp4.stat().st_size == 0:
            raise PipelineError("FFmpeg produced empty concatenated video output")

        # 7. Validate rendered MP4 using ffprobe
        probe_data = _probe_video_file(temp_final_mp4)

        streams = probe_data.get("streams", [])
        v_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
        a_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

        if not v_stream:
            raise PipelineError("Rendered video contains no video stream")
        if not a_stream:
            raise PipelineError("Rendered video contains no audio stream")

        v_codec = v_stream.get("codec_name", "")
        if v_codec != "h264":
            raise PipelineError(f"Expected video codec 'h264', got '{v_codec}'")

        v_pix_fmt = v_stream.get("pix_fmt", "")
        if v_pix_fmt != "yuv420p":
            raise PipelineError(f"Expected pixel format 'yuv420p', got '{v_pix_fmt}'")

        v_width = v_stream.get("width")
        v_height = v_stream.get("height")
        if v_width != target_width or v_height != target_height:
            raise PipelineError(f"Expected resolution {target_width}x{target_height}, got {v_width}x{v_height}")

        a_codec = a_stream.get("codec_name", "")
        if a_codec != "aac":
            raise PipelineError(f"Expected audio codec 'aac', got '{a_codec}'")

        format_info = probe_data.get("format", {})
        actual_duration = round(float(format_info.get("duration", 0.0)), 3)
        file_size_bytes = int(format_info.get("size", temp_final_mp4.stat().st_size))

        if actual_duration <= 0.5:
            raise PipelineError(f"Rendered video duration ({actual_duration}s) is suspiciously short")

        # 8. Promote temp output to final destination atomically
        shutil.move(str(temp_final_mp4), str(final_mp4_path))

        # 9. Update Video model output_path and commit session if attached
        video.output_path = str(final_mp4_path)
        sess = None
        try:
            sess = object_session(video)
        except Exception:
            sess = None

        if sess is not None:
            try:
                sess.commit()
            except Exception:
                sess.rollback()

        render_latency = round(time.perf_counter() - render_start, 3)

        # 10. Construct structured output
        return {
            "render": {
                "video_path": str(final_mp4_path),
                "duration_seconds": actual_duration,
                "resolution": f"{target_width}x{target_height}",
                "fps": target_fps,
                "codec": "h264",
                "audio_codec": "aac",
                "file_size_bytes": file_size_bytes,
                "scenes_rendered": len(all_indices),
            },
            # Top-level video_path and duration_seconds for backward compatibility
            "video_path": str(final_mp4_path),
            "duration_seconds": actual_duration,
            "meta": {
                "task": "render",
                "scenes_count": len(all_indices),
                "resolution": f"{target_width}x{target_height}",
                "fps": target_fps,
                "latency_seconds": render_latency,
                "file_size_bytes": file_size_bytes,
            },
        }

    finally:
        # Clean up temporary segments directory
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
