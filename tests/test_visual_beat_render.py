"""Unit and integration tests for Stage 3.4: VisualBeat renderer integration and beat execution.

Tests:
1. Cumulative frame-boundary calculations for beats (zero gaps, zero overlaps).
2. Frame allocation rounding and boundary conditions.
3. Usable beat resolution and storyboard metadata preservation.
4. Fallback to legacy scene rendering when beat timing is missing, invalid, or incomplete.
5. Beat image path resolution (parent image vs explicit source_image_id).
6. Crop feasibility handling: safe_reuse.
7. Crop feasibility handling: questionable_reuse (conservative scale clamp to 1.45x).
8. Crop feasibility handling: new_image_needed without generating new images (conservative full framing).
9. End-to-end synthetic render with multiple visual beats in correct order and durations.
10. End-to-end synthetic render with mixed scenes (beat-driven scene + legacy scene).
11. End-to-end synthetic render with new_image_needed fallback (zero API calls, fallback metadata recorded).
12. Preservation of canonical continuous narration audio and subtitle alignment.
13. Camera stability: smoothstep easing, even pixel dimensions, and absence of shake.
"""

import math
from pathlib import Path
import subprocess
import uuid
import pytest

from app.animation.crop import (
    QUESTIONABLE_REUSE_MAX_SCALE,
    SAFE_REUSE_MAX_SCALE,
    calculate_portrait_crop,
    evaluate_beat_crop_feasibility,
)
from app.config import settings
from app.images.mock import create_solid_png
from app.steps.render import (
    PipelineError,
    _calculate_beat_frame_allocations,
    _probe_video_file,
    _resolve_beat_image,
    _resolve_scene_beats,
    generate_render_step,
)
from app.voiceover.mock import create_mock_wav


class DummyVideo:
    def __init__(self, video_id=None):
        self.id = video_id or uuid.uuid4()
        self.topic = "Computer Chips Electricity"
        self.output_path = None


# ===========================================================================
# 1. Cumulative Frame Boundary & Allocation Tests
# ===========================================================================

def test_beat_frame_allocations_contiguous_and_exact():
    """Verify beat frames partition the scene with zero gaps, zero overlaps, and exact sum."""
    beats = [
        {"id": "b1", "order": 1, "start_time": 0.0, "end_time": 1.25},
        {"id": "b2", "order": 2, "start_time": 1.25, "end_time": 2.10},
        {"id": "b3", "order": 3, "start_time": 2.10, "end_time": 3.00},
    ]
    scene_start_frame = 30
    scene_total_frames = 90  # 3.0s at 30 fps
    fps = 30

    allocations = _calculate_beat_frame_allocations(
        beats=beats,
        scene_start_frame=scene_start_frame,
        scene_total_frames=scene_total_frames,
        target_fps=fps,
    )

    assert len(allocations) == 3

    # Check contiguous frame bounds: b[k].start_frame == b[k-1].end_frame
    total_allocated = 0
    prev_end = scene_start_frame
    for beat_data, b_start, b_frames, b_dur in allocations:
        assert b_start == prev_end
        assert b_frames >= 1
        assert b_dur > 0.0
        total_allocated += b_frames
        prev_end = b_start + b_frames

    # Exact sum equals scene frames
    assert total_allocated == scene_total_frames
    assert prev_end == scene_start_frame + scene_total_frames


def test_beat_frame_allocations_short_intervals_guarantee_min_one_frame():
    """Verify even sub-second micro beats receive at least 1 frame and end exactly at boundary."""
    beats = [
        {"id": "b1", "order": 1, "end_time": 0.02},
        {"id": "b2", "order": 2, "end_time": 0.04},
        {"id": "b3", "order": 3, "end_time": 0.06},
    ]
    scene_start = 0
    scene_frames = 5  # very short scene of 5 frames
    fps = 30

    allocations = _calculate_beat_frame_allocations(
        beats=beats,
        scene_start_frame=scene_start,
        scene_total_frames=scene_frames,
        target_fps=fps,
    )

    assert len(allocations) == 3
    frames = [b[2] for b in allocations]
    assert all(f >= 1 for f in frames)
    assert sum(frames) == scene_frames


# ===========================================================================
# 2. Usable Beat Resolution & Metadata Preservation Tests
# ===========================================================================

def test_resolve_scene_beats_merges_storyboard_and_timing():
    """Verify storyboard visual fields merge correctly with voiceover beat timing."""
    scene_meta = {
        "index": 1,
        "visual_beats": [
            {
                "id": "beat_1_1",
                "order": 1,
                "visual_description": "Motherboard with processor",
                "shot_type": "wide",
                "camera_angle": "low-angle",
                "focal_point": "Central chip",
                "visual_technique": "cinematic-photo",
                "image_strategy": "reuse_scene_image",
            },
            {
                "id": "beat_1_2",
                "order": 2,
                "visual_description": "Close up on silicon core",
                "shot_type": "close-up",
                "camera_angle": "eye-level",
                "focal_point": "Silicon die",
                "visual_technique": "macro-photo",
                "image_strategy": "reuse_scene_image",
            },
        ],
    }
    audio_meta = {
        "index": 1,
        "visual_beats": [
            {"order": 1, "start_time": 0.0, "end_time": 1.5, "timing_type": "aligned"},
            {"order": 2, "start_time": 1.5, "end_time": 3.2, "timing_type": "aligned"},
        ],
    }

    resolved, reason = _resolve_scene_beats(
        scene_idx=1,
        scene_meta=scene_meta,
        audio_meta=audio_meta,
        scene_start_time=0.0,
        scene_end_time=3.2,
    )

    assert reason is None
    assert len(resolved) == 2
    assert resolved[0]["shot_type"] == "wide"
    assert resolved[0]["camera_angle"] == "low-angle"
    assert resolved[0]["focal_point"] == "Central chip"
    assert resolved[0]["start_time"] == 0.0
    assert resolved[0]["end_time"] == 1.5

    assert resolved[1]["shot_type"] == "close-up"
    assert resolved[1]["focal_point"] == "Silicon die"
    assert resolved[1]["start_time"] == 1.5
    assert resolved[1]["end_time"] == 3.2


def test_resolve_scene_beats_fallback_on_missing_timing():
    """Verify missing timing fields cause fallback to scene-level rendering (returns [])."""
    scene_meta = {
        "index": 1,
        "visual_beats": [
            {"id": "b1", "order": 1, "visual_description": "Step 1"},
            {"id": "b2", "order": 2, "visual_description": "Step 2"},
        ],
    }
    # audio_meta missing visual_beats timing
    audio_meta = {"index": 1}

    resolved, reason = _resolve_scene_beats(
        scene_idx=1,
        scene_meta=scene_meta,
        audio_meta=audio_meta,
        scene_start_time=0.0,
        scene_end_time=2.5,
    )
    assert resolved == []
    assert reason == "missing_or_invalid_timing"


def test_resolve_scene_beats_fallback_on_inverted_timing():
    """Verify end_time <= start_time causes fallback to scene-level rendering."""
    scene_meta = {
        "index": 1,
        "visual_beats": [
            {"id": "b1", "order": 1, "visual_description": "Step 1"},
        ],
    }
    audio_meta = {
        "index": 1,
        "visual_beats": [
            {"id": "b1", "order": 1, "start_time": 2.0, "end_time": 1.0},  # inverted!
        ],
    }

    resolved, reason = _resolve_scene_beats(
        scene_idx=1,
        scene_meta=scene_meta,
        audio_meta=audio_meta,
        scene_start_time=0.0,
        scene_end_time=2.5,
    )
    assert resolved == []
    assert reason == "non_monotonic_or_overlapping"


# ===========================================================================
# 3. Beat Image Path Resolution Tests
# ===========================================================================

def test_resolve_beat_image_defaults_to_parent(tmp_path):
    """Verify beat with no source_image_id resolves to parent scene's image."""
    img_file = tmp_path / "scene_01.png"
    img_file.write_bytes(create_solid_png(100, 150, 10, 20, 30))

    valid_images = {1: {"file": "scene_01.png"}}
    img_items = [{"index": 1, "file": "scene_01.png"}]

    beat = {"id": "b1", "order": 1}
    abs_path, rel_path, is_distinct = _resolve_beat_image(
        beat=beat,
        scene_idx=1,
        valid_images=valid_images,
        img_items=img_items,
        storage_root=tmp_path,
    )

    assert abs_path == img_file
    assert rel_path == "scene_01.png"
    assert is_distinct is False


def test_resolve_beat_image_explicit_source_image_id(tmp_path):
    """Verify beat with source_image_id referencing another scene resolves to that image."""
    img1 = tmp_path / "scene_01.png"
    img1.write_bytes(create_solid_png(100, 150, 10, 20, 30))
    img2 = tmp_path / "scene_02.png"
    img2.write_bytes(create_solid_png(100, 150, 40, 50, 60))

    valid_images = {1: {"file": "scene_01.png"}, 2: {"file": "scene_02.png"}}
    img_items = [{"index": 1, "file": "scene_01.png"}, {"index": 2, "file": "scene_02.png"}]

    beat = {"id": "b1", "order": 1, "source_image_id": "2"}
    abs_path, rel_path, is_distinct = _resolve_beat_image(
        beat=beat,
        scene_idx=1,
        valid_images=valid_images,
        img_items=img_items,
        storage_root=tmp_path,
    )

    assert abs_path == img2
    assert rel_path == "scene_02.png"
    assert is_distinct is True


# ===========================================================================
# 4. Feasibility Classification & Handling Tests
# ===========================================================================

def test_crop_feasibility_safe_reuse():
    """Verify moderate reframing (wide to medium) yields safe_reuse."""
    beat = {"shot_type": "medium", "camera_angle": "eye-level"}
    parent = {"shot_type": "wide", "camera_angle": "eye-level"}

    feasibility = evaluate_beat_crop_feasibility(beat, parent, 1024, 1536)
    assert feasibility.status == "safe_reuse"
    assert feasibility.scale_factor <= SAFE_REUSE_MAX_SCALE


def test_crop_feasibility_questionable_reuse():
    """Verify higher reframing (wide to close-up) yields questionable_reuse."""
    beat = {"shot_type": "close-up", "camera_angle": "eye-level"}
    parent = {"shot_type": "wide", "camera_angle": "eye-level"}

    feasibility = evaluate_beat_crop_feasibility(beat, parent, 1024, 1536)
    assert feasibility.status == "questionable_reuse"
    assert SAFE_REUSE_MAX_SCALE < feasibility.scale_factor <= QUESTIONABLE_REUSE_MAX_SCALE


def test_crop_feasibility_new_image_needed():
    """Verify extreme reframing (wide to macro) or incompatible angle yields new_image_needed."""
    beat = {"shot_type": "macro", "camera_angle": "eye-level"}
    parent = {"shot_type": "wide", "camera_angle": "eye-level"}

    feasibility = evaluate_beat_crop_feasibility(beat, parent, 1024, 1536)
    assert feasibility.status == "new_image_needed"


# ===========================================================================
# 5. Controlled End-to-End FFmpeg Rendering Tests
# ===========================================================================

def test_end_to_end_render_with_visual_beats(tmp_path, monkeypatch):
    """Render a synthetic 0.8s clip containing 2 visual beats with local FFmpeg."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "render_fps", 30)

    video = DummyVideo()
    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)

    img_bytes = create_solid_png(1024, 1536, 30, 60, 120)
    img_rel = f"{video.id}/scene_01.png"
    (tmp_path / img_rel).write_bytes(img_bytes)

    wav_bytes = create_mock_wav(0.8, 24000)
    wav_rel = f"{video.id}/narration.wav"
    (tmp_path / wav_rel).write_bytes(wav_bytes)

    prev = {
        "storyboard": {
            "storyboard": {
                "scenes": [
                    {
                        "index": 1,
                        "shot_type": "wide",
                        "visual_beats": [
                            {
                                "id": "beat_1_1",
                                "order": 1,
                                "shot_type": "wide",
                                "visual_description": "Full view of setup",
                            },
                            {
                                "id": "beat_1_2",
                                "order": 2,
                                "shot_type": "medium",
                                "visual_description": "Closer view of device",
                            },
                        ],
                    }
                ]
            }
        },
        "images": {
            "images": {"items": [{"index": 1, "status": "ok", "file": img_rel}]}
        },
        "voiceover": {
            "audio_file": wav_rel,
            "voiceover": {
                "items": [
                    {
                        "index": 1,
                        "status": "ok",
                        "duration_seconds": 0.8,
                        "visual_beats": [
                            {"order": 1, "start_time": 0.0, "end_time": 0.4},
                            {"order": 2, "start_time": 0.4, "end_time": 0.8},
                        ],
                    }
                ]
            },
        },
    }

    res = generate_render_step(video, prev)

    assert "render" in res
    assert res["render"]["scenes_rendered"] == 1
    assert res["render"]["beats_rendered"] == 2
    assert res["render"]["render_mode"] == "visual_beats"

    # Verify animations metadata captures both beats
    animations = res["render"]["animations"]
    assert len(animations) == 2
    assert animations[0]["beat_id"] == "beat_1_1"
    assert animations[0]["render_mode"] == "visual_beat"
    assert animations[1]["beat_id"] == "beat_1_2"
    assert animations[1]["render_mode"] == "visual_beat"

    # Frame counts partition the total 24 frames (0.8s * 30fps)
    f1 = animations[0]["frames"]
    f2 = animations[1]["frames"]
    assert f1 + f2 == 24
    assert animations[0]["start_time"] == 0.0
    assert animations[1]["end_time"] == 0.8

    # Verify final video exists on disk
    final_file = Path(res["video_path"])
    assert final_file.is_file()
    assert final_file.stat().st_size > 1000

    # Verify clean temp directory cleanup
    tmp_dirs = list(video_dir.glob("tmp_render_*"))
    assert len(tmp_dirs) == 0


def test_end_to_end_render_mixed_scenes_beats_and_legacy(tmp_path, monkeypatch):
    """Render a synthetic 1.2s clip where Scene 1 has beats and Scene 2 uses legacy scene rendering."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "render_fps", 30)

    video = DummyVideo()
    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)

    img1_rel = f"{video.id}/scene_01.png"
    img2_rel = f"{video.id}/scene_02.png"
    (tmp_path / img1_rel).write_bytes(create_solid_png(1024, 1536, 10, 20, 30))
    (tmp_path / img2_rel).write_bytes(create_solid_png(1024, 1536, 40, 50, 60))

    wav_rel = f"{video.id}/narration.wav"
    (tmp_path / wav_rel).write_bytes(create_mock_wav(1.2, 24000))

    prev = {
        "storyboard": {
            "storyboard": {
                "scenes": [
                    {
                        "index": 1,
                        "shot_type": "wide",
                        "visual_beats": [
                            {"id": "b1_1", "order": 1, "shot_type": "wide"},
                            {"id": "b1_2", "order": 2, "shot_type": "medium"},
                        ],
                    },
                    {
                        "index": 2,
                        "shot_type": "close-up",
                        # Scene 2 has NO visual beats
                    },
                ]
            }
        },
        "images": {
            "images": {
                "items": [
                    {"index": 1, "status": "ok", "file": img1_rel},
                    {"index": 2, "status": "ok", "file": img2_rel},
                ]
            }
        },
        "voiceover": {
            "audio_file": wav_rel,
            "voiceover": {
                "items": [
                    {
                        "index": 1,
                        "status": "ok",
                        "duration_seconds": 0.6,
                        "visual_beats": [
                            {"order": 1, "start_time": 0.0, "end_time": 0.3},
                            {"order": 2, "start_time": 0.3, "end_time": 0.6},
                        ],
                    },
                    {
                        "index": 2,
                        "status": "ok",
                        "duration_seconds": 0.6,
                    },
                ]
            },
        },
    }

    res = generate_render_step(video, prev)

    assert res["render"]["scenes_rendered"] == 2
    assert res["render"]["beats_rendered"] == 2
    assert res["render"]["render_mode"] == "visual_beats"

    anims = res["render"]["animations"]
    assert len(anims) == 3
    # Scene 1: Beat 1 and Beat 2
    assert anims[0]["scene_index"] == 1
    assert anims[0]["render_mode"] == "visual_beat"
    assert anims[1]["scene_index"] == 1
    assert anims[1]["render_mode"] == "visual_beat"

    # Scene 2: Legacy scene-level render
    assert anims[2]["scene_index"] == 2
    assert anims[2]["render_mode"] == "scene"

    # Total frames match continuous duration (1.2s * 30fps = 36 frames)
    total_frames = sum(a["frames"] for a in anims)
    assert total_frames == 36


def test_end_to_end_render_new_image_needed_conservative_fallback(tmp_path, monkeypatch):
    """Verify beat with new_image_needed uses conservative full framing without generating images."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "render_fps", 30)

    video = DummyVideo()
    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)

    img_rel = f"{video.id}/scene_01.png"
    (tmp_path / img_rel).write_bytes(create_solid_png(1024, 1536, 80, 40, 20))

    wav_rel = f"{video.id}/narration.wav"
    (tmp_path / wav_rel).write_bytes(create_mock_wav(0.6, 24000))

    # Beat specifies 'macro' from 'wide' parent and 'new_image' strategy
    prev = {
        "storyboard": {
            "storyboard": {
                "scenes": [
                    {
                        "index": 1,
                        "shot_type": "wide",
                        "visual_beats": [
                            {
                                "id": "beat_extreme",
                                "order": 1,
                                "shot_type": "macro",
                                "image_strategy": "new_image",
                                "visual_description": "Sub-atomic detail of silicon",
                            }
                        ],
                    }
                ]
            }
        },
        "images": {
            "images": {"items": [{"index": 1, "status": "ok", "file": img_rel}]}
        },
        "voiceover": {
            "audio_file": wav_rel,
            "voiceover": {
                "items": [
                    {
                        "index": 1,
                        "status": "ok",
                        "duration_seconds": 0.6,
                        "visual_beats": [
                            {"order": 1, "start_time": 0.0, "end_time": 0.6}
                        ],
                    }
                ]
            },
        },
    }

    res = generate_render_step(video, prev)

    assert "render" in res
    anim = res["render"]["animations"][0]
    assert anim["feasibility_status"] == "new_image_needed"
    assert anim["fallback_applied"] == "conservative_parent_framing"
    assert len(anim["warnings"]) > 0
    assert "requires new image" in anim["warnings"][0]

    # Video successfully renders using the fallback
    final_file = Path(res["video_path"])
    assert final_file.is_file()


# ===========================================================================
# 6. Scope 1 — Audio/Video Duration Integrity & Boundary Tests
# ===========================================================================

def test_render_audio_matching_frame_boundary(tmp_path, monkeypatch):
    """Audio duration exactly on a frame boundary (0.600s = 18 frames at 30 fps)."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "render_fps", 30)

    video = DummyVideo()
    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)

    img_rel = f"{video.id}/scene_01.png"
    (tmp_path / img_rel).write_bytes(create_solid_png(1024, 1536, 10, 20, 30))

    wav_rel = f"{video.id}/narration.wav"
    # Exactly 0.600s
    (tmp_path / wav_rel).write_bytes(create_mock_wav(0.600, 24000))

    prev = {
        "storyboard": {
            "storyboard": {
                "scenes": [
                    {
                        "index": 1,
                        "shot_type": "wide",
                        "visual_beats": [
                            {"id": "b1", "order": 1, "start_time": 0.0, "end_time": 0.6},
                        ],
                    }
                ]
            }
        },
        "images": {"images": {"items": [{"index": 1, "status": "ok", "file": img_rel}]}},
        "voiceover": {
            "audio_file": wav_rel,
            "voiceover": {
                "items": [
                    {
                        "index": 1,
                        "status": "ok",
                        "duration_seconds": 0.600,
                        "visual_beats": [
                            {"order": 1, "start_time": 0.0, "end_time": 0.600},
                        ],
                    }
                ]
            },
        },
    }

    res = generate_render_step(video, prev)
    assert res["render"]["scenes_rendered"] == 1
    probe = _probe_video_file(Path(res["video_path"]))
    v_stream = next(s for s in probe["streams"] if s.get("codec_type") == "video")
    a_stream = next(s for s in probe["streams"] if s.get("codec_type") == "audio")
    v_dur = float(v_stream.get("duration") or probe["format"]["duration"])
    a_dur = float(a_stream.get("duration") or probe["format"]["duration"])
    assert abs(v_dur - 0.600) < 0.05
    assert a_dur >= 0.55


def test_render_audio_above_frame_boundary(tmp_path, monkeypatch):
    """Audio duration slightly above frame boundary (0.612s -> 19 frames = 0.633s).
    Ceiling policy and apad guarantee zero truncation of the narration audio.
    """
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "render_fps", 30)

    video = DummyVideo()
    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)

    img_rel = f"{video.id}/scene_01.png"
    (tmp_path / img_rel).write_bytes(create_solid_png(1024, 1536, 10, 20, 30))

    wav_rel = f"{video.id}/narration.wav"
    # 0.612s (at 30fps: 0.612 * 30 = 18.36 -> math.ceil is 19 frames = 0.6333s)
    (tmp_path / wav_rel).write_bytes(create_mock_wav(0.612, 24000))

    prev = {
        "storyboard": {
            "storyboard": {
                "scenes": [
                    {"index": 1, "shot_type": "wide"}
                ]
            }
        },
        "images": {"images": {"items": [{"index": 1, "status": "ok", "file": img_rel}]}},
        "voiceover": {
            "audio_file": wav_rel,
            "voiceover": {
                "items": [
                    {"index": 1, "status": "ok", "duration_seconds": 0.612}
                ]
            },
        },
    }

    res = generate_render_step(video, prev)
    # Total frames rendered should be 19 frames
    assert res["render"]["animations"][0]["frames"] == 19
    probe = _probe_video_file(Path(res["video_path"]))
    v_stream = next(s for s in probe["streams"] if s.get("codec_type") == "video")
    a_stream = next(s for s in probe["streams"] if s.get("codec_type") == "audio")
    v_dur = float(v_stream.get("duration") or probe["format"]["duration"])
    a_dur = float(a_stream.get("duration") or probe["format"]["duration"])
    # Audio stream duration must be >= continuous_duration (0.612s)
    assert a_dur >= 0.60
    # Video stream duration matches 19 frames = ~0.633s
    assert abs(v_dur - 19 / 30) < 0.05


def test_render_audio_below_frame_boundary(tmp_path, monkeypatch):
    """Audio duration slightly below frame boundary (0.590s -> 18 frames = 0.600s)."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "render_fps", 30)

    video = DummyVideo()
    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)

    img_rel = f"{video.id}/scene_01.png"
    (tmp_path / img_rel).write_bytes(create_solid_png(1024, 1536, 10, 20, 30))

    wav_rel = f"{video.id}/narration.wav"
    (tmp_path / wav_rel).write_bytes(create_mock_wav(0.590, 24000))

    prev = {
        "storyboard": {"storyboard": {"scenes": [{"index": 1, "shot_type": "wide"}]}},
        "images": {"images": {"items": [{"index": 1, "status": "ok", "file": img_rel}]}},
        "voiceover": {
            "audio_file": wav_rel,
            "voiceover": {
                "items": [{"index": 1, "status": "ok", "duration_seconds": 0.590}]
            },
        },
    }

    res = generate_render_step(video, prev)
    assert res["render"]["animations"][0]["frames"] == 18
    probe = _probe_video_file(Path(res["video_path"]))
    v_stream = next(s for s in probe["streams"] if s.get("codec_type") == "video")
    a_stream = next(s for s in probe["streams"] if s.get("codec_type") == "audio")
    v_dur = float(v_stream.get("duration") or probe["format"]["duration"])
    a_dur = float(a_stream.get("duration") or probe["format"]["duration"])
    assert abs(v_dur - 18 / 30) < 0.05
    assert a_dur >= 0.58


def test_render_final_audio_content_not_truncated(tmp_path, monkeypatch):
    """Verify high-frequency tone placed at the very tail of the audio file is preserved in final MP4."""
    import io
    import struct
    import wave

    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "render_fps", 30)

    video = DummyVideo()
    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)

    img_rel = f"{video.id}/scene_01.png"
    (tmp_path / img_rel).write_bytes(create_solid_png(1024, 1536, 10, 20, 30))

    # Create audio: 0.615s total (silence for first 0.580s, loud 1000Hz tone for last 0.035s)
    # At 30fps: 0.615 * 30 = 18.45 -> 19 frames (0.633s)
    sr = 24000
    total_samples = int(0.615 * sr)
    tail_samples = int(0.035 * sr)
    silence_samples = total_samples - tail_samples

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        # silence
        for _ in range(silence_samples):
            frames.extend(struct.pack("<h", 0))
        # tail tone
        for i in range(tail_samples):
            val = int(25000 * math.sin(2 * math.pi * 1000 * (i / sr)))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(frames)

    wav_rel = f"{video.id}/narration.wav"
    (tmp_path / wav_rel).write_bytes(buf.getvalue())

    prev = {
        "storyboard": {"storyboard": {"scenes": [{"index": 1, "shot_type": "wide"}]}},
        "images": {"images": {"items": [{"index": 1, "status": "ok", "file": img_rel}]}},
        "voiceover": {
            "audio_file": wav_rel,
            "voiceover": {"items": [{"index": 1, "status": "ok", "duration_seconds": 0.615}]},
        },
    }

    res = generate_render_step(video, prev)
    mp4_path = Path(res["video_path"])

    # Extract audio stream from rendered MP4 and verify the tail tone is present (non-zero energy in last 30ms)
    extracted_wav = tmp_path / "extracted.wav"
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp4_path), "-vn", "-c:a", "pcm_s16le", "-ac", "1", "-ar", "24000", str(extracted_wav)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    with wave.open(str(extracted_wav), "rb") as wf:
        n_frames = wf.getnframes()
        extracted_frames = wf.readframes(n_frames)
        samples = struct.unpack(f"<{n_frames}h", extracted_frames)
        # Check samples around the 0.585s mark (sample index ~ 14000 at 24kHz)
        # Verify maximum amplitude in the tail region exceeds threshold
        tail_slice = samples[int(0.585 * sr):int(0.615 * sr)]
        max_amplitude = max(abs(s) for s in tail_slice)
        assert max_amplitude > 5000, f"Expected loud tail signal, got max amplitude {max_amplitude}"


def test_probe_video_file_stream_duration_validation(tmp_path):
    """Test _probe_video_file validation logic on valid MP4 and corrupted file."""
    mp4_file = tmp_path / "test.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "color=c=black:s=64x64:d=0.6:r=30",
            "-f", "lavfi", "-i", "anullsrc=r=24000:cl=mono",
            "-t", "0.6",
            "-c:v", "libx264", "-c:a", "aac",
            str(mp4_file),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    probe = _probe_video_file(mp4_file)
    assert "format" in probe
    assert "streams" in probe
    assert len(probe["streams"]) >= 2
    assert float(probe["format"]["duration"]) > 0.5

    # Corrupted / invalid file should raise PipelineError
    corrupt_file = tmp_path / "bad.mp4"
    corrupt_file.write_bytes(b"not a valid mp4 file")
    with pytest.raises(PipelineError, match="ffprobe inspection failed"):
        _probe_video_file(corrupt_file)


# ===========================================================================
# 7. Scope 2 — Whole-Scene Timestamp Normalization Tests
# ===========================================================================

def test_resolve_scene_beats_relative_in_first_scene():
    """Scene 1 starting at t=0.0s with relative timestamps [0.0, 1.0] and [1.0, 2.0]."""
    scene_meta = {
        "index": 1,
        "visual_beats": [
            {"id": "b1", "order": 1, "shot_type": "wide"},
            {"id": "b2", "order": 2, "shot_type": "medium"},
        ],
    }
    audio_meta = {
        "index": 1,
        "visual_beats": [
            {"order": 1, "start_time": 0.0, "end_time": 1.0},
            {"order": 2, "start_time": 1.0, "end_time": 2.0},
        ],
    }

    resolved, reason = _resolve_scene_beats(
        scene_idx=1,
        scene_meta=scene_meta,
        audio_meta=audio_meta,
        scene_start_time=0.0,
        scene_end_time=2.0,
    )
    assert reason is None
    assert len(resolved) == 2
    assert resolved[0]["start_time"] == 0.0
    assert resolved[0]["end_time"] == 1.0
    assert resolved[1]["start_time"] == 1.0
    assert resolved[1]["end_time"] == 2.0


def test_resolve_scene_beats_relative_in_early_nonzero_scene():
    """Scene starting at t=0.35s with relative timestamps [0.0, 0.4] and [0.4, 1.0].
    Both beats must shift uniformly to [0.35, 0.75] and [0.75, 1.35].
    """
    scene_meta = {
        "index": 2,
        "visual_beats": [
            {"id": "b2_1", "order": 1, "shot_type": "wide"},
            {"id": "b2_2", "order": 2, "shot_type": "close-up"},
        ],
    }
    audio_meta = {
        "index": 2,
        "visual_beats": [
            {"order": 1, "start_time": 0.0, "end_time": 0.4},
            {"order": 2, "start_time": 0.4, "end_time": 1.0},
        ],
    }

    resolved, reason = _resolve_scene_beats(
        scene_idx=2,
        scene_meta=scene_meta,
        audio_meta=audio_meta,
        scene_start_time=0.35,
        scene_end_time=1.35,
    )
    assert reason is None
    assert len(resolved) == 2
    assert round(resolved[0]["start_time"], 4) == 0.35
    assert round(resolved[0]["end_time"], 4) == 0.75
    assert round(resolved[1]["start_time"], 4) == 0.75
    assert round(resolved[1]["end_time"], 4) == 1.35


def test_resolve_scene_beats_continuous_in_later_scene():
    """Scene starting at t=10.0s with continuous timestamps [10.0, 12.0] and [12.0, 14.0].
    Timestamps are recognized as continuous and remain unshifted.
    """
    scene_meta = {
        "index": 3,
        "visual_beats": [
            {"id": "b3_1", "order": 1, "shot_type": "wide"},
            {"id": "b3_2", "order": 2, "shot_type": "medium"},
        ],
    }
    audio_meta = {
        "index": 3,
        "visual_beats": [
            {"order": 1, "start_time": 10.0, "end_time": 12.0},
            {"order": 2, "start_time": 12.0, "end_time": 14.0},
        ],
    }

    resolved, reason = _resolve_scene_beats(
        scene_idx=3,
        scene_meta=scene_meta,
        audio_meta=audio_meta,
        scene_start_time=10.0,
        scene_end_time=14.0,
    )
    assert reason is None
    assert len(resolved) == 2
    assert resolved[0]["start_time"] == 10.0
    assert resolved[0]["end_time"] == 12.0
    assert resolved[1]["start_time"] == 12.0
    assert resolved[1]["end_time"] == 14.0


def test_resolve_scene_beats_relative_partial_shift_edge_case():
    """Scene starting at 0.35s with [0.0, 0.20] and [0.20, 1.00].
    Under per-beat heuristics, beat 2 (0.20) was NOT shifted because 0.20 > 0.35 - 0.20 = 0.15.
    Under whole-scene classification, both beats shift together uniformly.
    """
    scene_meta = {
        "index": 2,
        "visual_beats": [
            {"id": "b2_1", "order": 1, "shot_type": "wide"},
            {"id": "b2_2", "order": 2, "shot_type": "close-up"},
        ],
    }
    audio_meta = {
        "index": 2,
        "visual_beats": [
            {"order": 1, "start_time": 0.0, "end_time": 0.20},
            {"order": 2, "start_time": 0.20, "end_time": 1.00},
        ],
    }

    resolved, reason = _resolve_scene_beats(
        scene_idx=2,
        scene_meta=scene_meta,
        audio_meta=audio_meta,
        scene_start_time=0.35,
        scene_end_time=1.35,
    )
    assert reason is None
    assert len(resolved) == 2
    assert round(resolved[0]["start_time"], 4) == 0.35
    assert round(resolved[0]["end_time"], 4) == 0.55
    assert round(resolved[1]["start_time"], 4) == 0.55
    assert round(resolved[1]["end_time"], 4) == 1.35


def test_resolve_scene_beats_ambiguous_timestamps():
    """Timestamps that cannot be unambiguously classified as continuous or relative return ambiguous_timing."""
    scene_meta = {
        "index": 2,
        "visual_beats": [
            {"id": "b1", "order": 1, "shot_type": "wide"},
            {"id": "b2", "order": 2, "shot_type": "medium"},
        ],
    }
    # Scene starting at 0.10s with beats [0.05, 0.90] and [0.90, 1.80].
    # Satisfies both is_continuous (0.05 >= 0.10 - 0.25) and is_relative (0.05 <= 0.15).
    audio_meta = {
        "index": 2,
        "visual_beats": [
            {"order": 1, "start_time": 0.05, "end_time": 0.90},
            {"order": 2, "start_time": 0.90, "end_time": 1.80},
        ],
    }

    resolved, reason = _resolve_scene_beats(
        scene_idx=2,
        scene_meta=scene_meta,
        audio_meta=audio_meta,
        scene_start_time=0.10,
        scene_end_time=2.00,
    )
    assert resolved == []
    assert reason == "ambiguous_timing"


def test_resolve_scene_beats_out_of_bounds_timestamps():
    """Timestamps extending past scene boundary return out_of_bounds_timing."""
    scene_meta = {
        "index": 2,
        "visual_beats": [
            {"id": "b1", "order": 1, "shot_type": "wide"},
        ],
    }
    # Scene runs from 2.0 to 4.0. Beat ends at 7.0 (way past 4.0 + 0.35)
    audio_meta = {
        "index": 2,
        "visual_beats": [
            {"order": 1, "start_time": 2.0, "end_time": 7.0},
        ],
    }

    resolved, reason = _resolve_scene_beats(
        scene_idx=2,
        scene_meta=scene_meta,
        audio_meta=audio_meta,
        scene_start_time=2.0,
        scene_end_time=4.0,
    )
    assert resolved == []
    assert reason == "out_of_bounds_timing"


# ===========================================================================
# 8. Scope 3 — Fallback Reason Observability Tests
# ===========================================================================

def test_fallback_observability_no_beats(tmp_path, monkeypatch):
    """Legacy scene without visual beats has beat_fallback_reason: None."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "render_fps", 30)

    video = DummyVideo()
    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)

    img_rel = f"{video.id}/scene_01.png"
    (tmp_path / img_rel).write_bytes(create_solid_png(1024, 1536, 10, 20, 30))
    wav_rel = f"{video.id}/narration.wav"
    (tmp_path / wav_rel).write_bytes(create_mock_wav(0.6, 24000))

    prev = {
        "storyboard": {"storyboard": {"scenes": [{"index": 1, "shot_type": "wide"}]}},
        "images": {"images": {"items": [{"index": 1, "status": "ok", "file": img_rel}]}},
        "voiceover": {
            "audio_file": wav_rel,
            "voiceover": {"items": [{"index": 1, "status": "ok", "duration_seconds": 0.6}]},
        },
    }

    res = generate_render_step(video, prev)
    anim = res["render"]["animations"][0]
    assert anim["render_mode"] == "scene"
    assert anim["beat_fallback_reason"] is None
    assert res["render"]["beat_fallback_reasons"] == {}


def test_fallback_observability_missing_timing(tmp_path, monkeypatch):
    """Scene with missing beat timing records beat_fallback_reason: 'missing_or_invalid_timing'."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "render_fps", 30)

    video = DummyVideo()
    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)

    img_rel = f"{video.id}/scene_01.png"
    (tmp_path / img_rel).write_bytes(create_solid_png(1024, 1536, 10, 20, 30))
    wav_rel = f"{video.id}/narration.wav"
    (tmp_path / wav_rel).write_bytes(create_mock_wav(0.6, 24000))

    prev = {
        "storyboard": {
            "storyboard": {
                "scenes": [
                    {
                        "index": 1,
                        "shot_type": "wide",
                        "visual_beats": [{"id": "b1", "order": 1, "shot_type": "wide"}],
                    }
                ]
            }
        },
        "images": {"images": {"items": [{"index": 1, "status": "ok", "file": img_rel}]}},
        "voiceover": {
            "audio_file": wav_rel,
            "voiceover": {
                # Missing visual_beats timing
                "items": [{"index": 1, "status": "ok", "duration_seconds": 0.6}]
            },
        },
    }

    res = generate_render_step(video, prev)
    anim = res["render"]["animations"][0]
    assert anim["render_mode"] == "scene"
    assert anim["beat_fallback_reason"] == "missing_or_invalid_timing"
    assert res["render"]["beat_fallback_reasons"] == {1: "missing_or_invalid_timing"}


def test_fallback_observability_non_monotonic(tmp_path, monkeypatch):
    """Scene with non-monotonic timing records beat_fallback_reason: 'non_monotonic_or_overlapping'."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "render_fps", 30)

    video = DummyVideo()
    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)

    img_rel = f"{video.id}/scene_01.png"
    (tmp_path / img_rel).write_bytes(create_solid_png(1024, 1536, 10, 20, 30))
    wav_rel = f"{video.id}/narration.wav"
    (tmp_path / wav_rel).write_bytes(create_mock_wav(0.6, 24000))

    prev = {
        "storyboard": {
            "storyboard": {
                "scenes": [
                    {
                        "index": 1,
                        "shot_type": "wide",
                        "visual_beats": [
                            {"id": "b1", "order": 1, "shot_type": "wide"},
                            {"id": "b2", "order": 2, "shot_type": "close-up"},
                        ],
                    }
                ]
            }
        },
        "images": {"images": {"items": [{"index": 1, "status": "ok", "file": img_rel}]}},
        "voiceover": {
            "audio_file": wav_rel,
            "voiceover": {
                "items": [
                    {
                        "index": 1,
                        "status": "ok",
                        "duration_seconds": 0.6,
                        "visual_beats": [
                            {"order": 1, "start_time": 0.4, "end_time": 0.2},  # Inverted!
                            {"order": 2, "start_time": 0.2, "end_time": 0.6},
                        ],
                    }
                ]
            },
        },
    }

    res = generate_render_step(video, prev)
    anim = res["render"]["animations"][0]
    assert anim["render_mode"] == "scene"
    assert anim["beat_fallback_reason"] == "non_monotonic_or_overlapping"
    assert res["render"]["beat_fallback_reasons"] == {1: "non_monotonic_or_overlapping"}


def test_fallback_observability_insufficient_frames(tmp_path, monkeypatch):
    """Scene with fewer frames than visual beats records beat_fallback_reason: 'insufficient_frames'."""
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    monkeypatch.setattr(settings, "render_fps", 30)

    video = DummyVideo()
    video_dir = tmp_path / str(video.id)
    video_dir.mkdir(parents=True, exist_ok=True)

    img_rel = f"{video.id}/scene_01.png"
    (tmp_path / img_rel).write_bytes(create_solid_png(1024, 1536, 10, 20, 30))
    # 0.6s at 30fps = 18 frames
    wav_rel = f"{video.id}/narration.wav"
    (tmp_path / wav_rel).write_bytes(create_mock_wav(0.6, 24000))

    # 20 visual beats > 18 frames
    sb_beats = [{"id": f"b_{i}", "order": i, "shot_type": "wide"} for i in range(1, 21)]
    vo_beats = [
        {"order": i, "start_time": round((i - 1) * 0.03, 3), "end_time": round(i * 0.03, 3)}
        for i in range(1, 21)
    ]

    prev = {
        "storyboard": {
            "storyboard": {
                "scenes": [
                    {
                        "index": 1,
                        "shot_type": "wide",
                        "visual_beats": sb_beats,
                    }
                ]
            }
        },
        "images": {"images": {"items": [{"index": 1, "status": "ok", "file": img_rel}]}},
        "voiceover": {
            "audio_file": wav_rel,
            "voiceover": {
                "items": [
                    {
                        "index": 1,
                        "status": "ok",
                        "duration_seconds": 0.6,
                        "visual_beats": vo_beats,
                    }
                ]
            },
        },
    }

    # At 30fps, 0.6s is 18 frames, but we have 20 beats! 18 < 20 -> insufficient_frames
    res = generate_render_step(video, prev)
    anim = res["render"]["animations"][0]
    assert anim["render_mode"] == "scene"
    assert anim["beat_fallback_reason"] == "insufficient_frames"
    assert res["render"]["beat_fallback_reasons"] == {1: "insufficient_frames"}


# ===========================================================================
# 9. Scope 4 — Frame Allocation Defensive Guard
# ===========================================================================

def test_beat_frame_allocations_insufficient_frames_guard():
    """Scene with scene_total_frames < len(beats) returns [] without negative or inverted frames."""
    beats = [
        {"id": "b1", "order": 1, "start_time": 0.0, "end_time": 0.25},
        {"id": "b2", "order": 2, "start_time": 0.25, "end_time": 0.50},
        {"id": "b3", "order": 3, "start_time": 0.50, "end_time": 0.75},
        {"id": "b4", "order": 4, "start_time": 0.75, "end_time": 1.00},
    ]

    allocations = _calculate_beat_frame_allocations(
        beats=beats,
        scene_start_frame=0,
        scene_total_frames=2,  # 2 frames for 4 beats!
        target_fps=30,
    )
    assert allocations == []


# ===========================================================================
# 10. Stage 4.5 Camera Movement & Visual Motion Fix Tests
# ===========================================================================


def _create_test_image(storage_root, video_id, idx):
    img_bytes = create_solid_png(1024, 1536, 30, 60, 120)
    rel = f"{video_id}/scene_{idx:02d}.png"
    p = storage_root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(img_bytes)
    return rel


def _create_test_wav(storage_root, video_id, duration_sec):
    wav_bytes = create_mock_wav(duration_sec, 24000)
    rel = f"{video_id}/narration.wav"
    p = storage_root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(wav_bytes)
    return rel, duration_sec


def test_render_new_image_beats_receive_duration_aware_movement(tmp_path, monkeypatch):
    """Beats > 2.0s with image_strategy='new_image' receive active camera movement instead of static."""
    storage_root = tmp_path / "storage"
    storage_root.mkdir(parents=True)
    monkeypatch.setattr(settings, "storage_dir", str(storage_root))

    video = DummyVideo()
    img_rel = _create_test_image(storage_root, video.id, 1)
    wav_rel, total_dur = _create_test_wav(storage_root, video.id, duration_sec=4.5)

    # 1 scene with 1 beat of 4.5s duration
    sb_beats = [
        {
            "id": "beat_1_1",
            "order": 1,
            "narration_text": "A long establishing scene.",
            "shot_type": "medium",
            "image_strategy": "new_image",
        }
    ]
    vo_beats = [
        {
            "id": "beat_1_1",
            "order": 1,
            "start_time": 0.0,
            "end_time": 4.5,
            "duration_seconds": 4.5,
            "timing_type": "aligned",
        }
    ]

    prev = {
        "storyboard": {
            "storyboard": {
                "scenes": [
                    {
                        "index": 1,
                        "shot_type": "medium",
                        "visual_beats": sb_beats,
                    }
                ]
            }
        },
        "images": {"images": {"items": [{"index": 1, "status": "ok", "file": img_rel}]}},
        "voiceover": {
            "audio_file": wav_rel,
            "voiceover": {
                "items": [
                    {"index": 1, "status": "ok", "duration_seconds": 4.5, "visual_beats": vo_beats}
                ]
            },
        },
    }

    res = generate_render_step(video, prev)
    anims = res["render"]["animations"]
    assert len(anims) == 1
    assert anims[0]["render_mode"] == "visual_beat"
    # Stage 4.5 fix: Must NOT be static! Must have active zoom motion for 4.5s hold!
    assert anims[0]["animation_type"] == "zoom-in"
    assert anims[0]["start_scale"] == 1.0
    assert anims[0]["end_scale"] == 1.07  # 7% zoom for > 4.0s


def test_render_beat_explicit_static_honored(tmp_path, monkeypatch):
    """Beats > 2.0s with explicit camera_plan.movement='static' remain static as requested."""
    storage_root = tmp_path / "storage"
    storage_root.mkdir(parents=True)
    monkeypatch.setattr(settings, "storage_dir", str(storage_root))

    video = DummyVideo()
    img_rel = _create_test_image(storage_root, video.id, 1)
    wav_rel, total_dur = _create_test_wav(storage_root, video.id, duration_sec=3.5)

    sb_beats = [
        {
            "id": "beat_1_1",
            "order": 1,
            "narration_text": "A static diagram shot.",
            "shot_type": "medium",
            "image_strategy": "new_image",
            "camera_plan": {"movement": "static"},
        }
    ]
    vo_beats = [
        {
            "id": "beat_1_1",
            "order": 1,
            "start_time": 0.0,
            "end_time": 3.5,
            "duration_seconds": 3.5,
            "timing_type": "aligned",
        }
    ]

    prev = {
        "storyboard": {
            "storyboard": {
                "scenes": [
                    {
                        "index": 1,
                        "shot_type": "medium",
                        "visual_beats": sb_beats,
                    }
                ]
            }
        },
        "images": {"images": {"items": [{"index": 1, "status": "ok", "file": img_rel}]}},
        "voiceover": {
            "audio_file": wav_rel,
            "voiceover": {
                "items": [
                    {"index": 1, "status": "ok", "duration_seconds": 3.5, "visual_beats": vo_beats}
                ]
            },
        },
    }

    res = generate_render_step(video, prev)
    anims = res["render"]["animations"]
    assert len(anims) == 1
    assert anims[0]["animation_type"] == "static"
    assert anims[0]["start_scale"] == anims[0]["end_scale"] == 1.0


def test_render_alternating_zoom_directions_across_scenes(tmp_path, monkeypatch):
    """Verify camera movement alternates directions (zoom-in -> zoom-out) across consecutive scenes."""
    storage_root = tmp_path / "storage"
    storage_root.mkdir(parents=True)
    monkeypatch.setattr(settings, "storage_dir", str(storage_root))

    video = DummyVideo()
    img1_rel = _create_test_image(storage_root, video.id, 1)
    img2_rel = _create_test_image(storage_root, video.id, 2)
    wav_rel, total_dur = _create_test_wav(storage_root, video.id, duration_sec=6.0)

    prev = {
        "storyboard": {
            "storyboard": {
                "scenes": [
                    {
                        "index": 1,
                        "shot_type": "medium",
                        "visual_beats": [
                            {"id": "b1_1", "order": 1, "shot_type": "medium", "image_strategy": "new_image"}
                        ],
                    },
                    {
                        "index": 2,
                        "shot_type": "medium",
                        "visual_beats": [
                            {"id": "b2_1", "order": 1, "shot_type": "medium", "image_strategy": "new_image"}
                        ],
                    },
                ]
            }
        },
        "images": {
            "images": {
                "items": [
                    {"index": 1, "status": "ok", "file": img1_rel},
                    {"index": 2, "status": "ok", "file": img2_rel},
                ]
            }
        },
        "voiceover": {
            "audio_file": wav_rel,
            "voiceover": {
                "items": [
                    {
                        "index": 1,
                        "status": "ok",
                        "duration_seconds": 3.0,
                        "visual_beats": [{"id": "b1_1", "order": 1, "start_time": 0.0, "end_time": 3.0, "duration_seconds": 3.0}],
                    },
                    {
                        "index": 2,
                        "status": "ok",
                        "duration_seconds": 3.0,
                        "visual_beats": [{"id": "b2_1", "order": 1, "start_time": 3.0, "end_time": 6.0, "duration_seconds": 3.0}],
                    },
                ]
            },
        },
    }

    res = generate_render_step(video, prev)
    anims = res["render"]["animations"]
    assert len(anims) == 2
    # Beat 1 starts with zoom-in
    assert anims[0]["animation_type"] == "zoom-in"
    assert anims[0]["end_scale"] == 1.05
    # Beat 2 alternates to zoom-out (pull-out)
    assert anims[1]["animation_type"] == "zoom-out"
    assert anims[1]["start_scale"] == 1.05
    assert anims[1]["end_scale"] == 1.00


def test_render_cross_scene_reuse_receives_active_movement(tmp_path, monkeypatch):
    """Verify cross-scene reused image beat > 2.0s receives active camera movement."""
    storage_root = tmp_path / "storage"
    storage_root.mkdir(parents=True)
    monkeypatch.setattr(settings, "storage_dir", str(storage_root))

    video = DummyVideo()
    img1_rel = _create_test_image(storage_root, video.id, 1)
    img2_rel = _create_test_image(storage_root, video.id, 2)
    wav_rel, total_dur = _create_test_wav(storage_root, video.id, duration_sec=6.0)

    prev = {
        "storyboard": {
            "storyboard": {
                "scenes": [
                    {
                        "index": 1,
                        "shot_type": "medium",
                        "visual_beats": [
                            {"id": "b1_1", "order": 1, "shot_type": "medium", "image_strategy": "new_image"}
                        ],
                    },
                    {
                        "index": 2,
                        "shot_type": "medium",
                        "visual_beats": [
                            {
                                "id": "b2_1",
                                "order": 1,
                                "shot_type": "medium",
                                "image_strategy": "reuse_previous_image",
                                "source_image_id": "1",
                            }
                        ],
                    },
                ]
            }
        },
        "images": {
            "images": {
                "items": [
                    {"index": 1, "status": "ok", "file": img1_rel},
                    {"index": 2, "status": "ok", "file": img2_rel},
                ]
            }
        },
        "voiceover": {
            "audio_file": wav_rel,
            "voiceover": {
                "items": [
                    {
                        "index": 1,
                        "status": "ok",
                        "duration_seconds": 3.0,
                        "visual_beats": [{"id": "b1_1", "order": 1, "start_time": 0.0, "end_time": 3.0, "duration_seconds": 3.0}],
                    },
                    {
                        "index": 2,
                        "status": "ok",
                        "duration_seconds": 3.0,
                        "visual_beats": [{"id": "b2_1", "order": 1, "start_time": 3.0, "end_time": 6.0, "duration_seconds": 3.0}],
                    },
                ]
            },
        },
    }

    res = generate_render_step(video, prev)
    anims = res["render"]["animations"]
    assert len(anims) == 2
    # Beat 2 resolved to Scene 1's image asset
    assert anims[1]["image_file"] == img1_rel
    # And received active camera movement
    assert anims[1]["animation_type"] == "zoom-out"
    assert anims[1]["render_mode"] == "visual_beat"



