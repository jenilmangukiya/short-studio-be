# Shorts Studio Backend — Project Context & Architecture

## Overview

Shorts Studio is a FastAPI + PostgreSQL 16 backend for automating short-form vertical video creation (YouTube Shorts, Instagram Reels, TikTok). Every video progresses through a 6-step deterministic pipeline:

```
script → storyboard → images → voiceover → render → subtitles
```

---

## Canonical Continuous Narration Architecture

Shorts Studio generates **one continuous narration audio file per video** rather than separate per-scene audio clips. Continuous narration is the **only** supported audio synthesis, rendering, and subtitle-alignment method.

### 1. Voiceover Step (`app/steps/voiceover.py` & `app/voiceover/continuous.py`)

- **Script Collection**: `collect_narration_script(scenes)` joins all storyboard scene narrations in scene order (`index` 1..N) into a single continuous script while preserving sentence punctuation and scene associations.
- **Continuous Synthesis**: `generate_continuous_audio(full_script, provider, voice=..., speed=...)` synthesizes the entire script in one continuous pass (splitting only at natural sentence boundaries when a script exceeds `max_chunk_chars=600`, and concatenating with natural inter-chunk breath pauses).
- **Silence Normalization**: Trailing/leading dead air and excessive internal pauses are normalized on the continuous WAV via `trim_silence_wav_bytes()`, adding a 150ms tail padding so the final word never clips.
- **Canonical Storage**:
  - Saved to `<STORAGE_DIR>/<video_id>/audio/narration_<hash8>.wav`.
  - No per-scene WAV files (`scene_XX_<hash8>.wav`) are generated or written to disk.
- **Content-Addressable Reuse**:
  - Computes a 64-character SHA-256 `audio_hash` over `narration_script|provider|model|voice|speed|continuous_v1`.
  - Reuses the existing continuous WAV (`reused=True`) when `audio_hash` matches and the continuous WAV exists on disk (`size > 44` bytes).
- **Scene Timing Metadata**:
  - `estimate_scene_timings(scenes, total_duration)` computes proportional scene timing windows (`start_time`, `end_time`, `duration_seconds`, `timing_type="estimated"`) stored in `voiceover.items`.
  - These scene timing records do **not** contain a `"file"` field; they exist strictly to tell the downstream `render` step how long each scene's image should remain on screen and to help the `subtitles` step group aligned words into scene segments.

### 2. Canonical Audio API (`app/routers/videos.py`)

- **`GET /videos/{video_id}/audio`**:
  - Serves the canonical continuous narration WAV file (`audio/wav`) for the video.
  - Enforces authentication, channel ownership, and `STORAGE_DIR` path-traversal protection.
  - Returns `ETag: "<audio_hash>"` and `Cache-Control: private, no-cache`.

### 3. Video Rendering Step (`app/steps/render.py`)

- **Preconditions**: Requires `voiceover` step output to provide `audio_file` pointing to a valid continuous narration WAV on disk. Missing `audio_file` raises a `PipelineError`.
- **Two-Stage Continuous Render Pipeline**:
  1. **Per-Scene Video-Only Segments**: Renders each scene image (`scene_01.png` .. `scene_NN.png`) into an silent H.264 `.mp4` segment using AI-directed Ken Burns camera motion (`app/animation/`) for that scene's allocated duration (`duration_seconds`).
  2. **Unified Continuous Audio Muxing**: Concatenates all silent video segments via FFmpeg `concat` demuxer and muxes the single continuous narration WAV (`-i <continuous_audio.wav> -c:v copy -c:a aac -b:a 192k -ar 44100 -ac 2 -shortest`) into `<video_id>/final_video.mp4`.
- Sets `"audio_source": "continuous"` in `render` output metadata.

### 4. Subtitle Alignment & Burn-In Step (`app/steps/subtitles.py` & `app/subtitles/acoustic.py`)

- **Preconditions**: Requires both `render` (`final_video.mp4`) and `voiceover` (`audio_file` continuous WAV) to be present on disk.
- **Forced Alignment Against Complete Narration**:
  - `AcousticSubtitleProvider.generate(scenes, continuous_audio_path=cont_abs)` runs offline Vosk speech recognition + dynamic-programming script alignment (`align_script_with_vosk`) across the single continuous narration WAV.
  - Produces exact word-level timestamps (`WordTimestamp(text, start, end, confidence)`) with `"timing_source": "forced_alignment"`.
  - Partitions the aligned words back into scene-indexed `SubtitleSegment` objects and derives accurate scene `start_time`, `end_time`, and `duration_seconds` from the aligned word boundaries (updating `voiceover.items` with `"timing_type": "aligned"`).
- **Subtitle Artifacts & Captioned Video**:
  - Writes `subtitles.json`, `subtitles.vtt`, `subtitles.srt`, and `subtitles.ass` under `storage/<video_id>/`.
  - Burns word-by-word highlighted ASS captions into `<video_id>/final_video_captioned.mp4`.

---

## Removed Legacy Per-Scene Audio Functionality

The obsolete per-scene audio architecture has been completely removed from the backend:

1. **Removed Per-Scene Audio Slicing**:
   - Deleted `slice_scene_audio()` from `app/voiceover/continuous.py` and `app/voiceover/__init__.py`.
   - Removed per-scene WAV file creation (`<video_id>/audio/scene_XX_<hash8>.wav`) and `"file"` attributes on `voiceover.items` in `app/steps/voiceover.py`.
2. **Removed Per-Scene Audio Endpoint**:
   - Removed `GET /videos/{video_id}/audio/{index}` (`get_scene_audio`) from `app/routers/videos.py`.
3. **Removed Legacy Per-Scene Render Fallback**:
   - Removed the `else` fallback branch in `app/steps/render.py` that probed per-scene WAV files, encoded per-scene audio tracks, and concatenated AV clips.
4. **Removed Legacy Per-Scene Waveform Subtitle Alignment**:
   - Removed `_align_scene_audio()` and the `scene_waveform` per-scene fallback loop from `app/subtitles/acoustic.py`.
   - Removed the per-scene file hashing fallback in `_compute_alignment_hash()` in `app/steps/subtitles.py`.
5. **Removed Obsolete Verification Scripts**:
   - Deleted `scripts/verify_phase_b_edge_cases.py`, `scripts/verify_pipeline_voiceover.py`, and `scripts/run_real_render.py`.

---

## Frontend Cleanup Requirements (Separate Frontend Task)

When updating the frontend repository to match this backend cleanup:

1. **Remove Calls to `GET /videos/{video_id}/audio/{index}`**:
   - This endpoint has been removed and now returns `404`.
2. **Use `GET /videos/{video_id}/audio` Exclusively**:
   - Render a single unified audio player for the voiceover step using `GET /videos/{video_id}/audio` (or the `audio_file` metadata field on `step.output.voiceover`).
3. **Stop Expecting `item.file` on `voiceover.items`**:
   - Entries in `step.output.voiceover.items` still provide `index`, `scene_id`, `status`, `narration`, `start_time`, `end_time`, `duration_seconds`, and `timing_type` (useful for displaying scene text and time ranges like `0.00s – 3.42s`), but no longer include per-scene `file` paths.
   - If the frontend offers "play scene", seek the single continuous `<audio>` element to `item.start_time` and pause at `item.end_time`.

