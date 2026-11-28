# Shorts Studio Backend — Project Context & Architecture

## Overview

Shorts Studio is a FastAPI + PostgreSQL 16 backend for automating short-form vertical video creation (YouTube Shorts, Instagram Reels, TikTok). Every video progresses through a 6-step deterministic pipeline:

```
script → storyboard → images → voiceover → render → subtitles
```

---

## Storytelling, Script Quality & Visual Flow Architecture

### 1. Conversational Script Generation & Quality Checks (`app/prompts/script.md` & `app/steps/script.py`)

- **Spoken Storytelling Arc**: Scripts are written specifically for spoken short-form narration (`app/prompts/script.md`) as one connected cause-and-effect story rather than a list of disconnected facts:
  1. **Opening Hook (Sentence 1, 4–10 words ideal / max 15)**: Opens a concrete curiosity gap with a direct question or surprising claim.
  2. **Immediate Payoff / Myth-Bust (Sentences 2–3)**: Immediately addresses the hook (e.g., contrasting a common assumption with what actually happens) without throat-clearing filler.
  3. **Connected Cause-and-Effect Chain (Middle)**: Explains *how* and *why* using everyday words, plain-English explanations of any technical term, natural contractions, varied sentence rhythm, and explicit conversational transitions (`"Because of that..."`, `"That's why..."`, `"Instead..."`).
  4. **Memorable Final Insight (Last Sentence)**: Ends with a crisp, self-contained takeaway rather than academic hedging (`"no single explanation is settled"`) or social-media calls to action.
- **Automated Script Quality Gate (`evaluate_script_quality` & `estimate_spoken_duration_seconds`)**:
  - Runs automatically inside `validate_script_soft_checks()` before a script is accepted for storyboard generation.
  - Evaluates:
    1. **Sentence complexity & length**: Flags individual sentences >28 words, average sentence length >22.5 words, or stacked colons/semicolons.
    2. **Stiff textbook phrasing & unexplained jargon**: Flags encyclopedic disclaimers (`"studies haven't shown"`, `"scientists infer"`, `"referred to as"`) and dense 4+ syllable technical terms lacking plain-English context cues.
    3. **Disconnected fact-list writing & abrupt transitions**: Flags additive trivia chains (`"Furthermore..."`, `"X also..."`) and unlinked sentences that share neither discourse connectors nor rolling-window topic stems with preceding sentences.
    4. **Repetitive phrasing**: Flags consecutive identical sentence openings and repeated multi-word content trigrams.
    5. **Narrative progression**: Flags delayed payoff filler in sentence 2 and weak/hedged endings.
    6. **Spoken duration & TTS number formatting**: Estimates spoken duration (`estimate_spoken_duration_seconds`, targeting 36–65s for 110–150 words) and flags comma-formatted digits (`30,000`) that degrade TTS pronunciation.
  - When soft checks fail, `generate_script_step()` performs at most **one** targeted rewrite retry (`attempts <= 2`), passing the previous draft and specific quality issues to the LLM.

### 2. Separating Narrative Flow from Visual Scene Structure (`app/prompts/storyboard.md` & `app/steps/storyboard.py`)

- **Scenes as Visual Moments Over One Narrative**: The storyboard step treats the script as a single continuous spoken narrative and places 8–12 visual camera cuts (`scenes`) along that stream (`2.0–7.0s` / `5–17 words` per scene) while reproducing the script narration **verbatim**.
- **Cross-Scene Sentence Flow**: Scenes are not forced to start with a mini-introduction or end with a standalone thought. Multi-clause sentences (>17 words or covering two visual beats such as setup -> payoff) flow naturally across consecutive scenes at clean clause/phrase boundaries (`, and`, `, but`, `, while`, `, because`, participial clauses) without ever splitting mid-noun-phrase, mid-verb-phrase, or on dangling prepositions/conjunctions/articles.

---

## Canonical Continuous Narration Architecture

Shorts Studio generates **one continuous narration audio file per video** rather than separate per-scene audio clips. Continuous narration is the **only** supported audio synthesis, rendering, and subtitle-alignment method.

### 1. Voiceover Step, Balanced Chunking & Kokoro Prosody (`app/steps/voiceover.py`, `app/voiceover/continuous.py`, `app/voiceover/kokoro.py`)

- **Script Collection**: `collect_narration_script(scenes)` joins all storyboard scene narrations in scene order (`index` 1..N) into a single continuous script while preserving sentence punctuation and scene associations.
- **Balanced Sentence Chunking (`split_script_into_chunks`)**:
  - Splits long scripts only at paragraph or sentence boundaries (default `max_words=65, max_chars=380`), keeping every chunk safely below Kokoro-82M's 510-phoneme window (`MAX_PHONEME_LENGTH = 510`, ~1 ONNX inference call per chunk).
  - Uses dynamic-programming balanced partitioning (`_partition_units_balanced`) so multi-chunk scripts divide into roughly equal sentence groups (e.g., `[62, 60]` or `[45, 44, 46]` words) instead of leaving short trailing orphan chunks (`<15 words`), which would otherwise shift Kokoro's length-indexed voice style vector (`voice[len(tokens)]`) and alter vocal timbre.
- **TTS Text Normalization (`normalize_text_for_tts`)**:
  - Normalizes text strictly for acoustic synthesis (without altering display or subtitle text):
    - Strips commas inside numbers (`30,000` -> `30000` so Kokoro says `"thirty thousand"` instead of `"thirty, zero zero zero"`).
    - Expands decimal points (`2.5` -> `2 point 5`) so periods inside numbers do not trigger sentence breaks.
    - Strips inline double quotes (`“...”`, `"..."`) which cause a ~240ms prosodic stall in Kokoro's tokenizer.
    - Normalizes em-dashes, en-dashes, ellipses, colons, and semicolons into natural comma/period pauses, and expands common abbreviations (`vs.`, `Dr.`, `mph`, `km/h`, `°F`, `°C`, `%`).
- **Preserving Natural Silence & Inter-Chunk Breath Pauses**:
  - `KokoroVoiceoverProvider._execute_generate` invokes `kokoro.create(..., trim=False)` so `kokoro_onnx` does not strip all leading/trailing silence via `librosa.effects.trim` before `trim_silence_samples()` runs.
  - `trim_silence_samples()` preserves `35ms` leading breath headroom and `110ms` (terminal `.!?`) / `80ms` (non-terminal) trailing decay with 5ms anti-click micro-fades.
  - When joining multiple chunks, `compute_inter_chunk_gaps()` and `concatenate_audio_segments()` insert a `240ms` sentence breath pause (`90ms` for clause boundaries) with a 3ms boundary taper, yielding a ~`300ms–385ms` inter-chunk pause that matches Kokoro's natural intra-chunk sentence pauses (`360ms–480ms`).
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

## Phase 1D: Narrative Storytelling & Adaptive Duration Architecture

### 1. Adaptive Video Duration Configuration (`app/models.py`, `app/schemas.py`, `app/steps/duration.py`)

- **Video Model & Schema**:
  - `duration_mode`: `"auto"` (default) or `"manual"`.
  - `target_duration`: Integer seconds in `[30, 180]` (required when `duration_mode == "manual"`, `None` for `"auto"`).
  - Migration file: `alembic/versions/d4a8e21c3b70_add_duration_fields_to_videos.py`.
- **Duration Resolution Order (`resolve_video_target_duration`)**:
  1. **Manual User Configuration**: `video.duration_mode == "manual"` with `video.target_duration` always takes absolute precedence over AI recommendations.
  2. **Creative Direction Canonical Output**: `creative_direction.resolved_target_duration` or `creative_direction.recommended_duration_seconds` passed strictly from current pipeline execution state.
  3. **Strict Invalidation & Anti-Stale Protections**: Never inspects historical `video.steps` for previous step runs, ensuring reruns, rollbacks, and revisions cannot consume outdated duration state.
  4. **Fail-Closed Downstream Generation**: Script and Storyboard generation in 8-step videos require a valid resolved duration (`require_resolved=True`) and will raise `PipelineError` rather than silently defaulting to 60s.
  - Note: Legacy heuristics and implicit regex extraction from topic, description, or channel settings have been completely removed.
- **Dynamic Word Budget Calculation (`calculate_word_budget_for_duration`)**:
  - Dynamic speech pacing accounting for base speaking rate (~2.3 WPM net of pauses) and channel-configured TTS `speaking_speed`:
    - 30s: 60–85 words (at 1.0x speed)
    - 45s: 90–120 words
    - 60s: 115–155 words
    - 90s: 180–235 words
    - 120s: 240–310 words
    - 180s: 380–495 words

### 2. Story-Driven Creative Direction (`app/prompts/creative_direction.md` & `app/steps/creative_direction.py`)

- **Version**: `creative-direction-v5`.
- **Narrative Archetypes & Flexibility**: Story-driven structures replace formulaic templates. Narrative stages and pacing allocations serve as flexible guidance. Simple topics or direct explainers do not require artificial tension, manufactured conflict, or forced philosophical endings. Direct storytelling and simple explanations remain fully valid.
- **Removed Unconsumed Fields**: `emotional_progression` has been removed from `CreativeDirectionOutput` schema and prompts, eliminating unconsumed overhead.
- **Factual Integrity on Sensory Details**: When sensory scene-setting or hypothetical viewpoints are used, they must ground documented reality without introducing unsupported factual claims, fake events, or fabricated quotes.
- **Narrative Plan**: High-level flexible planning guidance (`narrative_perspective`, `opening_situation`, `information_revelation_strategy`, `story_progression`, `audience_takeaway`, `resolution_payoff`) without rigid stage count quotas.
- **Creative Constraints**: Made strictly optional (0 to 3); empty `creative_constraints = []` is fully valid when standard guidelines suffice.
- **Duration Recommendation**: Evaluates topic complexity to propose `recommended_duration_seconds` and `duration_rationale` (or respects manual duration in `resolved_target_duration`).

### 3. High-Retention Narration & Script Quality (`app/prompts/script.md` & `app/steps/script.py`)

- **Version**: `script-v3`.
- **Narrative Progression Across Scenes (Removal of Rules 17 & 66)**:
  - Eliminated micro-level sentence restrictions ("Each sentence must introduce new progression" and "Every sentence must advance the narrative").
  - Narrative progression is evaluated across the overall scene arc, explicitly encouraging atmospheric scene-setting, sensory grounding, and physical texture.
- **Channel Settings Filtering**:
  - `build_script_prompt()` filters out conflicting artistic instructions from `channel_settings` using `filter_conflicting_channel_settings()` to ensure raw 3D visual styles never leak into spoken narration.
- **Prompt Simplification**:
  - Eliminated triple-repeated prompt blocks in `build_script_prompt()` (`### Conversational Voice Requirement` and duplicate `## Instructions`).
  - Removed obsolete angle enum selection rules and recent-angles de-duplication from script prompts.
- **Schema Simplification (`ScriptOutput`)**:
  - Removed unconsumed `angle` field; `format_archetype` from Creative Direction governs angle.
  - Removed `hashtags` completely (zero product consumers).
  - Removed `model_config = ConfigDict(extra="ignore")` from `CreativeDirectionOutput` and `ScriptOutput`.
  - Made `claims_to_verify` optional metadata (`Field(default_factory=list)`) for auditing without failing validation or triggering retries.
- **Flexible Openings & Payoffs**:
  - Allows single punchy sentences or multi-sentence atmospheric cold opens without arbitrary 50-word ceilings.
  - Scene-setting phrases such as `"Imagine"` or `"Picture this"` are explicitly permitted for sensory grounding.
  - Removed mandate for an immediate scientific explanation in sentence 2; tension, mystery, and stakes are encouraged to breathe.
  - Endings can be simple, surprising, emotional, historical, or intellectual depending on the topic.
- **Streamlined Code Validators (`evaluate_script_quality` & `validate_script_soft_checks`)**:
  - Removed unreliable heuristics and arbitrary thresholds:
    - Syllable-based jargon detector (`_count_syllables() >= 4` and `EVERYDAY_MULTISYLLABLE_ALLOWLIST`).
    - Inter-sentence transition cohesion check (`SENTENCE_COHESION_MARKERS` and stem overlap).
    - Catalog opener detector based on `"Additionally"` / `"Furthermore"` / `"also"`.
    - Second-sentence filler (`DELAYED_PAYOFF_REGEX`) and deflated question regex (`DEFLATED_HOOK_REVEAL_REGEX`).
    - Fixed 28-word sentence length and 22.5-word average sentence thresholds.
    - Fixed 50-word hook ceiling.
    - Repetitive leading sentence subject check.
    - Subjective weak-ending check (`WEAK_ENDING_REGEX`).
  - Retained core safeguards:
    - Production-note / backstage leakage check (`META_INSTRUCTION_LEAK_REGEX`).
    - Genuine accidental duplication check (preserves intentional narrative parallelism / rhetorical emphasis).
    - Epistemic calibration (dogmatic overclaiming check) and redundant meta-summary check.
    - Hook match check (`narration` must begin verbatim with `hook`).
- **TTS Normalization & Formatting Safety**:
  - Verified Kokoro synthesis, text normalization (`normalize_text_for_tts`), balanced chunking, and Vosk acoustic alignment.
  - Ordinary spoken punctuation (colons, semicolons, dashes, ellipses, quotes, comma-formatted digits) is safely normalized into natural pauses and spoken numbers by the voice pipeline.
  - Full script regeneration retry is NEVER triggered for safely normalizable punctuation.
  - Objective technical failures are strictly reserved for unrenderable content: missing narration, bracketed directions/parentheses `[...]`/`(...)`, markdown formatting (`*`, `#`, `` ` ``), emojis, internal production instruction leakage, and significant duration/word deviation (> 25% off bounds).
- **Adaptive Duration Consistency & Primary Measure**:
  - `estimate_spoken_duration_seconds()` is the primary duration quality measure, with word count as supporting guidance.
  - Factors `voice_speed` into word rate, syllable rate, and sentence pause allowances consistently across duration budgeting and estimation.
  - Duration is treated as an approximate storytelling budget; minor deviations remain advisory warnings.
- **Objective Retry Policy & Diagnostic Isolation**:
  - Automatic retry (`attempts = 2`) is gated strictly on objective technical failures; subjective warnings never trigger retries.
  - When retry triggers, Attempt 1 draft, triggering reasons, and full provider telemetry are stored in `attempts_history`.
  - Attempt 2 is strictly re-validated; if Attempt 2 fails objective technical validation, it raises `LLMError` so invalid drafts can never silently become authoritative narration or enter downstream generation.

### 4. Downstream Adaptive Scaling (`app/steps/storyboard.py`, `app/steps/render.py`, `app/steps/subtitles.py`, `app/subtitles/acoustic.py`)

- **Adaptive Scene Scaling**:
  - Replaced hardcoded 8–12 scene limit with dynamic range without arbitrary caps:
    `min_scenes = max(5, int(math.floor(dur / 7.5)))`
    `max_scenes = max(min_scenes + 3, int(math.ceil(dur / 4.0)))`
    (Maintains 8 to 13 scenes for standard 60s videos, while scaling naturally from 5 scenes at 30s to 24–45 scenes at 180s).
- **Duration-Aware Timeouts**:
  - Video rendering (`app/steps/render.py`): `max(300.0, continuous_duration * 3.0 + 60.0)` seconds.
  - Subtitle burning (`app/steps/subtitles.py`): `max(300.0, continuous_duration * 2.0 + 60.0)` seconds.
  - Acoustic alignment (`app/subtitles/acoustic.py`): timeout scaled to 120.0s for long continuous audio.

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
4. **Phase 1D Video Creation & Duration Controls**:
   - Add duration controls to `CreateVideoDialog`:
     - Mode Selector: `Automatic` (default) vs `Custom Duration`.
     - Target Duration Picker (active when mode is `Custom`): Preset pills (`30s`, `45s`, `60s`, `90s`, `120s`, `180s`) or slider/stepper within `[30, 180]`.
     - Request payload: `{ topic, visual_style_id, duration_mode: "auto" | "manual", target_duration: number | null }`.
5. **Phase 1D.7 Schema Alignment**:
   - In Creative Direction interfaces (`types/video.ts`):
     - Remove `emotional_progression` (no longer emitted by backend).
     - Ensure `creative_constraints` accepts empty array `[]`.
     - Render `narrative_plan` with high-level fields (`narrative_perspective`, `opening_situation`, `information_revelation_strategy`, `story_progression`, `audience_takeaway`, `resolution_payoff`).
   - In Script interfaces:
     - Remove `angle` and `hashtags` (no longer emitted by backend).
     - Treat `claims_to_verify` as optional array.

