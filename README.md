# Shorts Studio

A backend for automating short-form video creation (YouTube Shorts / Instagram Reels).

A user registers, adds channels, and creates videos inside a channel by giving a topic and description. Each video moves through a pipeline of steps: **script → storyboard → images → voiceover → render**. During development you run the pipeline one step at a time by pressing "next", so you don't burn AI credits. Any step can be redone with a revision prompt.

> **Status:** backend only. The step handlers are mocks (no real AI calls yet). The only UI is the auto-generated API page at `/docs`.

## Tech stack

- **FastAPI**: the API
- **PostgreSQL 16** (in Docker): the database
- **SQLAlchemy 2.0**: database models
- **JWT + argon2**: login and password hashing

## Project structure

```
shorts-studio/
├── docker-compose.yml     # Postgres database
├── requirements.txt       # Python packages
├── .env                   # your local settings (not committed)
├── .env.example           # template for .env
└── app/
    ├── main.py            # app entry point, creates tables on startup
    ├── config.py          # reads settings from .env
    ├── db.py              # database connection and session
    ├── models.py          # tables: users, channels, videos, video_steps
    ├── schemas.py         # request/response shapes
    ├── security.py        # password hashing, JWT tokens
    ├── deps.py            # login check, ownership checks
    ├── pipeline.py        # step engine + mock handlers (plug real AI in here)
    └── routers/
        ├── auth.py        # register, login, me
        ├── channels.py    # channel CRUD
        └── videos.py      # videos + step controls
```

## Prerequisites

- Python 3.10+
- Docker Desktop (must be running)

## First-time setup (do once)

Run all commands from the project root (the folder containing `docker-compose.yml`).

```bash
# 1. Start the database
docker compose up -d

# 2. Create the virtual environment
python -m venv .venv

# 3. Activate it (see "Daily startup" below for your OS)

# 4. Install packages
pip install -r requirements.txt

# 5. Create your settings file
#    Windows:      copy .env.example .env
#    Mac/Linux:    cp .env.example .env
#    Then edit .env and set SECRET_KEY to a long random value:
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

## Daily startup

**1. Start the database** (skip if Docker already shows it running):

```bash
docker compose start
```

**2. Activate the virtual environment:**

| OS                 | Command                      |
| ------------------ | ---------------------------- |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows cmd        | `.venv\Scripts\activate.bat` |
| Mac / Linux        | `source .venv/bin/activate`  |

**3. Run the API:**

```bash
uvicorn app.main:app --reload
```

**4. Open the docs:** http://localhost:8000/docs

## Daily shutdown

```bash
# Stop the API: press Ctrl+C in its terminal, then:
docker compose stop
```

## Using the API

In `/docs`:

1. `POST /auth/register`, then click **Authorize** (username = your email) to log in
2. `POST /channels` to add a channel
3. `POST /channels/{id}/videos` with a topic and description
4. `POST /videos/{id}/start` creates the 5 pending steps (nothing runs yet)
5. `POST /videos/{id}/next` runs **one** step. Check progress with `GET /videos/{id}`. Repeat until `completed`.
6. `POST /videos/{id}/run-all` runs all remaining steps at once
7. `POST /videos/{id}/steps/{name}/rerun` with `{"prompt": "..."}` redoes a step; later steps become `stale`
8. `GET /videos/{id}/download` gets the final output

Step statuses: `pending`, `running`, `done`, `failed`, `stale`.

## Useful commands

```bash
# Check the database container
docker compose ps

# List tables
docker compose exec db psql -U studio -d studio -c "\dt"

# Query data (example)
docker compose exec db psql -U studio -d studio -c "select email from users;"

# View database logs
docker compose logs db

# WIPE the database (deletes all data) and start fresh
docker compose down -v
docker compose up -d
```

Use the wipe command whenever you change table structures during early development. Tables are created automatically on startup, but existing tables are never altered. Switch to Alembic migrations before you store data you care about.

## Troubleshooting

| Problem                           | Fix                                                                                                            |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| Can't connect to Docker daemon    | Open Docker Desktop and wait until it has started                                                              |
| Port 5432 already in use          | Change `5432:5432` to `5433:5432` in `docker-compose.yml` and use `localhost:5433` in `DATABASE_URL` in `.env` |
| `uvicorn: command not found`      | The virtual environment isn't activated                                                                        |
| `No module named 'app'`           | You're in the wrong folder; `cd` to the project root                                                           |
| Connection refused on startup     | The database is still starting; wait a few seconds and check `docker compose ps`                               |
| PowerShell blocks venv activation | Run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once                                                 |

## LLM Providers

Shorts Studio includes a small, swappable LLM layer in `app/llm/` supporting **Mock** (default, zero cost), **Google Gemini**, and **OpenAI**. You can switch providers globally or configure task-specific overrides purely using `.env`.

### 1. Global Configuration

In your `.env` file, choose a global provider and optional model:

```bash
# Provider choices: mock | gemini | openai
LLM_PROVIDER=mock
LLM_TIMEOUT_SECONDS=30.0
LLM_MAX_OUTPUT_TOKENS=4096
# Note: LLM_TEMPERATURE affects Gemini only (OpenAI provider omits temperature)
LLM_TEMPERATURE=0.7

# Gemini settings
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
GEMINI_THINKING_LEVEL=minimal  # minimal | low | medium | high

# OpenAI settings
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-6-luna
# Reasoning effort for OpenAI reasoning models.
# For gpt-6-luna, supported values are: none, low, medium, high, xhigh ("minimal" is rejected by the API).
# Recommended starting value: low
OPENAI_REASONING_EFFORT=low
```

### 2. Per-Task Overrides

Route specific pipeline tasks (`script`, `storyboard`, `metadata`, `factcheck`) to different providers or models without changing code:

```bash
# Example: Keep everything on mock, but run script generation on OpenAI:
LLM_PROVIDER_SCRIPT=openai
LLM_MODEL_SCRIPT=gpt-4o

# Example: Run storyboard scene generation on Gemini:
LLM_PROVIDER_STORYBOARD=gemini
LLM_MODEL_STORYBOARD=gemini-2.5-flash
```

If an override is omitted, it automatically falls back to `LLM_PROVIDER`.

### 3. Testing and Inspecting Routing

Use `scripts/try_llm.py` to inspect active routes and test models:

```bash
# 1. View active routing table (Task -> Provider -> Model)
python scripts/try_llm.py --show-routing

# 2. Test Mock provider (no API key needed)
python scripts/try_llm.py --provider mock --prompt "Write a short hook about space"

# 3. Test Gemini provider
python scripts/try_llm.py --provider gemini --prompt "Write a short hook about space"

# 4. Test OpenAI provider
python scripts/try_llm.py --provider openai --prompt "Write a short hook about space"

# 5. Test task routing (uses .env overrides or defaults)
python scripts/try_llm.py --task script --prompt "Hook on black holes"

# 6. Test structured output validation
python scripts/try_llm.py --provider mock --structured
```

### 4. Running Unit Tests

Run the full test suite (requires no network or API keys):

```bash
pytest tests/ -v
```

## Script Step

The `script` step in Shorts Studio is powered by the LLM layer (`app/steps/script.py` called from `app/pipeline.py`).

### How It Works

1. **Context & Recent Angles**: When executing for a video, it looks up the video topic, description, channel niche, and queries the latest 8 completed angles from the channel's previous scripts to avoid repetition.
2. **Writing Rules & Versioning**: Loads writing rules from [app/prompts/script.md](file:///Users/jenilmangukiya/projects/shorts-studio/app/prompts/script.md). It extracts the version tag (e.g., `script-v1`) and computes a short content hash (e.g. `script-v1:3e8c5809`) stored in execution metadata.
3. **Structured Output**: Uses `get_llm(task="script")` with the `ScriptOutput` Pydantic model (`title`, `hook`, `narration`, `angle`, `claims_to_verify`, `hashtags`).
4. **Code-Side Soft Checks & Retry**: Validates 110-150 words narration, <=15 words hook, hook-at-start, no markdown/brackets/emojis, 1-6 claims, 3-5 tags. In real provider modes, retries once with feedback if soft checks fail; warnings are attached to the output without blocking the pipeline.
5. **Revision Mode**: When `step.user_prompt` is provided (e.g. via `POST /videos/{id}/steps/script/rerun`), it feeds the previous script from `step.output["script"]` and asks the model to modify only what was requested.

### Editing Writing Rules

To tune narration style, pacing, banned openers, or audience tone, edit [app/prompts/script.md](file:///Users/jenilmangukiya/projects/shorts-studio/app/prompts/script.md) directly. Changes take effect on the next script generation without restarting the app or touching Python code.

### Testing Without Database: `try_script.py`

Use `scripts/try_script.py` to test prompts and inspect outputs locally without touching the database:

```bash
# 1. Test script generation with mock provider
python scripts/try_script.py --topic "Why airplane windows have tiny holes" --provider mock

# 2. Test with active LLM provider (routed via .env or CLI override)
python scripts/try_script.py --topic "Why airplane windows have tiny holes"

# 3. Test revision mode
python scripts/try_script.py --topic "Zipper teeth" --revise "Make the opening hook punchier" --prev-script path/to/script.json

# 4. View raw JSON output
python scripts/try_script.py --topic "How undersea cables work" --provider mock --json
```

## Storyboard Step

The `storyboard` step in Shorts Studio converts the generated script into 8–12 consecutive visual scenes with framing instructions, estimated speech timing, a recurring subject anchor, and code-assembled final image prompts (`app/steps/storyboard.py` called from `app/pipeline.py`).

### How It Works

1. **Input Script & Preconditions**: Reads the script output from `prev["script"]["script"]`. If the script step is not completed, it raises a clean `PipelineError`.
2. **Visual Style Resolution**: Checks the channel's `settings["visual_style"]`. If set to a non-empty string, it uses that style and tags `style_source: "channel"`. Otherwise, it falls back to `app/prompts/default_visual_style.txt` and tags `style_source: "default"`. Note: visual style owns only the look and aesthetic (e.g. materials, lighting, depth of field); framing and aspect ratio are owned strictly by code.
3. **Structured Scene Splitting & Subject Anchor**: Uses `get_llm(task="storyboard")` with `StoryboardLLMOutput`:
   - `subject_anchor`: 1–2 plain sentences describing the video's main recurring hero object(s) concretely (shape, colors, materials, size cues; no style words, text, logos, or real people).
   - Per-scene `uses_anchor`: boolean flag marked `true` only if that scene's image depicts the anchor object. In scenes where `uses_anchor` is `true`, the model's `image_prompt` describes only the action, state, angle, and surroundings to avoid contradictions with the anchor.
   - Each scene also contains `narration`, `image_prompt`, and `shot_type` (from `close-up`, `macro`, `wide`, `top-down`, `cutaway`, `cross-section`, `in-context`).
4. **Code-Side Final Prompt Assembly (`app/steps/prompt_compose.py`)**:
   - The code defines a single prompt composition function `compose_final_prompt(scene, subject_anchor, visual_style)`.
   - Assembly order:
     1. Scene `image_prompt`
     2. `subject_anchor` (only included if `uses_anchor` is `true` and anchor is non-empty)
     3. Resolved `visual_style`
     4. Fixed `COMPOSITION_RULES`: `"Vertical 9:16, main subject in the upper-middle of the frame, calm uncluttered lower third, no text, no letters, no logos, no watermark."`
   - Joins cleanly with single spaces and no duplicate punctuation. The code writes `final_prompt`; the model never does.
   - **Convenience Snapshot vs Image Generation**: The `final_prompt` written into storyboard storage is a convenience snapshot. The downstream `images` step must call `compose_final_prompt` (or `recompute_final_prompts`) itself at generation time rather than blindly trusting the stored value.
5. **Code-Side Calculations**: The code computes 1-based scene `index`, `est_seconds` (`words / 2.5`), total video duration `total_seconds`, and assembles `final_prompt` for each scene.
6. **Code-Side Soft Checks & Retry**: Validates that joined scene narrations reproduce the script narration verbatim, enforces 8–12 scenes, 2–7s scene durations, max 300 character `image_prompt`s with whole-word checks against forbidden words (`text`, `logo`, `caption`, etc.), subject anchor validity (<=250 chars, <=2 sentences, forbidden words check), at least one scene with `uses_anchor=True`, and each `final_prompt` non-empty and within `MAX_FINAL_PROMPT_LENGTH` (1000 chars in `app/steps/prompt_compose.py`). Retries once with feedback on non-mock providers.
7. **Deterministic Mock Mode**: When `LLM_PROVIDER_STORYBOARD=mock` (or default `mock`), it uses a deterministic local splitter that divides the script along sentence/clause boundaries with rotating shot types, clean placeholder prompts, a deterministic placeholder anchor (`"A small glossy blue cube with rounded edges."`), and sets `uses_anchor` false for scenes 2 and 5 (true for all others).
8. **Revision Mode**: When `step.user_prompt` is provided (e.g., via `POST /videos/{id}/steps/storyboard/rerun`), it feeds the previous storyboard JSON into the prompt, instructing the model to keep `subject_anchor` and `uses_anchor` flags unchanged unless explicitly instructed to change them, while code recomputes all `final_prompt` values and validates verbatim narration.

### Setting Channel Visual Style

You can configure a custom visual style per channel via the API:

```bash
PATCH /channels/{id}
Content-Type: application/json

{
  "settings": {
    "visual_style": "Vintage claymation, warm tungsten lighting, tactile plasticine surfaces, vertical 9:16 framing"
  }
}
```

### Editing Storyboard Rules

To adjust scene splitting rules, framing guidance, or composition constraints, edit [app/prompts/storyboard.md](file:///Users/jenilmangukiya/projects/Short-studio/shorts-studio-backend/app/prompts/storyboard.md) or edit the default fallback style in [app/prompts/default_visual_style.txt](file:///Users/jenilmangukiya/projects/Short-studio/shorts-studio-backend/app/prompts/default_visual_style.txt). Changes take effect on the next run with a new content hash automatically recorded in `meta["prompt_version"]`.

### Testing Without Database: `try_storyboard.py`

Use `scripts/try_storyboard.py` to test prompts and inspect scenes locally:

```bash
# 1. Test storyboard generation with mock provider
python scripts/try_storyboard.py --provider mock

# 2. Test with custom script text or file
python scripts/try_storyboard.py --script "Airplane windows have a hole. It balances pressure." --provider mock
python scripts/try_storyboard.py --script-file path/to/script.json --provider mock

# 3. Test with visual style override
python scripts/try_storyboard.py --provider mock --visual-style "Cyberpunk isometric 3D"

# 4. Test revision mode
python scripts/try_storyboard.py --provider mock --revise "Change scene 1 framing to top-down" --prev-storyboard path/to/storyboard.json

# 5. Output raw JSON
python scripts/try_storyboard.py --provider mock --json
```

## Images Step

The `images` step in Shorts Studio generates vertical images for each scene in the storyboard sequentially, utilizing code-side prompt composition, content-addressable caching/reuse, partial progress persistence, and granular error handling (`app/steps/images.py` called from `app/pipeline.py`).

### How It Works

1. **Preconditions & Guards**:
   - Reads scenes from `prev["storyboard"]["storyboard"]` (or legacy `prev["storyboard"]`).
   - If the storyboard step is missing or contains 0 scenes, fails with a readable `PipelineError`.
   - If scene count exceeds `IMAGE_MAX_PER_VIDEO` (default: 12), fails **before** any generation to prevent runaway costs.
2. **Code-Side Prompt Composition**:
   - For each scene, invokes `compose_final_prompt(scene, subject_anchor, visual_style)` at generation time rather than blindly trusting stored storyboard snapshots.
3. **Execution Time**:
   - In real generation with OpenAI, generating 8–12 images sequentially takes several minutes. The step runs sequentially in order so you can monitor progress.
4. **Content-Addressable Reuse (Never Pay Twice)**:
   - For every scene, computes a full 64-character SHA256 `prompt_hash` over `prompt|model|size|quality`.
   - The first 8 characters form the file suffix: `scene_<NN>_<hash8>.png`.
   - If previous step output (`step.output`) has an item for the same scene with a matching `prompt_hash`, the file still exists on disk, and `status == "ok"`, generation is skipped (`reused=True`) with zero API calls.
   - Rerunning a failed or partially completed step only generates missing or modified scenes.
   - **Disk Files**: Outdated image files with older hashes remain safely on disk until a future cleanup routine.
5. **Partial Progress Persistence**:
   - After each image is generated or reused, partial progress is saved to `step.output` (`complete=False`) and committed to the database. `step.status` remains untouched while running.
   - If a later image fails or the server restarts, previously paid and generated images are preserved in the step output.
6. **Error Handling**:
   - **Configuration / Auth / Organization Verification**: Missing API keys, missing models, 401 unauthorized, or 403 organization-not-verified abort the pipeline immediately with a readable error.
   - **Rate Limits (429)**: Retries up to 2 times with exponential backoff (1s, 2s). If exhausted, aborts the run immediately, preserves partial progress in `step.output`, and raises an error instructing to rerun later to resume.
   - **Content-Policy Blocks**: Flagged prompts mark only that individual scene as failed (`status="failed"`, storing the error reason), and the run continues generating the remaining scenes. If any scenes failed, a `PipelineError` listing the failed scene numbers is raised at the end.
7. **Serving Route**:
   - `GET /videos/{video_id}/images/{index}` serves the generated image directly.
   - Requires login and video ownership.
   - Serves the file for any scene with `status == "ok"` regardless of whether the overall step is running, failed, or done.
   - Validates that the file path resolves strictly within `STORAGE_DIR` to prevent path traversal.
   - Includes `ETag: "<prompt_hash>"` and `Cache-Control: private, no-cache` to ensure regenerated images are never served stale.
8. **Instructions Warning**:
   - Step instructions (`step.user_prompt`) are ignored for this step; if provided, a warning is added: `"Instructions are not used by the images step. Edit the storyboard prompt instead."`

### Configuration Settings

| Setting | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `IMAGE_PROVIDER` | string | `mock` | Active provider: `mock` or `openai`. |
| `OPENAI_IMAGE_API_KEY` | string | `None` | OpenAI API key for image generation. Isolated from and **never** falls back to `OPENAI_API_KEY`. |
| `OPENAI_IMAGE_MODEL` | string | `None` | Model name copied directly from your OpenAI dashboard project's model list (e.g. `gpt-image-2`). No default is hardcoded in code. |
| `OPENAI_IMAGE_SIZE` | string | `1024x1536` | Preset portrait dimensions (2:3). For custom 9:16 resolutions, width & height must be divisible by 16 (e.g. `864x1536`). |
| `OPENAI_IMAGE_QUALITY` | string | `low` | Generation quality (`low`, `medium`, `high`, `xhigh`, `max`). |
| `OPENAI_IMAGE_EST_COST_USD` | float | `None` | Optional per-image cost estimate in USD. Prices are never hard-coded in code. |
| `IMAGE_MAX_PER_VIDEO` | int | `12` | Maximum scenes allowed per video. |
| `IMAGE_TIMEOUT_SECONDS` | float | `120.0` | Timeout per image generation call in seconds. |

### Storage Layout

Generated images are saved under `STORAGE_DIR`:
```
storage/
└── <video_id>/
    └── images/
        ├── scene_01_a1b2c3d4.png
        ├── scene_02_e5f6g7h8.png
        └── ...
```
All stored file references in the database are stored relative to `STORAGE_DIR` (e.g. `<video_id>/images/scene_01_a1b2c3d4.png`).

### Testing Without Database: `try_image.py`

Use `scripts/try_image.py` to test image generation locally with zero database dependencies:

```bash
# 1. Generate single image with mock provider (solid color PNG, no AI cost)
python scripts/try_image.py --prompt "Airplane window with condensation"

# 2. Extract and compose a scene from an exported storyboard JSON file
python scripts/try_image.py --storyboard-file path/to/storyboard.json --scene 1

# 3. Test with custom size and quality
python scripts/try_image.py --prompt "Macro close up of airplane rivets" --size 864x1536 --quality low

# 4. Generate with OpenAI (makes exactly ONE real API call and warns before starting)
python scripts/try_image.py --prompt "Airplane window at sunset" --provider openai
```

## Roadmap

- [x] Make the `script` step real (LLM) with structured output
- [x] Make the `storyboard` step real (scenes with timing and image prompts)
- [x] Make the `images` step real (provider layer, reuse caching, OpenAI + mock)
- [ ] Mock/real switch per step plus a daily AI-call cap
- [ ] Per-step approval (`awaiting_approval` status) and version history
- [ ] Real voiceover, captions, music, and FFmpeg render
- [ ] Alembic migrations
- [ ] Background worker queue (RQ or Dramatiq) instead of in-process tasks
- [ ] Frontend (Next.js or Streamlit)
- [ ] Publishing to YouTube and Instagram



