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

## Roadmap

- [x] Make the `script` step real (LLM) with structured output
- [ ] Make the `storyboard` step real (scenes with timing and image prompts)
- [ ] Mock/real switch per step plus a daily AI-call cap
- [ ] Per-step approval (`awaiting_approval` status) and version history
- [ ] Real images, voiceover, captions, music, and FFmpeg render
- [ ] Alembic migrations
- [ ] Background worker queue (RQ or Dramatiq) instead of in-process tasks
- [ ] Frontend (Next.js or Streamlit)
- [ ] Publishing to YouTube and Instagram

