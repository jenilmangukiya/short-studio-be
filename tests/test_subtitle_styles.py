"""Tests for subtitle style registry, ASS styling, and word-level highlight animation."""

from __future__ import annotations

import json
from pathlib import Path
import uuid
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.db import Base, get_db
from app.deps import get_owned_video
from app.main import app
from app.models import Channel, User, Video, VideoStep
from app.pipeline import LEGACY_STEPS, STEPS, init_steps, rerun_step
from app.schemas import RerunBody, SubtitleStyleOut
from app.steps.subtitles import generate_subtitles_step
from app.subtitles.mock import MockSubtitleProvider
from app.voiceover.mock import create_mock_wav

from app.subtitles.styles import (
    CLEAN_CINEMATIC,
    DEFAULT_SUBTITLE_STYLE_ID,
    DYNAMIC_HIGHLIGHT,
    REMOTION_NEON,
    SUBTITLE_STYLES,
    SubtitleStyle,
    get_subtitle_style,
    list_subtitle_styles,
)
from app.subtitles.types import (
    SubtitleResult,
    SubtitleSegment,
    WordTimestamp,
)


def _make_sample_result() -> SubtitleResult:
    """Helper to build a realistic SubtitleResult for testing ASS output."""
    seg = SubtitleSegment(
        index=1,
        text="Why do airplanes fly so smoothly",
        start=1.0,
        end=3.4,
        words=[
            WordTimestamp(text="Why", start=1.0, end=1.35),
            WordTimestamp(text="do", start=1.36, end=1.55),
            WordTimestamp(text="airplanes", start=1.56, end=2.15),
            WordTimestamp(text="fly", start=2.16, end=2.45),
            WordTimestamp(text="so", start=2.46, end=2.70),
            WordTimestamp(text="smoothly", start=2.71, end=3.35),
        ],
    )
    return SubtitleResult(
        segments=[seg],
        total_words=6,
        total_duration_seconds=3.5,
        provider="test",
        model="test",
    )


class TestSubtitleStyleRegistry:
    """Verify registry population, style lookup, and fallback behavior."""

    def test_registered_style_ids(self):
        """All three required styles must be present in the registry."""
        assert "clean_cinematic" in SUBTITLE_STYLES
        assert "remotion_neon" in SUBTITLE_STYLES
        assert "dynamic_highlight" in SUBTITLE_STYLES
        assert len(SUBTITLE_STYLES) == 3

    def test_default_style_id(self):
        """DEFAULT_SUBTITLE_STYLE_ID must point to clean_cinematic."""
        assert DEFAULT_SUBTITLE_STYLE_ID == "clean_cinematic"

    def test_get_subtitle_style_valid(self):
        """Retrieving known styles returns the exact registered instance."""
        assert get_subtitle_style("clean_cinematic") is CLEAN_CINEMATIC
        assert get_subtitle_style("remotion_neon") is REMOTION_NEON
        assert get_subtitle_style("dynamic_highlight") is DYNAMIC_HIGHLIGHT

    def test_get_subtitle_style_case_insensitive_and_trimmed(self):
        """Lookup handles whitespace and uppercase input gracefully."""
        assert get_subtitle_style("  REMOTION_NEON  ") is REMOTION_NEON
        assert get_subtitle_style("Clean_Cinematic") is CLEAN_CINEMATIC
        assert get_subtitle_style("DYNAMIC_HIGHLIGHT") is DYNAMIC_HIGHLIGHT

    def test_get_subtitle_style_unknown_fallback(self):
        """Unknown or invalid style IDs must safely fall back to clean_cinematic."""
        assert get_subtitle_style("unknown_style") is CLEAN_CINEMATIC
        assert get_subtitle_style("invalid") is CLEAN_CINEMATIC
        assert get_subtitle_style("") is CLEAN_CINEMATIC
        assert get_subtitle_style(None) is CLEAN_CINEMATIC

    def test_list_subtitle_styles(self):
        """list_subtitle_styles returns all registered styles."""
        styles = list_subtitle_styles()
        ids = [s.id for s in styles]
        assert ids == ["clean_cinematic", "remotion_neon", "dynamic_highlight"]


class TestAssStyleHeaderGeneration:
    """Verify ASS [V4+ Styles] header generation for each style."""

    def test_clean_cinematic_ass_header(self):
        """Clean cinematic generates modern translucent box header (BorderStyle: 3)."""
        header = CLEAN_CINEMATIC.build_ass_style_header("Default")
        expected = (
            "Style: Default,Arial,60,&H00FFFFFF,&H000000FF,&H5A000000,&H5A000000,"
            "-1,0,0,0,100,100,0,0,3,14,0,2,40,40,360,1"
        )
        assert header == expected

    def test_remotion_neon_ass_header(self):
        """Remotion neon specifies 58px font size, translucent box (BorderStyle: 3), and 340px vertical margin."""
        header = REMOTION_NEON.build_ass_style_header("Default")
        assert "Arial,58" in header
        assert ",3,14,0,2,40,40,340,1" in header
        assert "&H00FFFFFF" in header
        assert "&H5A000000" in header

    def test_dynamic_highlight_ass_header(self):
        """Dynamic highlight specifies 64px font size and translucent box (BorderStyle: 3)."""
        header = DYNAMIC_HIGHLIGHT.build_ass_style_header("Default")
        assert "Arial,64" in header
        assert ",3,14,0,2,40,40,360,1" in header
        assert "&H5A000000" in header


class TestToAssRendering:
    """Verify SubtitleResult.to_ass() behavior across all styles."""

    def test_to_ass_default_argument_is_clean_cinematic(self):
        """Calling to_ass() without arguments uses clean_cinematic and matches modern box preview."""
        res = _make_sample_result()
        ass_default = res.to_ass()
        ass_explicit = res.to_ass("clean_cinematic")

        assert ass_default == ass_explicit
        # Verify translucent box style line is generated
        assert (
            "Style: Default,Arial,60,&H00FFFFFF,&H000000FF,&H5A000000,&H5A000000,"
            "-1,0,0,0,100,100,0,0,3,14,0,2,40,40,360,1"
        ) in ass_default
        # Verify amber-300 highlight tag and scale punch
        assert r"{\c&H004DD3FC&\fscx108\fscy108}Why{\c&H00FFFFFF&\fscx100\fscy100}" in ass_default

    def test_to_ass_with_unknown_style_falls_back_safely(self):
        """Calling to_ass('non_existent') falls back safely to clean_cinematic."""
        res = _make_sample_result()
        ass_unknown = res.to_ass("non_existent")
        ass_default = res.to_ass("clean_cinematic")
        assert ass_unknown == ass_default

    def test_to_ass_remotion_neon_colors_and_scaling(self):
        """Remotion neon emits electric cyan (&H00FCF528& in BGR) and 106% scale punch."""
        res = _make_sample_result()
        ass_text = res.to_ass("remotion_neon")

        # 1. Font size 58 and margin 340 in header
        assert "Style: Default,Arial,58" in ass_text
        assert ",40,40,340,1" in ass_text

        # 2. Electric Cyan (#28F5FC in BGR order is &H00FCF528&)
        assert r"\c&H00FCF528&" in ass_text

        # 3. 106% scale tag on active word
        assert r"\fscx106\fscy106" in ass_text

        # 4. Correct reset to white and 100% scale
        assert r"{\c&H00FCF528&\fscx106\fscy106}Why{\c&H00FFFFFF&\fscx100\fscy100}" in ass_text

        # 5. Text case is preserved (not uppercase)
        assert "Why do airplanes" in ass_text
        assert "WHY DO AIRPLANES" not in ass_text

    def test_to_ass_dynamic_highlight_uppercase_and_yellow(self):
        """Dynamic highlight transforms text to UPPERCASE, uses bright yellow (&H0000E6FF& in BGR), and 110% scale."""
        res = _make_sample_result()
        ass_text = res.to_ass("dynamic_highlight")

        # 1. Bright yellow (#FFE600 in BGR order is &H0000E6FF&)
        assert r"\c&H0000E6FF&" in ass_text

        # 2. 110% scale tag on active word
        assert r"\fscx110\fscy110" in ass_text

        # 3. Text is fully uppercased
        assert r"{\c&H0000E6FF&\fscx110\fscy110}WHY{\c&H00FFFFFF&\fscx100\fscy100}" in ass_text
        assert "DO AIRPLANES FLY" in ass_text
        # Lowercase original words should not appear in dialogue lines
        dialogue_lines = [l for l in ass_text.splitlines() if l.startswith("Dialogue:")]
        for dl in dialogue_lines:
            assert "Why" not in dl
            assert "airplanes" not in dl

    def test_to_ass_style_resets_prevent_leaking(self):
        """Every active word tag must reset color to white and scale to 100% so formatting does not leak."""
        res = _make_sample_result()
        for style_id in ("clean_cinematic", "remotion_neon", "dynamic_highlight"):
            ass_text = res.to_ass(style_id)
            dialogue_lines = [l for l in ass_text.splitlines() if l.startswith("Dialogue:")]
            assert len(dialogue_lines) == 6  # 6 words in sample

            for dl in dialogue_lines:
                # Must contain the closing color reset
                assert r"{\c&H00FFFFFF&" in dl
                if style_id in ("remotion_neon", "dynamic_highlight"):
                    # Must contain the scale reset
                    assert r"\fscx100\fscy100}" in dl

    def test_to_ass_preserves_timing_and_grouping_across_styles(self):
        """Selecting different styles must produce identical timestamps and dialogue line count."""
        res = _make_sample_result()
        ass_clean = res.to_ass("clean_cinematic")
        ass_neon = res.to_ass("remotion_neon")
        ass_dyn = res.to_ass("dynamic_highlight")

        lines_clean = [l for l in ass_clean.splitlines() if l.startswith("Dialogue:")]
        lines_neon = [l for l in ass_neon.splitlines() if l.startswith("Dialogue:")]
        lines_dyn = [l for l in ass_dyn.splitlines() if l.startswith("Dialogue:")]

        assert len(lines_clean) == len(lines_neon) == len(lines_dyn)

        # Timestamps are fields 1 and 2 in CSV Dialogue: Layer, Start, End, ...
        for lc, ln, ld in zip(lines_clean, lines_neon, lines_dyn):
            parts_c = lc.split(",")
            parts_n = ln.split(",")
            parts_d = ld.split(",")
            # Start time
            assert parts_c[1] == parts_n[1] == parts_d[1]
            # End time
            assert parts_c[2] == parts_n[2] == parts_d[2]
            # Style name
            assert parts_c[3] == parts_n[3] == parts_d[3] == "Default"


# ==============================================================================
# Phase 2: API, Rerun, and Persistence Tests
# ==============================================================================

@pytest.fixture
def db_session():
    """Isolated in-memory SQLite database session for API/pipeline tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session


def _create_fixture_video(session: Session, topic: str = "Test Airplane Wings") -> tuple[Video, Channel]:
    """Helper to populate a complete User -> Channel -> Video hierarchy."""
    user = User(email=f"user_{uuid.uuid4().hex[:8]}@example.com", password_hash="hash")
    session.add(user)
    session.flush()

    channel = Channel(
        user_id=user.id,
        name=f"Channel {uuid.uuid4().hex[:6]}",
        platform="youtube",
        niche="aviation",
        settings={},
    )
    session.add(channel)
    session.flush()

    video = Video(
        channel_id=channel.id,
        topic=topic,
        status="in_progress",
        output_path="/tmp/final_video.mp4",
    )
    session.add(video)
    session.flush()

    init_steps(session, video, steps=LEGACY_STEPS)
    session.commit()
    return video, channel


class TestSubtitleStylesAPIEndpoint:
    """Test GET /videos/subtitle-styles endpoint and response schema."""

    def test_get_subtitle_styles_endpoint(self):
        """GET /videos/subtitle-styles returns registered styles in display order with default marker."""
        client = TestClient(app)
        resp = client.get("/videos/subtitle-styles")
        assert resp.status_code == 200
        data = resp.json()

        assert len(data) == 3
        ids = [item["id"] for item in data]
        assert ids == ["clean_cinematic", "remotion_neon", "dynamic_highlight"]

        clean = next(it for it in data if it["id"] == "clean_cinematic")
        assert clean["name"] == "Clean Cinematic"
        assert clean["is_default"] is True
        assert "description" in clean

        remotion = next(it for it in data if it["id"] == "remotion_neon")
        assert remotion["name"] == "Remotion Neon"
        assert remotion["is_default"] is False

        dynamic = next(it for it in data if it["id"] == "dynamic_highlight")
        assert dynamic["name"] == "Dynamic Highlight"
        assert dynamic["is_default"] is False


class TestSubtitleRerunValidation:
    """Test validation rules for subtitle_style in RerunBody."""

    def test_rerun_body_valid_styles(self):
        """Registered styles are accepted and normalized."""
        b1 = RerunBody(subtitle_style="clean_cinematic")
        assert b1.subtitle_style == "clean_cinematic"

        b2 = RerunBody(subtitle_style="REMOTION_NEON  ")
        assert b2.subtitle_style == "remotion_neon"

        b3 = RerunBody(subtitle_style="dynamic_highlight")
        assert b3.subtitle_style == "dynamic_highlight"

        b_none = RerunBody()
        assert b_none.subtitle_style is None

    def test_rerun_body_rejects_unknown_style(self):
        """Unrecognized style IDs must raise a validation error."""
        with pytest.raises(ValidationError) as exc:
            RerunBody(subtitle_style="unknown_fancy_style")
        assert "Unknown subtitle style 'unknown_fancy_style'" in str(exc.value)

    def test_rerun_endpoint_returns_422_on_invalid_style(self, db_session):
        """POST /videos/{id}/steps/subtitles/rerun returns HTTP 422 for unknown style."""
        video, _ = _create_fixture_video(db_session)
        app.dependency_overrides[get_db] = lambda: db_session
        app.dependency_overrides[get_owned_video] = lambda: video
        client = TestClient(app)
        try:
            resp = client.post(
                f"/videos/{video.id}/steps/subtitles/rerun",
                json={"subtitle_style": "not_a_real_style"},
            )
            assert resp.status_code == 422
            err_data = resp.json()
            assert "Unknown subtitle style" in str(err_data)
        finally:
            app.dependency_overrides.clear()


class TestSubtitleStylePersistenceAndExecution:
    """Test style persistence, rerun workflow, asset reuse, and legacy video compatibility."""

    def test_rerun_with_valid_style_updates_output_and_generates_correct_ass(self, tmp_path, monkeypatch, db_session):
        """Rerunning with a new style persists the selection and burns captioned video with new style."""
        monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
        video, _ = _create_fixture_video(db_session)

        # Mark all steps done
        for s in video.steps:
            s.status = "done"

        video_dir = tmp_path / str(video.id)
        video_dir.mkdir(parents=True)
        audio_dir = video_dir / "audio"
        audio_dir.mkdir(parents=True)

        raw_mp4 = video_dir / "final_video.mp4"
        raw_mp4.write_bytes(b"DUMMY_MP4_CONTENT")
        video.output_path = str(raw_mp4)

        cont_wav = audio_dir / "narration_test.wav"
        cont_wav.write_bytes(create_mock_wav(2.0))

        # Setup voiceover and render previous outputs
        vo_out = {
            "voiceover": {
                "audio_file": f"{video.id}/audio/narration_test.wav",
                "audio_hash": "testhash12345",
                "narration_script": "Why do airplanes fly smoothly?",
                "items": [
                    {"index": 1, "status": "ok", "narration": "Why do airplanes fly smoothly?", "duration_seconds": 2.0}
                ],
            }
        }
        render_out = {"video_file": f"{video.id}/final_video.mp4"}
        prev = {"voiceover": vo_out, "render": render_out}

        # 1. Rerun subtitle step with 'remotion_neon'
        rerun_step(db_session, video, "subtitles", prompt=None, subtitle_style="remotion_neon")

        sub_step = next(s for s in video.steps if s.name == "subtitles")
        assert sub_step.status == "pending"
        # Upstream steps are still 'done'
        assert all(s.status == "done" for s in video.steps if s.name != "subtitles")
        # Video output path is preserved
        assert video.output_path == str(raw_mp4)

        # 2. Execute subtitle step
        output = generate_subtitles_step(
            video, prev, step=sub_step, subtitle_provider=MockSubtitleProvider()
        )

        # 3. Verify style is persisted in canonical location
        assert output["subtitles"]["style"] == "remotion_neon"
        assert sub_step.output["subtitles"]["style"] == "remotion_neon"

        # 4. Verify ASS artifact generated with remotion_neon
        ass_path = video_dir / "subtitles.ass"
        assert ass_path.is_file()
        ass_content = ass_path.read_text(encoding="utf-8")
        assert "Style: Default,Arial,58" in ass_content
        assert r"\c&H00FCF528&" in ass_content  # Electric cyan

        # 5. Verify subtitles.json records style
        json_path = video_dir / "subtitles.json"
        assert json_path.is_file()
        stored_json = json.loads(json_path.read_text(encoding="utf-8"))
        assert stored_json["style"] == "remotion_neon"

    def test_rerun_omitted_style_preserves_existing_style(self, tmp_path, monkeypatch, db_session):
        """When subtitle_style is omitted in rerun, the previously selected style is preserved."""
        monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
        video, _ = _create_fixture_video(db_session)
        for s in video.steps:
            s.status = "done"

        sub_step = next(s for s in video.steps if s.name == "subtitles")
        sub_step.status = "done"
        sub_step.output = {
            "subtitles": {
                "style": "remotion_neon",
                "segments": [],
            }
        }
        db_session.commit()

        # Rerun without specifying subtitle_style
        rerun_step(db_session, video, "subtitles", prompt="adjust timings")

        # Style should be preserved in pending step
        assert sub_step.status == "pending"
        assert sub_step.output["subtitles"]["style"] == "remotion_neon"

    def test_legacy_video_without_style_resolves_to_clean_cinematic(self, db_session):
        """Legacy video records lacking style metadata resolve to clean_cinematic via API and router."""
        video, _ = _create_fixture_video(db_session)
        sub_step = next(s for s in video.steps if s.name == "subtitles")
        sub_step.status = "done"
        sub_step.output = {
            "subtitles": {
                "format": "word_level",
                "segments": [{"index": 1, "text": "Hello world", "start": 0.0, "end": 1.0, "words": []}],
                # Notice: No 'style' key present in legacy record
            }
        }
        db_session.commit()

        app.dependency_overrides[get_db] = lambda: db_session
        app.dependency_overrides[get_owned_video] = lambda: video
        client = TestClient(app)
        try:
            # 1. Check GET /videos/{id}/subtitles endpoint
            resp_sub = client.get(f"/videos/{video.id}/subtitles")
            assert resp_sub.status_code == 200
            data_sub = resp_sub.json()
            assert data_sub["style"] == "clean_cinematic"

            # 2. Check GET /videos/{id} endpoint (StepOut schema validation)
            resp_vid = client.get(f"/videos/{video.id}")
            assert resp_vid.status_code == 200
            data_vid = resp_vid.json()
            target_step = next(s for s in data_vid["steps"] if s["name"] == "subtitles")
            assert target_step["output"]["subtitles"]["style"] == "clean_cinematic"
        finally:
            app.dependency_overrides.clear()

    def test_video_without_selected_style_uses_clean_cinematic_even_with_channel_setting(
        self, tmp_path, monkeypatch, db_session
    ):
        """A video without a selected style always uses clean_cinematic, ignoring any channel setting."""
        monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
        video, channel = _create_fixture_video(db_session)

        # Set arbitrary subtitle settings on the channel
        channel.settings = {
            "default_subtitle_style": "remotion_neon",
            "subtitle_style": "dynamic_highlight",
        }
        db_session.commit()

        video_dir = tmp_path / str(video.id)
        video_dir.mkdir(parents=True)
        audio_dir = video_dir / "audio"
        audio_dir.mkdir(parents=True)

        raw_mp4 = video_dir / "final_video.mp4"
        raw_mp4.write_bytes(b"DUMMY_MP4_CONTENT")
        video.output_path = str(raw_mp4)

        cont_wav = audio_dir / "narration_test.wav"
        cont_wav.write_bytes(create_mock_wav(2.0))

        vo_out = {
            "voiceover": {
                "audio_file": f"{video.id}/audio/narration_test.wav",
                "audio_hash": "testhash12345",
                "narration_script": "Why do airplanes fly smoothly?",
                "items": [
                    {"index": 1, "status": "ok", "narration": "Why do airplanes fly smoothly?", "duration_seconds": 2.0}
                ],
            }
        }
        render_out = {"video_file": f"{video.id}/final_video.mp4"}
        prev = {"voiceover": vo_out, "render": render_out}

        sub_step = next(s for s in video.steps if s.name == "subtitles")
        # Step has no output and no previously selected style
        sub_step.output = None

        # Execute subtitle step
        output = generate_subtitles_step(
            video, prev, step=sub_step, subtitle_provider=MockSubtitleProvider()
        )

        # Must resolve strictly to clean_cinematic
        assert output["subtitles"]["style"] == "clean_cinematic"
        assert sub_step.output["subtitles"]["style"] == "clean_cinematic"

        # Verify ASS artifact used clean_cinematic (Arial 60, margin 360, box BorderStyle: 3)
        ass_path = video_dir / "subtitles.ass"
        assert ass_path.is_file()
        ass_content = ass_path.read_text(encoding="utf-8")
        assert "Style: Default,Arial,60" in ass_content
        assert ",40,40,360,1" in ass_content
        assert "Arial,58" not in ass_content

