"""Tests for Alembic database migrations:
- Verifies Alembic configuration and target_metadata registration.
- Verifies current production/dev database has zero unmigrated schema differences (alembic check).
- Verifies a fresh empty database can be initialized from scratch via upgrade head,
  and downgraded cleanly via downgrade base, preserving partial unique indexes and constraints.
"""
import uuid
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect, text

from app.db import Base, SessionLocal, engine
import app.models  # noqa: F401


@pytest.fixture(scope="module")
def check_postgres():
    """Ensure PostgreSQL is accessible before running PostgreSQL migration integration tests."""
    if engine.dialect.name != "postgresql":
        pytest.skip("PostgreSQL dialect required for migration integration tests")
    try:
        with SessionLocal() as db:
            db.execute(text("SELECT 1"))
    except Exception as exc:
        pytest.skip(f"PostgreSQL container not reachable: {exc}")


def test_alembic_config_and_metadata():
    """Verify that alembic.ini is present and target_metadata includes all application tables."""
    cfg = Config("alembic.ini")
    assert cfg.get_main_option("script_location") is not None

    table_names = set(Base.metadata.tables.keys())
    expected = {
        "users",
        "channels",
        "videos",
        "video_steps",
        "video_step_revisions",
        "pipeline_jobs",
    }
    assert expected.issubset(table_names), f"Missing tables in Base.metadata: {expected - table_names}"


def test_alembic_check_current_database(check_postgres):
    """Verify that the active database matches the Alembic migration history with no pending operations."""
    cfg = Config("alembic.ini")
    # command.check raises CommandError if unapplied migrations or schema drift exist
    command.check(cfg)


def test_alembic_fresh_database_lifecycle(check_postgres):
    """Verify a fresh empty database can be migrated to head and downgraded to base without errors."""
    test_db_name = f"test_mig_{uuid.uuid4().hex[:8]}"

    # Create temporary database
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"CREATE DATABASE {test_db_name}"))

    test_url = f"postgresql+psycopg://studio:studio@localhost:5432/{test_db_name}"
    test_eng = None

    try:
        cfg = Config("alembic.ini")
        cfg.set_main_option("sqlalchemy.url", test_url)

        # 1. Run upgrade head on fresh database
        command.upgrade(cfg, "head")

        test_eng = create_engine(test_url)
        insp = inspect(test_eng)
        tables = set(insp.get_table_names())
        expected_tables = {
            "users",
            "channels",
            "videos",
            "video_steps",
            "video_step_revisions",
            "pipeline_jobs",
            "alembic_version",
        }
        assert expected_tables.issubset(tables), f"Missing migrated tables: {expected_tables - tables}"

        # Verify partial unique index on pipeline_jobs
        indexes = insp.get_indexes("pipeline_jobs")
        active_idx = next((i for i in indexes if i["name"] == "uq_pipeline_jobs_active_step"), None)
        assert active_idx is not None, "Partial unique index uq_pipeline_jobs_active_step not found"
        assert active_idx["unique"] is True

        # Verify foreign keys
        fks = insp.get_foreign_keys("pipeline_jobs")
        assert any(fk["referred_table"] == "videos" for fk in fks)

        # 2. Downgrade to base
        command.downgrade(cfg, "base")
        test_eng.dispose()
        test_eng = create_engine(test_url)
        tables_after = inspect(test_eng).get_table_names()
        assert "pipeline_jobs" not in tables_after
        assert "users" not in tables_after
        assert "videos" not in tables_after

    finally:
        if test_eng:
            test_eng.dispose()
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(f"DROP DATABASE IF EXISTS {test_db_name} WITH (FORCE)"))
