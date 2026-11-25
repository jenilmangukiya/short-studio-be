from typing import Any
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from .config import settings

_args = (
    {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
)
engine = create_engine(settings.database_url, pool_pre_ping=True, connect_args=_args)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


def get_db():
    with SessionLocal() as db:
        yield db


def acquire_video_publication_lock(session, video_id):
    """Acquire an exclusive lock on the Video row to synchronize publication.

    In PostgreSQL, uses SELECT ... FOR UPDATE.
    In SQLite, issues an update on the row to escalate the transaction to a reserved/exclusive lock.
    """
    from sqlalchemy import select, text
    from .models import Video

    bind = session.get_bind()
    if bind and bind.dialect.name == "sqlite":
        session.execute(
            text("UPDATE videos SET updated_at = updated_at WHERE id = :vid"),
            {"vid": str(video_id)},
        )
    return session.scalar(
        select(Video)
        .where(Video.id == video_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )


def atomic_promote_file(
    temp_path,
    final_path,
    session,
    video_id,
    expected_hash: str | None,
    expected_style_id: str | None,
    step_name: str,
    expected_execution_id: str | None = None,
    _post_commit_hook: Any = None,
) -> str:
    """Promote a generated file to an immutable generation path and update canonical destination.
    Guarantees that database commit failure leaves previous valid media intact and unlinks the uncommitted generation.
    Post-commit canonical file update is strictly conditional on this generation still being authoritative.
    Returns the authoritative promoted path string.
    """
    import os
    import shutil
    import uuid
    from pathlib import Path
    from .styles import canonical_snapshot_hash

    temp_p = Path(temp_path)
    final_p = Path(final_path)

    if step_name == "render":
        from .steps.render import PipelineError
    elif step_name in ("subtitles", "subtitle"):
        from .steps.subtitles import PipelineError
    else:
        from .pipeline import PipelineError

    if not temp_p.is_file() or temp_p.stat().st_size == 0:
        raise PipelineError(f"Temporary output file '{temp_p}' is empty or missing")

    video_dir = final_p.parent
    gen_dir = video_dir / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)

    hash8 = expected_hash[:8] if expected_hash else uuid.uuid4().hex[:8]
    gen_file = gen_dir / f"{step_name}_{hash8}_{uuid.uuid4().hex[:8]}.mp4"

    # 1. Stage into immutable generation path first (does NOT touch existing canonical file)
    shutil.move(str(temp_p), str(gen_file))

    # 2. Under exclusive Video row publication lock, verify staleness and commit to DB
    if session is not None:
        try:
            locked_vid = acquire_video_publication_lock(session, video_id)
            if locked_vid is not None:
                curr_hash = canonical_snapshot_hash(
                    getattr(locked_vid, "visual_style_snapshot", None)
                )
                curr_id = getattr(locked_vid, "visual_style_id", None)
                st = next(
                    (
                        s
                        for s in (locked_vid.steps or [])
                        if s.name in (step_name, "subtitle")
                    ),
                    None,
                )
                if (
                    (expected_hash and curr_hash != expected_hash)
                    or (expected_style_id and curr_id != expected_style_id)
                    or (st is not None and getattr(st, "status", None) == "stale")
                ):
                    gen_file.unlink(missing_ok=True)
                    session.rollback()
                    raise PipelineError(
                        f"{step_name.capitalize()} job was superseded by visual style change; output discarded"
                    )

                if step_name == "render":
                    locked_vid.output_path = str(gen_file)
                    if st is not None:
                        rout = dict(st.output) if isinstance(st.output, dict) else {}
                        rout["video_path"] = str(gen_file)
                        if "render" in rout and isinstance(rout["render"], dict):
                            rout["render"]["video_path"] = str(gen_file)
                        rmeta = (
                            dict(rout.get("meta"))
                            if isinstance(rout.get("meta"), dict)
                            else {}
                        )
                        if expected_execution_id:
                            rmeta["execution_id"] = str(expected_execution_id)
                        rout["meta"] = rmeta
                        st.output = rout
                elif step_name in ("subtitles", "subtitle"):
                    if st is not None:
                        sout = dict(st.output) if isinstance(st.output, dict) else {}
                        sub_dict = (
                            dict(sout.get("subtitles"))
                            if isinstance(sout.get("subtitles"), dict)
                            else {}
                        )
                        sub_dict["captioned_video_path"] = str(gen_file)
                        sout["subtitles"] = sub_dict
                        sout["captioned_video_path"] = str(gen_file)
                        smeta = (
                            dict(sout.get("meta"))
                            if isinstance(sout.get("meta"), dict)
                            else {}
                        )
                        if expected_execution_id:
                            smeta["execution_id"] = str(expected_execution_id)
                        sout["meta"] = smeta
                        st.output = sout

            session.commit()
        except PipelineError:
            raise
        except Exception as exc:
            gen_file.unlink(missing_ok=True)
            session.rollback()
            raise PipelineError(
                f"Database commit failed during {step_name} promotion: {exc}"
            ) from exc

    # Optional test hook for deterministic interleaving between commit and canonical replacement
    if _post_commit_hook is not None:
        _post_commit_hook()

    # 3. Only after DB transaction commits successfully, update canonical convenience pointer atomically.
    # Verification: Ensure this generation is STILL the authoritative output before replacing final_p!
    if session is not None:
        try:
            check_vid = acquire_video_publication_lock(session, video_id)
            if check_vid is not None:
                curr_hash = canonical_snapshot_hash(
                    getattr(check_vid, "visual_style_snapshot", None)
                )
                curr_id = getattr(check_vid, "visual_style_id", None)
                st = next(
                    (
                        s
                        for s in (check_vid.steps or [])
                        if s.name in (step_name, "subtitle")
                    ),
                    None,
                )

                is_authoritative = True
                if (
                    (expected_hash and curr_hash != expected_hash)
                    or (expected_style_id and curr_id != expected_style_id)
                    or (st is not None and getattr(st, "status", None) == "stale")
                ):
                    is_authoritative = False

                # Step-specific generation identity check:
                if step_name == "render":
                    # For render, verify that the current authoritative video.output_path
                    # identifies the exact generation file being promoted!
                    if getattr(check_vid, "output_path", None) != str(gen_file):
                        is_authoritative = False
                elif step_name in ("subtitles", "subtitle"):
                    # For subtitles, verify the authoritative captioned output path in the current subtitle step output.
                    # Do not use video.output_path as a substitute; it identifies the rendered input video.
                    curr_captioned = None
                    if st is not None and isinstance(st.output, dict):
                        curr_captioned = st.output.get(
                            "captioned_video_path"
                        ) or st.output.get("subtitles", {}).get("captioned_video_path")
                    if curr_captioned != str(gen_file):
                        is_authoritative = False

                # Verify the step's current execution identity where available:
                if (
                    expected_execution_id
                    and st is not None
                    and isinstance(st.output, dict)
                ):
                    curr_exec = st.output.get("meta", {}).get("execution_id")
                    if curr_exec and curr_exec != str(expected_execution_id):
                        is_authoritative = False

                if not is_authoritative:
                    # Output was superseded or invalidated before canonical file replacement!
                    # Do NOT replace final_p. Return gen_file path.
                    return str(gen_file)
        except Exception:
            return str(gen_file)

    try:
        final_p.parent.mkdir(parents=True, exist_ok=True)
        tmp_canon = video_dir / f"tmp_canon_{uuid.uuid4().hex[:8]}.mp4"
        shutil.copyfile(str(gen_file), str(tmp_canon))
        os.replace(str(tmp_canon), str(final_p))
    except Exception:
        pass

    return str(final_p)
