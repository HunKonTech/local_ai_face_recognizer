"""Database engine and session management.

All application code should obtain sessions via :func:`get_session` or
the :func:`session_scope` context manager.  The engine is created once
at startup via :func:`init_db`.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.models import Base

log = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionFactory: sessionmaker | None = None


def init_db(db_path: Path | str) -> Engine:
    """Create (or open) the SQLite database and run schema migrations.

    Call once at application startup.

    Args:
        db_path: Absolute or relative path to the SQLite file.
                 The parent directory is created if it does not exist.

    Returns:
        The SQLAlchemy :class:`Engine` instance.
    """
    global _engine, _SessionFactory

    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    url = f"sqlite:///{db_path}"
    log.info("Opening database: %s", url)

    engine = create_engine(
        url,
        connect_args={"check_same_thread": False},
        echo=False,  # Set True to debug SQL
    )

    # Enable WAL mode for better concurrency (readers don't block writers)
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):  # noqa: ANN001
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    # Create all tables that don't exist yet (idempotent)
    Base.metadata.create_all(engine)

    _engine = engine
    _SessionFactory = sessionmaker(bind=engine, expire_on_commit=False)

    log.info("Database ready: %s tables", len(Base.metadata.tables))
    return engine


def get_engine() -> Engine:
    """Return the application-level engine (must call :func:`init_db` first)."""
    if _engine is None:
        raise RuntimeError("Database not initialised — call init_db() first.")
    return _engine


def get_session() -> Session:
    """Create and return a new :class:`Session`.

    The caller is responsible for closing it.  Prefer :func:`session_scope`
    for automatic commit/rollback handling.
    """
    if _SessionFactory is None:
        raise RuntimeError("Database not initialised — call init_db() first.")
    return _SessionFactory()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """Provide a transactional session scope.

    Commits on clean exit, rolls back on any exception, always closes.

    Usage::

        with session_scope() as session:
            session.add(some_object)
    """
    session = get_session()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
