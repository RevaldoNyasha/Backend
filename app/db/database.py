from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


@event.listens_for(Engine, "connect")
def _enable_sqlite_foreign_keys(dbapi_connection: Any, _connection_record: Any) -> None:
    """SQLite ignores FOREIGN KEY / ON DELETE CASCADE unless this is set per connection.

    Registered on every Engine (including test engines) so SQLite behaves like
    PostgreSQL will after the move to Neon.
    """
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def prepare_database_url(database_url: str) -> str:
    parsed_url = make_url(database_url)
    if parsed_url.get_backend_name() != "sqlite":
        return database_url

    database_path_value = parsed_url.database
    if not database_path_value or database_path_value == ":memory:":
        return database_url

    database_path = Path(database_path_value)
    if not database_path.is_absolute():
        database_path = Path.cwd() / database_path
    database_path.parent.mkdir(parents=True, exist_ok=True)
    return database_url


def _create_engine(database_url: str) -> Engine:
    parsed_url = make_url(database_url)
    engine_options: dict[str, Any] = {"pool_pre_ping": True}

    if parsed_url.get_backend_name() == "sqlite":
        engine_options["connect_args"] = {"check_same_thread": False}
        if parsed_url.database in (None, "", ":memory:"):
            engine_options["poolclass"] = StaticPool

    return create_engine(database_url, **engine_options)


def get_engine() -> Engine:
    global _engine

    if not settings.database_url:
        raise RuntimeError("DATABASE_URL must be configured before database access")

    if _engine is None:
        prepared_database_url = prepare_database_url(settings.database_url)
        _engine = _create_engine(prepared_database_url)

    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _session_factory

    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

    return _session_factory


def get_db() -> Generator[Session, None, None]:
    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()
