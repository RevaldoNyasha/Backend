from pathlib import Path

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.base import Base
from app.db.migrations import BASELINE_REVISION, alembic_config, upgrade_database
from app.models import Document, Project, User  # noqa: F401


@pytest.fixture
def database_url(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'app.db').as_posix()}"


def current_revision(engine: Engine) -> str | None:
    with engine.connect() as connection:
        return MigrationContext.configure(connection).get_current_revision()


def schema_differences(engine: Engine) -> list:
    with engine.connect() as connection:
        return compare_metadata(MigrationContext.configure(connection), Base.metadata)


def head_revision() -> str:
    from alembic.script import ScriptDirectory

    from app.db.migrations import alembic_config

    return ScriptDirectory.from_config(alembic_config("sqlite://")).get_current_head()


def test_migrations_build_the_same_schema_as_the_models(database_url: str) -> None:
    upgrade_database(database_url)

    engine = create_engine(database_url)
    try:
        assert current_revision(engine) == head_revision()
        assert schema_differences(engine) == []
    finally:
        engine.dispose()


def test_pre_migration_database_is_stamped_and_keeps_its_data(database_url: str) -> None:
    # Recreate a database from the old `create_all` startup: the baseline schema,
    # a row of data, and no alembic_version table.
    command.upgrade(alembic_config(database_url), BASELINE_REVISION)
    engine = create_engine(database_url)
    with engine.begin() as connection:
        connection.execute(text("DROP TABLE alembic_version"))
    with Session(engine) as session:
        session.add(User(id="u1", name="Ada", email="ada@example.com", password_hash="x"))
        session.commit()
    engine.dispose()

    upgrade_database(database_url)

    engine = create_engine(database_url)
    try:
        assert current_revision(engine) == head_revision()
        with Session(engine) as session:
            assert session.get(User, "u1") is not None
    finally:
        engine.dispose()


def test_upgrade_is_idempotent(database_url: str) -> None:
    upgrade_database(database_url)
    upgrade_database(database_url)

    engine = create_engine(database_url)
    try:
        assert current_revision(engine) == head_revision()
    finally:
        engine.dispose()


def test_baseline_revision_exists() -> None:
    from alembic.script import ScriptDirectory

    from app.db.migrations import alembic_config

    script = ScriptDirectory.from_config(alembic_config("sqlite://"))
    assert script.get_revision(BASELINE_REVISION) is not None


def test_sqlite_enforces_foreign_keys(database_url: str) -> None:
    upgrade_database(database_url)
    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            assert connection.execute(text("PRAGMA foreign_keys")).scalar() == 1

        with Session(engine) as session:
            session.add(Project(user_id="missing-user", name="Orphan", description=""))
            with pytest.raises(IntegrityError):
                session.commit()
    finally:
        engine.dispose()


def test_deleting_a_user_cascades_to_their_projects(database_url: str) -> None:
    upgrade_database(database_url)
    engine = create_engine(database_url)
    try:
        with Session(engine) as session:
            session.add(User(id="u1", name="Ada", email="ada@example.com", password_hash="x"))
            session.flush()
            session.add(Project(id="p1", user_id="u1", name="Study", description=""))
            session.commit()

            session.execute(text("DELETE FROM users WHERE id = 'u1'"))
            session.commit()

            assert session.get(Project, "p1") is None
    finally:
        engine.dispose()
