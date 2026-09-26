"""Apply Alembic migrations from inside the app (used on development startup)."""

import logging

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, inspect

from app.core.config import BACKEND_ROOT
from app.db.database import prepare_database_url

logger = logging.getLogger(__name__)

# Revision that matches the schema `Base.metadata.create_all` used to build
# before migrations existed.
BASELINE_REVISION = "0001_baseline"


def alembic_config(database_url: str) -> Config:
    config = Config(str(BACKEND_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(BACKEND_ROOT / "alembic"))
    config.set_main_option("sqlalchemy.url", database_url.replace("%", "%%"))
    config.attributes["configure_logger"] = False
    return config


def upgrade_database(database_url: str) -> None:
    """Bring the database to the latest revision.

    A database created by the old `create_all` startup has the tables but no
    `alembic_version`; it is stamped at the baseline first so the baseline's
    CREATE TABLEs are not re-run against existing tables.
    """
    database_url = prepare_database_url(database_url)
    config = alembic_config(database_url)
    engine = create_engine(database_url)
    try:
        tables = set(inspect(engine).get_table_names())
    finally:
        engine.dispose()

    if "alembic_version" not in tables and "users" in tables:
        logger.info("Stamping pre-migration database at %s", BASELINE_REVISION)
        command.stamp(config, BASELINE_REVISION)

    command.upgrade(config, "head")
