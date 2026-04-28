"""Smoke test: `alembic upgrade head` against an empty database.

The autouse ``reset_database`` fixture in conftest populates the schema
via ``Base.metadata.create_all``, which bypasses Alembic entirely. That
means a broken migration can ship to prod while the test suite stays
green. This test runs the actual Alembic upgrade path: drop every table
(including ``alembic_version``), run ``upgrade head``, and assert every
expected table plus the pgvector extension landed.

If this test fails on a PR, the migration series cannot successfully
bootstrap a fresh production database.
"""

import asyncio
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from news_service.core.config import get_settings

EXPECTED_TABLES = {
    "users",
    "subscriptions",
    "subscription_sources",
    "sources",
    "news_items",
    "sent_items",
    "source_removal_log",
    "failed_tasks",
}


@pytest_asyncio.fixture(loop_scope="session")
async def empty_database() -> AsyncGenerator[None]:
    """Drop every table including ``alembic_version`` before the test.

    The session-level ``reset_database`` autouse fixture already ran
    ``Base.metadata.create_all`` before this fixture fires. We undo that
    so the migration path starts from a truly empty schema.
    """

    engine = create_async_engine(get_settings().database_url)
    try:
        async with engine.begin() as conn:
            await conn.execute(text("DROP SCHEMA public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        yield
    finally:
        await engine.dispose()


@pytest.mark.xfail(
    reason=(
        "Known bug: 0001_baseline calls Base.metadata.create_all (current models, "
        "incl. delivery_webhook_url), then 0003 tries to ADD that column -> "
        "DuplicateColumnError. Fresh `alembic upgrade head` cannot bootstrap an "
        "empty DB. Fix: rewrite 0001 with explicit op.create_table calls."
    ),
    strict=True,
)
@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.usefixtures("empty_database")
async def test_alembic_upgrade_head_creates_the_full_production_schema() -> None:
    alembic_ini = Path(__file__).resolve().parents[2] / "alembic.ini"
    assert alembic_ini.exists(), f"alembic.ini not found at {alembic_ini}"

    cfg = Config(str(alembic_ini))
    cfg.set_main_option("script_location", str(alembic_ini.parent / "alembic"))
    cfg.set_main_option("sqlalchemy.url", get_settings().database_url)

    await asyncio.to_thread(command.upgrade, cfg, "head")

    engine = create_async_engine(get_settings().database_url)
    try:
        async with engine.connect() as conn:
            table_rows = await conn.execute(
                text(
                    "SELECT tablename FROM pg_tables "
                    "WHERE schemaname = 'public' AND tablename != 'alembic_version'"
                )
            )
            actual_tables = {row[0] for row in table_rows}

            version = await conn.scalar(text("SELECT version_num FROM alembic_version"))

            ext_rows = await conn.execute(
                text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
            )
            vector_installed = ext_rows.first() is not None
    finally:
        await engine.dispose()

    assert actual_tables == EXPECTED_TABLES, (
        f"alembic upgrade head produced the wrong set of tables; "
        f"missing={EXPECTED_TABLES - actual_tables}, "
        f"extra={actual_tables - EXPECTED_TABLES}"
    )
    assert version, "alembic_version table has no row after upgrade head"
    assert vector_installed, (
        "pgvector extension is not installed after upgrade head; "
        "embeddings will fail to index in production"
    )
