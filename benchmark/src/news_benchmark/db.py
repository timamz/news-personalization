"""
Throwaway benchmark DB lifecycle against devbox Postgres.

Creates a fresh database `news_bench_<run_id>`, runs `alembic upgrade head`
against it (exercising real migrations, not create_all), and drops it at
run end unless keep_db_on_failure is set.

Uses asyncpg via SQLAlchemy for CREATE/DROP; alembic is invoked
programmatically by setting DATABASE_URL in the environment and calling
alembic.config.main(["upgrade", "head"]) pointed at the backend's
alembic.ini.

Usage:

    url = await create_bench_db(cfg, run_id)
    run_alembic_upgrade(url)
    try:
        ...
    finally:
        await drop_bench_db(cfg, run_id)
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine

from news_benchmark.config import BenchmarkConfig

_BACKEND_ROOT = Path(__file__).resolve().parents[3] / "backend"


async def create_bench_db(cfg: BenchmarkConfig, run_id: str) -> str:
    """CREATE DATABASE news_bench_<run_id> on devbox. Returns its async URL."""
    admin_url = cfg.admin_db_url()
    db_name = f"news_bench_{run_id}"
    engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    async with engine.connect() as conn:
        await conn.exec_driver_sql(f'DROP DATABASE IF EXISTS "{db_name}"')
        await conn.exec_driver_sql(f'CREATE DATABASE "{db_name}"')
    await engine.dispose()
    return cfg.bench_db_url(run_id)


async def drop_bench_db(cfg: BenchmarkConfig, run_id: str) -> None:
    """DROP the throwaway DB. Forcibly closes sessions first."""
    admin_url = cfg.admin_db_url()
    db_name = f"news_bench_{run_id}"
    engine = create_async_engine(admin_url, isolation_level="AUTOCOMMIT")
    async with engine.connect() as conn:
        await conn.exec_driver_sql(
            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
            f"WHERE datname = '{db_name}' AND pid <> pg_backend_pid()"
        )
        await conn.exec_driver_sql(f'DROP DATABASE IF EXISTS "{db_name}"')
    await engine.dispose()


def run_alembic_upgrade(async_url: str) -> None:
    """Create the full schema in a subprocess, then stamp alembic to head.

    The backend's baseline migration (0001_baseline) does a full
    ``Base.metadata.create_all`` using the *current* SQLAlchemy model
    definitions. Subsequent migrations (0002, 0003, ...) then try to
    ALTER those same tables and fail because the baseline already
    installed them with the current shape. For a throwaway bench DB
    we skip the alembic history entirely: install pgvector, run
    create_all, and stamp alembic_version to head so models work.
    """
    import subprocess
    import sys
    import textwrap

    env = dict(os.environ)
    env["DATABASE_URL"] = async_url

    program = textwrap.dedent(
        """
        import asyncio
        from sqlalchemy.ext.asyncio import create_async_engine
        from news_service.core.config import get_settings
        from news_service.models import Base

        async def main():
            settings = get_settings()
            engine = create_async_engine(settings.database_url, echo=False)
            async with engine.begin() as conn:
                await conn.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
                await conn.run_sync(Base.metadata.create_all)
                await conn.exec_driver_sql(
                    "CREATE TABLE IF NOT EXISTS alembic_version "
                    "(version_num VARCHAR(32) NOT NULL PRIMARY KEY)"
                )
                await conn.exec_driver_sql(
                    "INSERT INTO alembic_version (version_num) "
                    "SELECT '0003_user_delivery_webhook_url' "
                    "WHERE NOT EXISTS (SELECT 1 FROM alembic_version)"
                )
            await engine.dispose()

        asyncio.run(main())
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        cwd=str(_BACKEND_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"schema install failed (rc={result.returncode}):\n"
            f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        )
