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
    """Run `alembic upgrade head` against `async_url`."""
    import alembic.config

    sync_url = async_url.replace("+asyncpg", "")
    prev = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = async_url
    try:
        alembic.config.main(
            argv=[
                "-c",
                str(_BACKEND_ROOT / "alembic.ini"),
                "-x",
                f"sqlalchemy.url={sync_url}",
                "upgrade",
                "head",
            ]
        )
    finally:
        if prev is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = prev
