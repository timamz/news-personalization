"""
Session-wide setup for benchmark integration tests.

Order is load-bearing. Before any news_service module is imported we
must:

  1. Load ``.env`` files so OPENAI_API_KEY and YANDEX_SEARCH_API_KEY land
     in the environment.
  2. Install the FakeClock datetime monkey-patch so the backend sees
     virtual time from the first call onward.
  3. Install the litellm cost-ledger wrappers so every LLM call made by
     news_service is measured.
  4. Create a throwaway Postgres database on devbox, install the schema,
     and set DATABASE_URL + REDIS_URL so every ``settings = get_settings()``
     call in news_service modules (most of which cache at import) reads
     the throwaway endpoints.

Only after that does pytest collect test modules, which are free to
``from news_service import ...``.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime
from pathlib import Path

import pytest
from dotenv import load_dotenv


def _load_env_files() -> None:
    """Populate os.environ from the first .env we can find.

    The worktree's ``backend/.env`` usually does not exist; the main
    repo's does. We walk candidate paths and load each with
    ``override=False`` so an already-set env var wins.
    """
    here = Path(__file__).resolve()
    candidates: list[Path] = [
        here.parents[1] / ".env",
        here.parents[2] / "backend" / ".env",
    ]
    parts = here.parts
    if ".claude" in parts and "worktrees" in parts:
        idx = parts.index(".claude")
        main_repo = Path(*parts[:idx])
        candidates.append(main_repo / "backend" / ".env")
    for p in candidates:
        if p.exists():
            load_dotenv(p, override=False)


_load_env_files()


_RUN_ID = uuid.uuid4().hex[:8]

from news_benchmark.clock import install_clock_patch  # noqa: E402

install_clock_patch(start=datetime.fromisoformat("2026-03-01T00:00:00+00:00"))

from news_benchmark.cost_ledger import install_litellm_wrappers  # noqa: E402

install_litellm_wrappers()

from news_benchmark.config import BenchmarkConfig  # noqa: E402
from news_benchmark.db import (  # noqa: E402
    create_bench_db,
    drop_bench_db,
    run_alembic_upgrade,
)

_CFG = BenchmarkConfig()
_BENCH_DB_URL = asyncio.run(create_bench_db(_CFG, _RUN_ID))
run_alembic_upgrade(_BENCH_DB_URL)

os.environ["DATABASE_URL"] = _BENCH_DB_URL
os.environ.setdefault("REDIS_URL", _CFG.benchmark_redis_url)
os.environ.setdefault("LITELLM_MODEL", _CFG.litellm_model)
os.environ.setdefault("LITELLM_EMBEDDING_MODEL", _CFG.litellm_embedding_model)
os.environ.setdefault("LITELLM_JUDGE_MODEL", _CFG.litellm_judge_model)


def pytest_sessionfinish(session, exitstatus):  # type: ignore[no-untyped-def]
    """Drop the throwaway DB unless we want it retained for post-mortem."""
    if _CFG.keep_db_on_failure and exitstatus != 0:
        return
    try:
        asyncio.run(drop_bench_db(_CFG, _RUN_ID))
    except Exception as exc:
        print(f"[conftest] drop_bench_db failed: {exc}")


@pytest.fixture
def run_id() -> str:
    return _RUN_ID


@pytest.fixture
async def world():
    """Install every news_service monkey-patch and uninstall after the test."""
    from news_benchmark.fakes.world import World

    w = World()
    w.install()
    try:
        yield w
    finally:
        w.uninstall()


@pytest.fixture
async def db_session():
    """Yield an AsyncSession bound to the throwaway bench DB.

    A fresh session per fixture invocation. Tests that need to observe
    writes committed from a scoped session inside a tool call should
    open an additional session via ``async_session_factory`` rather than
    reusing this one, because SQLAlchemy sessions cache per-session.
    """
    from news_service.db.session import async_session_factory

    async with async_session_factory() as sess:
        yield sess
