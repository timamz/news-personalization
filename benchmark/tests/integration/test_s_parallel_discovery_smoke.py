"""
Parallel-discovery smoke test: fire all ten economics prompts at the
live devbox backend in parallel and assert the outcome matches the
behaviour we depend on.

Unlike the other ``test_s_*`` scenarios in this folder -- which run the
backend in-process against a throwaway Postgres + fake clock -- this
scenario hits a real backend and a real DB. The point is to catch
regressions that only surface under honest parallel load: DB pool
saturation, SOCKS / LLM-endpoint contention, Celery soft-time-limit
kills, and Conversational-Agent prompt drift that silently disables
auto-discovery.

Because of that cost (roughly USD 0.40 per run and several minutes of
wall-clock), the test is gated behind ``RUN_PARALLEL_DISCOVERY_SMOKE=1``
and is otherwise skipped. Opt in with::

    RUN_PARALLEL_DISCOVERY_SMOKE=1 uv run pytest \\
        tests/integration/test_s_parallel_discovery_smoke.py -s

The driver code (prompts, one-turn runner, DB poll) lives in
``benchmark/economics`` -- this file only wires those pieces into the
two behavioural assertions we care about:

1. Every subscription ends up with at least one attached source. Zero
   sources means the subscription is dead on arrival; tightly-scoped
   topics (e.g. "official Apple announcements only") legitimately
   resolve to a single canonical feed, so one is acceptable.
2. Across all ten subscriptions combined, every source kind
   (rss, telegram_channel, reddit_subreddit) is represented at least
   once.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

import asyncpg
import httpx
import pytest

_ECONOMICS_DIR = Path(__file__).resolve().parents[2] / "economics"
sys.path.insert(0, str(_ECONOMICS_DIR))

from prompts import PROMPTS  # noqa: E402, I001
from run_baseline import DEFAULT_API_URL, DEFAULT_DB_URL, run_one_prompt  # noqa: E402


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_PARALLEL_DISCOVERY_SMOKE") != "1",
    reason=(
        "Live-devbox smoke test gated by RUN_PARALLEL_DISCOVERY_SMOKE=1 "
        "(hits the real backend, costs real LLM money, takes several minutes)."
    ),
)


_API_URL = os.environ.get("ECON_API_URL", DEFAULT_API_URL)
_DB_URL = os.environ.get("ECON_DB_URL", DEFAULT_DB_URL)
_REQUIRED_KINDS = ("rss", "telegram_channel", "reddit_subreddit")
_MIN_SOURCES_PER_SUB = 1


async def _run_ten_prompts_in_parallel() -> list[dict[str, Any]]:
    run_id = uuid.uuid4().hex[:8]
    async with httpx.AsyncClient() as http:
        pool = await asyncpg.create_pool(_DB_URL, min_size=2, max_size=8)
        try:
            results = await asyncio.gather(
                *(run_one_prompt(http, pool, _API_URL, run_id, p) for p in PROMPTS),
                return_exceptions=True,
            )
        finally:
            await pool.close()

    normalized: list[dict[str, Any]] = []
    for prompt, result in zip(PROMPTS, results, strict=True):
        if isinstance(result, BaseException):
            normalized.append(
                {
                    "prompt_id": prompt["id"],
                    "error": f"{type(result).__name__}: {result}",
                    "subscriptions": [],
                }
            )
        else:
            normalized.append(result)
    return normalized


async def test_ten_parallel_discoveries_attach_multiple_sources_of_every_kind() -> None:
    results = await _run_ten_prompts_in_parallel()

    sparse_subs: list[str] = []
    kinds_seen: Counter[str] = Counter()
    for result in results:
        prompt_id = result.get("prompt_id", "?")
        if result.get("error"):
            sparse_subs.append(f"{prompt_id}: turn errored ({result['error']})")
            continue
        subs = result.get("subscriptions") or []
        if not subs:
            sparse_subs.append(f"{prompt_id}: no subscription was created")
            continue
        for sub in subs:
            src_count = int(sub.get("source_count") or 0)
            if src_count < _MIN_SOURCES_PER_SUB:
                sparse_subs.append(
                    f"{prompt_id} ({sub['id']}): only {src_count} source(s) attached; "
                    f"expected at least {_MIN_SOURCES_PER_SUB}"
                )
            for kind, count in (sub.get("sources_by_kind") or {}).items():
                kinds_seen[kind] += int(count)

    assert not sparse_subs, "subscriptions without enough attached sources:\n  - " + "\n  - ".join(
        sparse_subs
    )

    missing_kinds = [k for k in _REQUIRED_KINDS if kinds_seen.get(k, 0) == 0]
    assert not missing_kinds, (
        f"discovery never attached any source of kind(s) {missing_kinds}; "
        f"per-kind totals across all ten onboardings were {dict(kinds_seen)}"
    )
