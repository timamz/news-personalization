"""
S-e2e-month: one-month steady-state cost benchmark.

Drives one digest-mode and one event-mode subscription through 30
simulated days using the existing harness primitives (FakeClock,
CostLedger, World fakes, VirtualScheduler, CeleryShim). The goal is
a single, minimal scenario that exercises every cost-bearing path
the production system touches in steady state:

  * ingest polling every 30 minutes (``_poll_all_feeds``)
  * per-poll event assessment + judge + webhook delivery
    (via the Celery shim that routes
    ``deliver_event_notifications_batch``)
  * daily digest cron check + digest writer + digest judge
    (``_schedule_due_digests`` enqueues ``_deliver_digest``)
  * weekly event verifier tick
    (``_reflect_event_subscriptions``)

The scenario's source counts are pinned from
``benchmark/economics/constants.py`` (``AVG_SOURCES_PER_*_SUB``) and
per-source item throughput from ``AVG_ITEMS_PER_SOURCE_PER_DAY`` so
any shift in those measurements flows straight into the simulation
without code edits.

Opt-in: this scenario hits the real LLM provider configured for the
conftest. Expect a few dollars of real spend and several minutes of
wall-clock for the default 30-day window. Gate it with::

    RUN_E2E_MONTH=1 uv run pytest \\
        tests/integration/test_s_e2e_month.py -s

At the end the test writes a full per-row cost ledger snapshot plus
aggregates to ``benchmark/economics/results/e2e_month_<run_id>.json``
and prints a short stdout summary so CI logs show the final number.

Scope: cost measurement only. Content quality is NOT evaluated here --
the v3 fabric-based benchmark (removed in commit 40d0eb3, preserved
at commit 535dab2) is the right home for quality scoring.
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from news_benchmark.clock import CLOCK
from news_benchmark.cost_ledger import LEDGER
from news_benchmark.fakes.adapters import FakeAdapter
from news_benchmark.scheduler import VirtualScheduler
from tests.integration._e2e_month_corpus import (
    DIGEST_RETRIEVAL_QUERY,
    DIGEST_USER_SPEC,
    EVENT_RETRIEVAL_QUERY,
    EVENT_USER_SPEC,
    build_timeline,
)

_ECONOMICS_DIR = Path(__file__).resolve().parents[2] / "economics"
sys.path.insert(0, str(_ECONOMICS_DIR))

from constants import (  # noqa: E402, I001
    AVG_ITEMS_PER_SOURCE_PER_DAY,
    AVG_SOURCES_PER_DIGEST_SUB,
    AVG_SOURCES_PER_EVENT_SUB,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_E2E_MONTH") != "1",
    reason=(
        "Month-long steady-state benchmark gated by RUN_E2E_MONTH=1 "
        "(hits the real LLM provider, costs real money, runs several minutes)."
    ),
)


_SIM_START = datetime(2026, 5, 1, tzinfo=UTC)
_SIM_DAYS = 30

_DIGEST_WEBHOOK_URL = "https://bench.invalid/webhook/e2e-month-digest"
_EVENT_WEBHOOK_URL = "https://bench.invalid/webhook/e2e-month-event"
_DIGEST_CRON = "0 8 * * *"

_DIGEST_SOURCE_COUNT = round(AVG_SOURCES_PER_DIGEST_SUB)
_EVENT_SOURCE_COUNT = round(AVG_SOURCES_PER_EVENT_SUB)

_RESULTS_DIR = _ECONOMICS_DIR / "results"


async def _seed_subscription(
    *,
    world,
    user_id: uuid.UUID,
    sub_id: uuid.UUID,
    delivery_mode: str,
    user_spec: str,
    retrieval_query: str,
    source_urls: list[str],
    webhook_url: str,
    schedule_cron: str | None,
) -> None:
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.source import Source
    from news_service.models.subscription import Subscription
    from news_service.models.subscription_source import SubscriptionSource
    from news_service.models.user import User

    topic_embedding = await embed_text(retrieval_query)

    async with async_session_factory() as s:
        s.add(
            User(
                id=user_id,
                api_key=f"bench-{user_id.hex}",
                language="en",
                timezone="UTC",
                delivery_webhook_url=webhook_url,
                has_onboarded=True,
            )
        )
        s.add(
            Subscription(
                id=sub_id,
                user_id=user_id,
                user_spec=user_spec,
                delivery_mode=delivery_mode,
                schedule_cron=schedule_cron,
                digest_language="en",
                delivery_webhook_url=webhook_url,
                topic_embedding=topic_embedding,
                is_active=True,
            )
        )
        for idx, source_url in enumerate(source_urls):
            source_id = uuid.uuid4()
            s.add(
                Source(
                    id=source_id,
                    url=source_url,
                    title=f"e2e-{delivery_mode}-src-{idx:02d}",
                    source_description=f"Synthetic {delivery_mode} source for e2e benchmark.",
                )
            )
            s.add(
                SubscriptionSource(
                    subscription_id=sub_id,
                    source_id=source_id,
                    is_user_specified=True,
                )
            )
        await s.commit()


def _schedule_cycles(
    *,
    sched: VirtualScheduler,
    poll_impl,
    digest_impl,
    verifier_impl,
    start: datetime,
    days: int,
) -> None:
    poll_every = timedelta(minutes=30)
    for tick in range(days * 48):
        fire_at = start + tick * poll_every
        sched.schedule(fire_at, poll_impl, label=f"poll#{tick:04d}")

    for day in range(days):
        fire_at = start + timedelta(days=day, hours=8, minutes=1)
        sched.schedule(fire_at, digest_impl, label=f"digest-cron#{day:02d}")

    for day in range(6, days, 7):
        fire_at = start + timedelta(days=day, hours=1)
        sched.schedule(fire_at, verifier_impl, label=f"verifier#{day:02d}")


def _summarize_rows(rows: list[Any]) -> dict[str, Any]:
    by_agent: dict[str, dict[str, float | int]] = {}
    by_call_type: dict[str, dict[str, float | int]] = {}
    total_usd = 0.0
    total_prompt = 0
    total_completion = 0
    for r in rows:
        total_usd += r.usd_cost
        total_prompt += r.prompt_tokens
        total_completion += r.completion_tokens
        a = by_agent.setdefault(r.agent_path or "<untagged>", {"calls": 0, "usd": 0.0})
        a["calls"] = int(a["calls"]) + 1
        a["usd"] = float(a["usd"]) + r.usd_cost
        c = by_call_type.setdefault(r.call_type, {"calls": 0, "usd": 0.0})
        c["calls"] = int(c["calls"]) + 1
        c["usd"] = float(c["usd"]) + r.usd_cost
    return {
        "total_calls": len(rows),
        "total_usd": round(total_usd, 6),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": total_completion,
        "by_agent": {
            k: {"calls": v["calls"], "usd": round(v["usd"], 6)} for k, v in by_agent.items()
        },
        "by_call_type": {
            k: {"calls": v["calls"], "usd": round(v["usd"], 6)} for k, v in by_call_type.items()
        },
    }


@pytest.mark.asyncio
async def test_one_month_steady_state_cost(world) -> None:
    """Drive 30 simulated days of one digest + one event sub, write the exact cost."""
    run_id = uuid.uuid4().hex[:8]

    CLOCK.reset_to(_SIM_START)

    digest_user_id = uuid.uuid4()
    digest_sub_id = uuid.uuid4()
    event_user_id = uuid.uuid4()
    event_sub_id = uuid.uuid4()

    digest_source_urls = [
        f"https://e2e-month-digest-{i:02d}.invalid/feed.xml" for i in range(_DIGEST_SOURCE_COUNT)
    ]
    event_source_urls = [
        f"https://e2e-month-event-{i:02d}.invalid/feed.xml" for i in range(_EVENT_SOURCE_COUNT)
    ]

    await _seed_subscription(
        world=world,
        user_id=digest_user_id,
        sub_id=digest_sub_id,
        delivery_mode="digest",
        user_spec=DIGEST_USER_SPEC,
        retrieval_query=DIGEST_RETRIEVAL_QUERY,
        source_urls=digest_source_urls,
        webhook_url=_DIGEST_WEBHOOK_URL,
        schedule_cron=_DIGEST_CRON,
    )
    await _seed_subscription(
        world=world,
        user_id=event_user_id,
        sub_id=event_sub_id,
        delivery_mode="event",
        user_spec=EVENT_USER_SPEC,
        retrieval_query=EVENT_RETRIEVAL_QUERY,
        source_urls=event_source_urls,
        webhook_url=_EVENT_WEBHOOK_URL,
        schedule_cron=None,
    )

    digest_items = build_timeline(
        source_urls=digest_source_urls,
        topic="digest",
        start=_SIM_START,
        days=_SIM_DAYS,
        items_per_source_per_day=AVG_ITEMS_PER_SOURCE_PER_DAY,
    )
    event_items = build_timeline(
        source_urls=event_source_urls,
        topic="event",
        start=_SIM_START,
        days=_SIM_DAYS,
        items_per_source_per_day=AVG_ITEMS_PER_SOURCE_PER_DAY,
    )
    for url in digest_source_urls:
        world.adapters[url] = FakeAdapter(
            source_url=url,
            items=[i for i in digest_items if i.source_url == url],
        )
    for url in event_source_urls:
        world.adapters[url] = FakeAdapter(
            source_url=url,
            items=[i for i in event_items if i.source_url == url],
        )

    from news_service.tasks.deliver_digest import _deliver_digest  # noqa: F401
    from news_service.tasks.poll_feeds import _poll_all_feeds
    from news_service.tasks.reflect_events import _reflect_event_subscriptions
    from news_service.tasks.schedule_digests import _schedule_due_digests

    async def poll_tick() -> None:
        await _poll_all_feeds()
        await world.celery.drain()

    async def digest_tick() -> None:
        await _schedule_due_digests()
        await world.celery.drain()

    async def verifier_tick() -> None:
        await _reflect_event_subscriptions()
        await world.celery.drain()

    ledger_start_index = len(LEDGER.rows())

    sched = VirtualScheduler()
    _schedule_cycles(
        sched=sched,
        poll_impl=poll_tick,
        digest_impl=digest_tick,
        verifier_impl=verifier_tick,
        start=_SIM_START,
        days=_SIM_DAYS,
    )
    end = _SIM_START + timedelta(days=_SIM_DAYS)
    await sched.run(until=end)

    new_rows = LEDGER.rows()[ledger_start_index:]
    summary = _summarize_rows(new_rows)

    digest_delivered = len(world.delivery.for_url(_DIGEST_WEBHOOK_URL))
    event_delivered = len(world.delivery.for_url(_EVENT_WEBHOOK_URL))

    result_payload = {
        "run_id": run_id,
        "simulated_start": _SIM_START.isoformat(),
        "simulated_end": end.isoformat(),
        "simulated_days": _SIM_DAYS,
        "digest_sub_id": str(digest_sub_id),
        "event_sub_id": str(event_sub_id),
        "digest_source_count": _DIGEST_SOURCE_COUNT,
        "event_source_count": _EVENT_SOURCE_COUNT,
        "items_per_source_per_day": AVG_ITEMS_PER_SOURCE_PER_DAY,
        "digest_items_injected": len(digest_items),
        "event_items_injected": len(event_items),
        "digest_deliveries": digest_delivered,
        "event_deliveries": event_delivered,
        "cost_summary": summary,
        "ledger_rows": [asdict(r) for r in new_rows],
    }

    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _RESULTS_DIR / f"e2e_month_{run_id}.json"
    out_path.write_text(json.dumps(result_payload, indent=2, default=str))

    print(f"\n[e2e-month {run_id}] simulated_days={_SIM_DAYS} wrote={out_path}")
    print(
        f"[e2e-month {run_id}] digest_deliveries={digest_delivered} "
        f"event_deliveries={event_delivered} "
        f"items_injected={len(digest_items) + len(event_items)}"
    )
    print(f"[e2e-month {run_id}] total_cost_usd={summary['total_usd']}")
    for agent, stats in sorted(summary["by_agent"].items(), key=lambda kv: -kv[1]["usd"]):
        print(
            f"[e2e-month {run_id}]   {agent:<20s} calls={stats['calls']:>5d} usd={stats['usd']:.6f}"
        )

    assert summary["total_calls"] > 0, "ledger captured no LLM calls -- harness wired wrong"
    assert digest_delivered >= 1, (
        f"expected at least one digest delivery across {_SIM_DAYS} days, got {digest_delivered}"
    )
