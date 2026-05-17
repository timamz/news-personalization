"""
Smoke: a single onboarding turn through the real Conv Agent.

Iteration vehicle for debugging the 7-day e2e hang. Wraps every
``litellm.acompletion`` / ``litellm.aembedding`` with timing logs so a
stall surfaces as a printed line, and hard-caps the whole turn at
180 seconds so a hung call doesn't lock the test.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid

import pytest

from news_benchmark.clock import CLOCK
from news_benchmark.cost_ledger import LEDGER
from tests.integration._e2e_month_corpus import build_timeline

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_SMOKE_ONBOARD") != "1",
    reason="opt-in: RUN_SMOKE_ONBOARD=1",
)


_DIGEST_SOURCE_UNIVERSE = [
    "https://www.euractiv.com/section/energy/feed/",
    "https://www.politico.eu/section/energy/feed/",
    "https://euobserver.com/feeds/energy.rss",
]

_WEBHOOK = "https://bench.invalid/webhook/smoke-onboard"

_MESSAGE = (
    "Hi. Set up a DAILY DIGEST subscription for EU energy and climate policy. "
    "English. Deliver at 08:00 UTC daily. Webhook URL "
    f"{_WEBHOOK}. Find sources and create the subscription now. "
    "Do not ask follow-up questions."
)


def _install_llm_call_tracer() -> None:
    """Wrap litellm.acompletion / aembedding with timing prints."""
    import litellm

    orig_acompletion = litellm.acompletion
    orig_aembedding = litellm.aembedding

    async def traced_acompletion(*args, **kwargs):
        model = kwargs.get("model", "?")
        t0 = time.monotonic()
        wall = int(time.monotonic())
        print(f"[trace t={wall:06d}] acompletion BEGIN model={model}", flush=True)
        try:
            result = await orig_acompletion(*args, **kwargs)
            dur = time.monotonic() - t0
            print(f"[trace t={wall:06d}] acompletion END   model={model} dur={dur:.2f}s", flush=True)
            return result
        except Exception as exc:
            dur = time.monotonic() - t0
            print(
                f"[trace t={wall:06d}] acompletion FAIL  model={model} dur={dur:.2f}s "
                f"err={type(exc).__name__}: {exc}",
                flush=True,
            )
            raise

    async def traced_aembedding(*args, **kwargs):
        model = kwargs.get("model", "?")
        t0 = time.monotonic()
        wall = int(time.monotonic())
        print(f"[trace t={wall:06d}] aembedding BEGIN model={model}", flush=True)
        try:
            result = await orig_aembedding(*args, **kwargs)
            dur = time.monotonic() - t0
            print(f"[trace t={wall:06d}] aembedding END   model={model} dur={dur:.2f}s", flush=True)
            return result
        except Exception as exc:
            dur = time.monotonic() - t0
            print(
                f"[trace t={wall:06d}] aembedding FAIL  model={model} dur={dur:.2f}s "
                f"err={type(exc).__name__}: {exc}",
                flush=True,
            )
            raise

    litellm.acompletion = traced_acompletion
    litellm.aembedding = traced_aembedding


@pytest.mark.asyncio
async def test_single_onboard_turn(world) -> None:
    """One onboarding turn with per-LLM-call tracing and 180s outer cap."""
    from datetime import UTC, datetime

    from news_benchmark.fakes.adapters import FakeAdapter
    from news_service.db.session import async_session_factory
    from news_service.models.user import User
    from news_service.schemas.conversation import ConversationState

    _install_llm_call_tracer()

    sim_start = datetime(2026, 5, 1, tzinfo=UTC)
    CLOCK.reset_to(sim_start)

    items = build_timeline(
        source_urls=_DIGEST_SOURCE_UNIVERSE,
        topic="digest",
        start=sim_start,
        days=2,
        items_per_source_per_day=2,
    )
    by_source: dict[str, list] = {}
    for it in items:
        by_source.setdefault(it.source_url, []).append(it)
    for url in _DIGEST_SOURCE_UNIVERSE:
        world.adapters[url] = FakeAdapter(
            source_url=url, items=sorted(by_source.get(url, []), key=lambda x: x.fake_ts)
        )

    user_id = uuid.uuid4()
    async with async_session_factory() as s:
        s.add(
            User(
                id=user_id,
                api_key=f"smoke-{user_id.hex}",
                language="en",
                timezone="UTC",
                has_onboarded=False,
            )
        )
        await s.commit()

    state = ConversationState(user_id=str(user_id), user_language="en")

    from news_benchmark.simulator import run_one_turn

    t0 = time.monotonic()
    print(f"[smoke] turn begin", flush=True)

    async def _do() -> str:
        async with async_session_factory() as s:
            user = await s.get(User, user_id)
            assert user is not None
            return await run_one_turn(state=state, user=user, db_session=s, message=_MESSAGE)

    reply = await asyncio.wait_for(_do(), timeout=180.0)
    print(f"[smoke] turn done in {time.monotonic() - t0:.2f}s", flush=True)
    print(f"[smoke] reply preview: {reply[:200]!r}", flush=True)
    print(f"[smoke] ledger: {len(LEDGER.rows())} calls, ${LEDGER.total_usd():.4f}", flush=True)
    print(f"[smoke] drain begin", flush=True)
    await asyncio.wait_for(world.celery.drain(), timeout=300.0)
    print(f"[smoke] drain done in {time.monotonic() - t0:.2f}s total", flush=True)
    print(f"[smoke] final ledger: {len(LEDGER.rows())} calls, ${LEDGER.total_usd():.4f}", flush=True)
