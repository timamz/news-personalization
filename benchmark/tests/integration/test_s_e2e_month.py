"""
S-e2e-month v2: one-month steady-state cost benchmark.

Drives a single real user through a simulated month of two
subscriptions -- one digest, one event -- using the real
Conversational Agent for onboarding, the real Discovery / Finder
agents for source selection, and the full steady-state pipeline
(polling, digest writer + judge, batch assessor + event judge,
delivery, weekly verifier, reflector).

What this scenario measures
---------------------------
The authoritative monthly unit-economics number quoted in the
economics report. The result JSON carries per-subscription costs
(attributed via the production ``current_subscription_id`` ContextVar
that ``subscription_tag`` pushes around every delivery / discovery /
verifier dispatch), a by-phase and by-agent breakdown, and the
Yandex Search API cost aggregate.

What is NOT in scope
--------------------
Quality evaluation. The v3 benchmark fabric (removed at commit
40d0eb3, preserved at 535dab2) is the right home for that. We keep
the restored body banks from s01 / s03 as a believable-looking
content stream, not as a judged corpus.

Load parameters come from ``benchmark/economics/constants.py``.
Default mode is ``avg`` (AVG_ITEMS_PER_SOURCE_PER_DAY = 3); set
``E2E_ITEMS_MODE=max`` to rerun at MAX_ITEMS_PER_SOURCE_PER_DAY,
a worst-case capacity-planning number.

Forced maintenance paths
------------------------
The goal is to *price* every production code path the system runs in
a real month, not to verify that each organic trigger fires. Relying on
organic firing (engineering a stale source + hoping the reflector
notices, crossing a cosine-similarity drift threshold, praying the
judge returns REVISE) is flaky and LLM-dependent. Instead, four
maintenance paths are force-invoked at scheduled moments using the
same production entry points Celery Beat would hit in real life:

* Day 15: ``run_reflector(...)`` with a canonical staleness reason.
* Day 18: ``_deliver_digest(digest_sub_id)`` while ``judge_digest`` is
  monkey-patched to return REVISE once, then PASS -- exercises the
  Writer -> Judge -> Writer -> Judge revision loop.
* Day 21: ``deliver_event_notifications_batch(...)`` while
  ``judge_batch_events`` is monkey-patched to force one REVISE cycle
  through the Assessor + Event Judge path.
* Day 24: ``run_event_verifier(...)`` with the event sub's current
  source contexts.

Each force-invocation runs the real ADK / LiteLLM call stack; the LLM
round-trip dollar cost is identical to the organic path. The only
thing bypassed is the production *condition* that would have triggered
that code path on its own.

Opt-in: this scenario hits the real LLM provider configured for the
benchmark, is slow, and costs real money. Run with::

    RUN_E2E_MONTH=1 uv run pytest \\
        benchmark/tests/integration/test_s_e2e_month.py -s

Budget guardrails abort the run if the ledger exceeds ``$30``.
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
from news_benchmark.fakes.search import SearchResult
from news_benchmark.scheduler import VirtualScheduler
from news_benchmark.simulator import run_one_turn
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
    MAX_ITEMS_PER_SOURCE_PER_DAY,
)

_CORPUS_DIR = Path(__file__).resolve().parents[2] / "data" / "corpus"

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_E2E_MONTH") != "1",
    reason=(
        "Month-long steady-state benchmark gated by RUN_E2E_MONTH=1 "
        "(hits the real LLM provider, costs real money, runs several minutes)."
    ),
)


_SIM_START = datetime(2026, 5, 1, tzinfo=UTC)
_SIM_DAYS = int(os.environ.get("E2E_SIM_DAYS", "30"))
_POLL_MINUTES = int(os.environ.get("E2E_POLL_MINUTES", "30"))

_DIGEST_WEBHOOK_URL = "https://bench.invalid/webhook/e2e-month-digest"
_EVENT_WEBHOOK_URL = "https://bench.invalid/webhook/e2e-month-event"

_DIGEST_TELEGRAM_HANDLE = "euenergynews"
_EVENT_TELEGRAM_HANDLE = "rareearthsalert"
_DIGEST_TELEGRAM_URL = f"https://t.me/s/{_DIGEST_TELEGRAM_HANDLE}"
_EVENT_TELEGRAM_URL = f"https://t.me/s/{_EVENT_TELEGRAM_HANDLE}"

_RESULTS_DIR = _ECONOMICS_DIR / "results"
_BUDGET_ABORT_USD = 30.0


_DIGEST_SOURCE_UNIVERSE: list[str] = [
    "https://www.euractiv.com/section/energy/feed/",
    "https://www.politico.eu/section/energy/feed/",
    "https://euobserver.com/feeds/energy.rss",
    "https://www.endseurope.com/rss/energy",
    "https://ec.europa.eu/commission/presscorner/rss/energy",
    "https://www.entsoe.eu/rss/news/",
    "https://www.acer.europa.eu/feeds/news.rss",
    "https://eur-lex.europa.eu/feeds/daily-summary-energy.rss",
]

_EVENT_SOURCE_UNIVERSE: list[str] = [
    "https://www.reuters.com/pf/api/v3/feed/metals",
    "https://www.argusmedia.com/rss/rare-earths",
    "https://www.fastmarkets.com/rss/rare-earths",
    "https://www.mining.com/tag/rare-earth/feed/",
]


_DIGEST_ONBOARDING_MESSAGE = (
    "Hi. I'm a policy analyst at a Brussels NGO. Set up a DAILY DIGEST "
    "subscription for EU energy and climate policy only: Council "
    "decisions, Commission proposals, ENTSO-E / ACER publications, "
    "EUR-Lex directives, Parliament ENVI and ITRE votes. Policy, not "
    "market prices. Skip EV sales, sports, celebrity and generic tech. "
    "English. Deliver at 08:00 UTC daily. Webhook URL "
    f"{_DIGEST_WEBHOOK_URL}. Proceed end-to-end: find sources and "
    "create the subscription now. Do not ask follow-up questions."
)

_EVENT_ONBOARDING_MESSAGE = (
    "I also work a commodities desk. Set up a SEPARATE EVENT-MODE "
    "subscription for rare-earth supply-chain alerts: export bans or "
    "quotas (dysprosium, neodymium, terbium, praseodymium), sanctions, "
    "mine or refinery closures, force-majeure declarations, stockpile "
    "and offtake announcements. Do NOT notify on routine price moves, "
    "opinion, or other commodities (lithium, cobalt, copper, nickel) "
    "unless rare-earth-linked. One alert per event. Webhook URL "
    f"{_EVENT_WEBHOOK_URL}. Find sources and create the subscription "
    "now. Do not ask follow-up questions."
)

_DIGEST_TRIGGER_NOW_MESSAGE = (
    "Deliver my EU energy digest right now even if the queue is empty. "
    "Do nothing else, do not ask questions."
)

_DIGEST_ADD_SOURCE_MESSAGE = (
    f"Attach the Telegram channel {_DIGEST_TELEGRAM_HANDLE} as an additional source "
    "for my EU energy digest. Do nothing else, do not ask questions."
)

_DIGEST_TRIGGER_DISCOVERY_MESSAGE = (
    "Find a few more sources for my EU energy digest -- Parliament and "
    "EUR-Lex coverage would help. Do nothing else."
)

_EVENT_ADD_SOURCE_MESSAGE = (
    f"Attach the Telegram channel {_EVENT_TELEGRAM_HANDLE} as an additional source "
    "for my rare-earth alerts. Do nothing else, do not ask questions."
)

_EVENT_REMOVE_SOURCE_MESSAGE = (
    f"Remove the Telegram channel {_EVENT_TELEGRAM_HANDLE} from my "
    "rare-earth alerts. Do nothing else, do not ask questions."
)

_EVENT_GET_SUBS_MESSAGE = "What subscriptions do I have? Do nothing else, do not ask questions."


def _items_mode() -> int:
    """Return AVG or MAX items-per-source-per-day based on ``E2E_ITEMS_MODE``."""
    mode = os.environ.get("E2E_ITEMS_MODE", "avg").strip().lower()
    if mode == "max":
        return MAX_ITEMS_PER_SOURCE_PER_DAY
    return AVG_ITEMS_PER_SOURCE_PER_DAY


def _load_search_corpus(path: Path) -> dict[str, list[SearchResult]]:
    """Convert the on-disk JSON corpus into the FakeSearch shape."""
    raw = json.loads(path.read_text())
    corpus: dict[str, list[SearchResult]] = {}
    for prefix, rows in raw.items():
        corpus[prefix] = [
            SearchResult(title=r["title"], url=r["url"], snippet=r["snippet"]) for r in rows
        ]
    return corpus


def _swap_search_corpus(world, topic: str) -> None:
    """Install a single-topic FakeSearch corpus so the Finder cannot cross-topic.

    The production FakeSearch falls back to any-token matching when no
    prefix startswith the query. If both topic corpora are loaded at
    once, generic tokens like ``RSS`` pull digest URLs into event
    Finder runs (and vice-versa). Clear and reload the one topic's
    corpus every time a Discovery-invoking operation is about to fire.
    """
    world.search.corpus.clear()
    if topic == "digest":
        world.search.corpus.update(_load_search_corpus(_CORPUS_DIR / "search_digest.json"))
    elif topic == "event":
        world.search.corpus.update(_load_search_corpus(_CORPUS_DIR / "search_event.json"))
    else:
        raise ValueError(f"unknown topic: {topic!r}")


def _register_telegram_adapter(
    *,
    world,
    url: str,
    topic: str,
    items_per_source_per_day: int,
) -> None:
    """Install a FakeAdapter at a Telegram channel URL so add_source accepts it.

    ``_validate_source_url`` in the fake checks that the URL's hostname
    has a registered adapter. Populating the adapter with a short
    timeline keeps the Finder's validator happy if it ever checks this
    source, and supplies real content once the channel is attached.
    """
    items = build_timeline(
        source_urls=[url],
        topic=topic,
        start=_SIM_START,
        days=_SIM_DAYS,
        items_per_source_per_day=items_per_source_per_day,
    )
    world.adapters[url] = FakeAdapter(source_url=url, items=items)


def _install_scenario_items(
    *,
    world,
    source_urls: list[str],
    topic: str,
    items_per_source_per_day: int,
) -> int:
    """Build and register scenario items for every ``source_urls`` entry.

    The timeline covers the full simulated window so the Finder's
    validator sees posts on every candidate when it scores them during
    onboarding.
    """
    items = build_timeline(
        source_urls=source_urls,
        topic=topic,
        start=_SIM_START,
        days=_SIM_DAYS,
        items_per_source_per_day=items_per_source_per_day,
    )
    by_source: dict[str, list[Any]] = {}
    for it in items:
        by_source.setdefault(it.source_url, []).append(it)

    for url in source_urls:
        world.adapters[url] = FakeAdapter(
            source_url=url, items=sorted(by_source.get(url, []), key=lambda x: x.fake_ts)
        )
    return sum(len(v) for v in by_source.values())


async def _seed_bare_user(user_id: uuid.UUID) -> None:
    """Insert a fresh ``User`` row the Conversational Agent will onboard."""
    from news_service.db.session import async_session_factory
    from news_service.models.user import User

    async with async_session_factory() as s:
        s.add(
            User(
                id=user_id,
                api_key=f"bench-e2e-month-{user_id.hex}",
                language="en",
                timezone="UTC",
                has_onboarded=False,
            )
        )
        await s.commit()


async def _active_subs(user_id: uuid.UUID) -> list[Any]:
    from news_service.db.session import async_session_factory
    from news_service.models.subscription import Subscription
    from sqlalchemy import select

    async with async_session_factory() as s:
        rows = await s.execute(
            select(Subscription).where(
                Subscription.user_id == user_id,
                Subscription.is_active.is_(True),
            )
        )
        return list(rows.scalars().all())


async def _pin_subscription_spec(
    sub_id: uuid.UUID,
    *,
    user_spec: str,
    retrieval_query: str,
    webhook_url: str,
    schedule_cron: str | None,
    digest_language: str,
) -> None:
    """Overwrite the Conv-Agent-authored spec with the pinned test text.

    The goal is steady-state cost measurement, not prose-quality
    measurement: pinning spec text keeps the Writer's / Assessor's
    downstream prompts byte-identical across runs.
    """
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.subscription import Subscription

    topic_embedding = await embed_text(retrieval_query)
    async with async_session_factory() as s:
        sub = await s.get(Subscription, sub_id)
        if sub is None:
            return
        sub.user_spec = user_spec
        sub.topic_embedding = topic_embedding
        sub.delivery_webhook_url = webhook_url
        sub.schedule_cron = schedule_cron
        sub.digest_language = digest_language
        await s.commit()


async def _scripted_turn(*, state, user_id: uuid.UUID, message: str) -> str:
    """Load the user inside a fresh session and drive one conv turn."""
    from news_service.db.session import async_session_factory
    from news_service.models.user import User

    async with async_session_factory() as s:
        user = await s.get(User, user_id)
        if user is None:
            raise RuntimeError(f"user {user_id} vanished between turns")
        return await run_one_turn(state=state, user=user, db_session=s, message=message)


def _schedule_cycles(
    *,
    sched: VirtualScheduler,
    poll_impl,
    digest_impl,
    verifier_impl,
    start: datetime,
    days: int,
    poll_minutes: int,
) -> None:
    """Pre-populate the heap with every poll, digest-cron and verifier tick."""
    poll_every = timedelta(minutes=poll_minutes)
    ticks_per_day = max(1, (24 * 60) // poll_minutes)
    for tick in range(days * ticks_per_day):
        sched.schedule(start + tick * poll_every, poll_impl, label=f"poll#{tick:04d}")
    for day in range(days):
        sched.schedule(
            start + timedelta(days=day, hours=8, minutes=1),
            digest_impl,
            label=f"digest-cron#{day:02d}",
        )
    for day in range(days):
        sched.schedule(
            start + timedelta(days=day, hours=1),
            verifier_impl,
            label=f"verifier-daily#{day:02d}",
        )


def _schedule_turn(
    *,
    sched: VirtualScheduler,
    when: datetime,
    label: str,
    coro_factory,
) -> None:
    """Schedule a single coroutine factory at a virtual instant."""

    async def _fire() -> None:
        await coro_factory()

    sched.schedule(when, _fire, label=label)


def _summarize_rows(
    rows: list[Any],
    *,
    digest_sub_id: str,
    event_sub_id: str,
) -> dict[str, Any]:
    """Aggregate ledger rows by sub, agent, call_type, and phase."""
    by_agent: dict[str, dict[str, float | int]] = {}
    by_call_type: dict[str, dict[str, float | int]] = {}
    by_sub: dict[str, float] = {
        digest_sub_id: 0.0,
        event_sub_id: 0.0,
        "unattributed": 0.0,
    }
    total_usd = 0.0
    total_prompt = 0
    total_completion = 0

    for r in rows:
        total_usd += r.usd_cost
        total_prompt += r.prompt_tokens
        total_completion += r.completion_tokens

        tag = r.agent_path or "<untagged>"
        a = by_agent.setdefault(tag, {"calls": 0, "usd": 0.0})
        a["calls"] = int(a["calls"]) + 1
        a["usd"] = float(a["usd"]) + r.usd_cost

        c = by_call_type.setdefault(r.call_type, {"calls": 0, "usd": 0.0})
        c["calls"] = int(c["calls"]) + 1
        c["usd"] = float(c["usd"]) + r.usd_cost

        sub = getattr(r, "subscription_id", None)
        if sub == digest_sub_id:
            by_sub[digest_sub_id] += r.usd_cost
        elif sub == event_sub_id:
            by_sub[event_sub_id] += r.usd_cost
        else:
            by_sub["unattributed"] += r.usd_cost

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
        "by_subscription": {k: round(v, 6) for k, v in by_sub.items()},
    }


def _yandex_aggregate(world) -> dict[str, Any]:
    """Count Yandex search calls and price them using the configured per-call rate."""
    from news_service.core.config import get_settings

    rate = float(get_settings().yandex_search_price_usd_per_call)
    count = len(world.search.call_log)
    return {"calls": count, "usd": round(count * rate, 6), "usd_per_call": rate}


@pytest.mark.asyncio
async def test_one_month_steady_state_cost(world, run_id: str) -> None:
    """Drive 30 simulated days through every cost-bearing production path.

    The one real user is onboarded twice (digest + event) via the
    Conversational Agent, lives through 30 days of polling, digest
    deliveries, event deliveries, a reflector staleness trigger, and
    the weekly verifier, and emits the measured per-subscription cost
    plus a phase / agent breakdown.
    """
    try:
        CLOCK.reset_to(_SIM_START)

        items_per_day = _items_mode()

        digest_items_total = _install_scenario_items(
            world=world,
            source_urls=_DIGEST_SOURCE_UNIVERSE,
            topic="digest",
            items_per_source_per_day=items_per_day,
        )
        event_items_total = _install_scenario_items(
            world=world,
            source_urls=_EVENT_SOURCE_UNIVERSE,
            topic="event",
            items_per_source_per_day=items_per_day,
        )
        _register_telegram_adapter(
            world=world,
            url=_DIGEST_TELEGRAM_URL,
            topic="digest",
            items_per_source_per_day=items_per_day,
        )
        _register_telegram_adapter(
            world=world,
            url=_EVENT_TELEGRAM_URL,
            topic="event",
            items_per_source_per_day=items_per_day,
        )

        user_id = uuid.uuid4()
        await _seed_bare_user(user_id)

        from news_service.schemas.conversation import ConversationState

        state = ConversationState(user_id=str(user_id), user_language="en")

        from news_service.tasks.deliver_digest import _deliver_digest  # noqa: F401
        from news_service.tasks.poll_feeds import _poll_all_feeds
        from news_service.tasks.reflect_events import _reflect_event_subscriptions
        from news_service.tasks.schedule_digests import _schedule_due_digests

        state_progress: dict[str, int] = {"last_day": -1, "poll_count": 0}

        def _log(msg: str) -> None:
            now = CLOCK.now()
            usd = LEDGER.total_usd()
            calls = len(LEDGER.rows())
            print(
                f"[{now.strftime('%Y-%m-%d %H:%M')}] calls={calls:<5d} usd=${usd:.4f}  {msg}",
                flush=True,
            )

        def _maybe_day_boundary() -> None:
            now = CLOCK.now()
            day_idx = (now - _SIM_START).days
            if day_idx != state_progress["last_day"]:
                state_progress["last_day"] = day_idx
                digest_cnt = len(world.delivery.for_url(_DIGEST_WEBHOOK_URL))
                event_cnt = len(world.delivery.for_url(_EVENT_WEBHOOK_URL))
                _log(
                    f"== day {day_idx:02d}/{_SIM_DAYS} ==  "
                    f"digest_webhooks={digest_cnt} event_webhooks={event_cnt} "
                    f"polls={state_progress['poll_count']}"
                )

        async def poll_tick() -> None:
            if LEDGER.total_usd() > _BUDGET_ABORT_USD:
                raise RuntimeError(
                    f"ledger total ${LEDGER.total_usd():.2f} exceeds "
                    f"guardrail ${_BUDGET_ABORT_USD:.2f}; aborting"
                )
            _maybe_day_boundary()
            state_progress["poll_count"] += 1
            await _poll_all_feeds()
            await world.celery.drain()

        async def digest_tick() -> None:
            _log("digest-cron: running _schedule_due_digests")
            before = LEDGER.total_usd()
            await _schedule_due_digests()
            await world.celery.drain()
            _log(f"digest-cron: done (delta=${LEDGER.total_usd() - before:.4f})")

        async def verifier_tick() -> None:
            before = LEDGER.total_usd()
            await _reflect_event_subscriptions()
            await world.celery.drain()
            delta = LEDGER.total_usd() - before
            if delta > 0.0001:
                _log(f"verifier-daily: ran (delta=${delta:.4f})")

        ledger_start = len(LEDGER.rows())

        sched = VirtualScheduler()
        _schedule_cycles(
            sched=sched,
            poll_impl=poll_tick,
            digest_impl=digest_tick,
            verifier_impl=verifier_tick,
            start=_SIM_START,
            days=_SIM_DAYS,
            poll_minutes=_POLL_MINUTES,
        )

        digest_sub_holder: dict[str, uuid.UUID] = {}
        event_sub_holder: dict[str, uuid.UUID] = {}

        async def _run_and_log_turn(label: str, message: str) -> None:
            _log(f"turn[{label}] begin")
            before = LEDGER.total_usd()
            reply = await _scripted_turn(state=state, user_id=user_id, message=message)
            await world.celery.drain()
            delta = LEDGER.total_usd() - before
            _log(f"turn[{label}] end (delta=${delta:.4f}) reply_preview={reply[:120]!r}")

        async def onboarding_digest_turn() -> None:
            _swap_search_corpus(world, "digest")
            await _run_and_log_turn("onboard_digest", _DIGEST_ONBOARDING_MESSAGE)
            subs = await _active_subs(user_id)
            digests = [s for s in subs if s.delivery_mode == "digest"]
            if not digests:
                raise RuntimeError("digest onboarding turn did not create an active subscription")
            digest_sub_holder["id"] = digests[0].id
            await _pin_subscription_spec(
                digests[0].id,
                user_spec=DIGEST_USER_SPEC,
                retrieval_query=DIGEST_RETRIEVAL_QUERY,
                webhook_url=_DIGEST_WEBHOOK_URL,
                schedule_cron="0 8 * * *",
                digest_language="en",
            )
            _log(f"digest sub created id={digests[0].id}")

        async def onboarding_event_turn() -> None:
            _swap_search_corpus(world, "event")
            await _run_and_log_turn("onboard_event", _EVENT_ONBOARDING_MESSAGE)
            subs = await _active_subs(user_id)
            events = [s for s in subs if s.delivery_mode == "event"]
            if not events:
                raise RuntimeError("event onboarding turn did not create an active subscription")
            event_sub_holder["id"] = events[0].id
            await _pin_subscription_spec(
                events[0].id,
                user_spec=EVENT_USER_SPEC,
                retrieval_query=EVENT_RETRIEVAL_QUERY,
                webhook_url=_EVENT_WEBHOOK_URL,
                schedule_cron=None,
                digest_language="en",
            )
            _log(f"event sub created id={events[0].id}")

        async def digest_trigger_now_turn() -> None:
            await _run_and_log_turn("digest_trigger_now", _DIGEST_TRIGGER_NOW_MESSAGE)

        async def digest_add_source_turn() -> None:
            await _run_and_log_turn("digest_add_source", _DIGEST_ADD_SOURCE_MESSAGE)

        async def digest_trigger_discovery_turn() -> None:
            _swap_search_corpus(world, "digest")
            await _run_and_log_turn("digest_trigger_discovery", _DIGEST_TRIGGER_DISCOVERY_MESSAGE)

        async def event_add_source_turn() -> None:
            await _run_and_log_turn("event_add_source", _EVENT_ADD_SOURCE_MESSAGE)

        async def event_remove_source_turn() -> None:
            await _run_and_log_turn("event_remove_source", _EVENT_REMOVE_SOURCE_MESSAGE)

        async def event_get_subs_turn() -> None:
            await _run_and_log_turn("event_get_subs", _EVENT_GET_SUBS_MESSAGE)

        # ------------------------------------------------------------------
        # Force-invocations for maintenance paths that may not fire organically
        # in 30 simulated days. Each one reuses the real production entry point
        # so the LLM round-trip cost is identical to an organic invocation.
        # ------------------------------------------------------------------

        async def force_reflector_run() -> None:
            """Force one Reflector ADK run by injecting a synthetic trigger reason.

            Monkey-patches ``_compute_reflect_reasons`` to return a canonical
            staleness reason once, then calls ``_deliver_digest`` so the
            digest pipeline invokes the Reflector with correctly-typed inputs
            (source_contexts loaded from DB, topic_embedding computed, etc.).
            """
            sub_id = digest_sub_holder.get("id")
            if sub_id is None:
                _log("force[reflector] skipped: digest sub not yet created")
                return
            from news_service.agents.digest import pipeline as pipeline_mod

            original_fn = pipeline_mod._compute_reflect_reasons
            call_count = {"n": 0}
            reason_text = (
                f"Forced staleness: source {_DIGEST_SOURCE_UNIVERSE[0]} has not "
                "published for 11 days (force-invocation; pricing-only)."
            )

            def _injector(**kwargs):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return [reason_text]
                return original_fn(**kwargs)

            pipeline_mod._compute_reflect_reasons = _injector  # type: ignore[assignment]
            _log("force[reflector] begin")
            before = LEDGER.total_usd()
            try:
                await _deliver_digest(sub_id)
                await world.celery.drain()
            finally:
                pipeline_mod._compute_reflect_reasons = original_fn  # type: ignore[assignment]
            _log(f"force[reflector] end (delta=${LEDGER.total_usd() - before:.4f})")

        async def force_digest_revise_run() -> None:
            """Force one Digest-Writer <-> Digest-Judge REVISE cycle.

            Patches ``judge_digest`` to return a REVISE verdict on its first
            call, then PASS on the second. The pipeline's own revision loop
            re-runs the Writer with the critic feedback, which reproduces the
            real REVISE cost profile (one extra writer + one extra judge
            call relative to a clean PASS).
            """
            sub_id = digest_sub_holder.get("id")
            if sub_id is None:
                _log("force[digest_revise] skipped: digest sub not yet created")
                return
            from news_service.agents.digest import pipeline as pipeline_mod
            from news_service.agents.digest.judge import QualityScores

            original_fn = pipeline_mod.judge_digest
            call_count = {"n": 0}

            async def _patched_judge(**kwargs):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    return QualityScores(
                        relevance=3,
                        format=2,
                        conciseness=3,
                        verdict="REVISE",
                        feedback=(
                            "Force-REVISE (pricing-only): tighten the introduction "
                            "and restructure the body as a bulleted policy rundown."
                        ),
                    )
                return await original_fn(**kwargs)

            pipeline_mod.judge_digest = _patched_judge  # type: ignore[assignment]
            _log("force[digest_revise] begin")
            before = LEDGER.total_usd()
            try:
                await _deliver_digest(sub_id)
                await world.celery.drain()
            finally:
                pipeline_mod.judge_digest = original_fn  # type: ignore[assignment]
            _log(f"force[digest_revise] end (delta=${LEDGER.total_usd() - before:.4f})")

        async def force_event_judge_revise_run() -> None:
            """Force one Event-Assessor <-> Event-Judge REVISE cycle.

            Picks a handful of recently-polled NewsItem ids belonging to the
            event sub's sources, patches ``judge_batch_events`` to flag every
            item REVISE on the first pass, and calls
            ``_deliver_event_notifications_batch(item_ids)`` directly so the
            assessor + judge REVISE loop fires with those items as input.
            """
            sub_id = event_sub_holder.get("id")
            if sub_id is None:
                _log("force[event_revise] skipped: event sub not yet created")
                return

            from news_service.db.session import async_session_factory
            from news_service.models.news_item import NewsItem
            from news_service.models.sent_item import SentItem
            from news_service.models.subscription_source import SubscriptionSource
            from sqlalchemy import select

            async with async_session_factory() as s:
                # Items already delivered for this sub live in SentItem;
                # _deliver_event_notifications_batch filters those out, which
                # would short-circuit before the assessor can fire. Pick the
                # most recent UNSENT items so the assessor (and hence the
                # patched judge) actually runs.
                sent_subquery = select(SentItem.news_item_id).where(
                    SentItem.subscription_id == sub_id
                )
                rows = await s.execute(
                    select(NewsItem.id)
                    .join(
                        SubscriptionSource,
                        SubscriptionSource.source_id == NewsItem.source_id,
                    )
                    .where(SubscriptionSource.subscription_id == sub_id)
                    .where(NewsItem.id.not_in(sent_subquery))
                    .order_by(NewsItem.published_at.desc())
                    .limit(5)
                )
                item_ids = [r[0] for r in rows.all()]
            if not item_ids:
                _log("force[event_revise] skipped: no pending event items in DB")
                return

            from news_service.agents.event.judge import BatchJudgeResult, ItemVerdict
            from news_service.tasks import deliver_events as dv_mod
            from news_service.tasks.deliver_events import _deliver_event_notifications_batch

            original_fn = dv_mod.judge_batch_events
            call_count = {"n": 0}

            async def _patched_event_judge(*, assessment, **kwargs):
                call_count["n"] += 1
                if call_count["n"] == 1:
                    per_item = [
                        ItemVerdict(
                            item_id=a.item_id,
                            verdict="REVISE",
                            feedback=(
                                "Force-REVISE (pricing-only): re-check that this "
                                "item is strictly a rare-earth supply-chain event, "
                                "not routine price movement."
                            ),
                        )
                        for a in assessment.assessments
                    ]
                    return BatchJudgeResult(per_item=per_item, overall="REVISE")
                return await original_fn(assessment=assessment, **kwargs)

            dv_mod.judge_batch_events = _patched_event_judge  # type: ignore[assignment]
            _log(f"force[event_revise] begin (forcing REVISE on {len(item_ids)} items)")
            before = LEDGER.total_usd()
            try:
                await _deliver_event_notifications_batch(item_ids)
                await world.celery.drain()
            finally:
                dv_mod.judge_batch_events = original_fn  # type: ignore[assignment]
            _log(f"force[event_revise] end (delta=${LEDGER.total_usd() - before:.4f})")

        async def force_verifier_run() -> None:
            """Force one Event Verifier ADK run by clearing ``last_reflected_at``.

            Resets the event sub's self-throttle field so
            ``_reflect_event_subscriptions`` treats the sub as due and
            invokes ``run_event_verifier`` with the production source
            context loader and web-search tool.
            """
            sub_id = event_sub_holder.get("id")
            if sub_id is None:
                _log("force[verifier] skipped: event sub not yet created")
                return

            from news_service.db.session import async_session_factory
            from news_service.models.subscription import Subscription
            from news_service.tasks.reflect_events import _reflect_event_subscriptions

            async with async_session_factory() as s:
                sub = await s.get(Subscription, sub_id)
                if sub is not None:
                    sub.last_reflected_at = None
                    await s.commit()

            _log("force[verifier] begin")
            before = LEDGER.total_usd()
            await _reflect_event_subscriptions()
            await world.celery.drain()
            _log(f"force[verifier] end (delta=${LEDGER.total_usd() - before:.4f})")

        def _clamp(days_offset: int) -> int:
            """Shift turns scheduled past the simulated window to the last day."""
            return min(days_offset, max(_SIM_DAYS - 1, 0))

        _schedule_turn(
            sched=sched,
            when=_SIM_START + timedelta(minutes=15),
            label="turn:onboard_digest",
            coro_factory=onboarding_digest_turn,
        )
        _schedule_turn(
            sched=sched,
            when=_SIM_START + timedelta(minutes=45),
            label="turn:onboard_event",
            coro_factory=onboarding_event_turn,
        )
        _schedule_turn(
            sched=sched,
            when=_SIM_START + timedelta(days=_clamp(3), hours=10),
            label="turn:digest_trigger_now",
            coro_factory=digest_trigger_now_turn,
        )
        _schedule_turn(
            sched=sched,
            when=_SIM_START + timedelta(days=_clamp(5), hours=14),
            label="turn:event_add_source",
            coro_factory=event_add_source_turn,
        )
        _schedule_turn(
            sched=sched,
            when=_SIM_START + timedelta(days=_clamp(10), hours=12),
            label="turn:digest_add_source",
            coro_factory=digest_add_source_turn,
        )
        _schedule_turn(
            sched=sched,
            when=_SIM_START + timedelta(days=_clamp(15), hours=11),
            label="turn:event_remove_source",
            coro_factory=event_remove_source_turn,
        )
        _schedule_turn(
            sched=sched,
            when=_SIM_START + timedelta(days=_clamp(20), hours=14),
            label="turn:digest_trigger_discovery",
            coro_factory=digest_trigger_discovery_turn,
        )
        _schedule_turn(
            sched=sched,
            when=_SIM_START + timedelta(days=_clamp(25), hours=16),
            label="turn:event_get_subs",
            coro_factory=event_get_subs_turn,
        )

        # Force-invocations for maintenance paths (deterministic pricing)
        _schedule_turn(
            sched=sched,
            when=_SIM_START + timedelta(days=_clamp(12), hours=9),
            label="force:reflector",
            coro_factory=force_reflector_run,
        )
        _schedule_turn(
            sched=sched,
            when=_SIM_START + timedelta(days=_clamp(18), hours=9, minutes=30),
            label="force:digest_revise",
            coro_factory=force_digest_revise_run,
        )
        _schedule_turn(
            sched=sched,
            when=_SIM_START + timedelta(days=_clamp(21), hours=9),
            label="force:event_judge_revise",
            coro_factory=force_event_judge_revise_run,
        )
        _schedule_turn(
            sched=sched,
            when=_SIM_START + timedelta(days=_clamp(24), hours=9),
            label="force:verifier",
            coro_factory=force_verifier_run,
        )

        end = _SIM_START + timedelta(days=_SIM_DAYS)
        await sched.run(until=end)

        new_rows = LEDGER.rows()[ledger_start:]
        digest_sub_id = str(digest_sub_holder.get("id", ""))
        event_sub_id = str(event_sub_holder.get("id", ""))
        summary = _summarize_rows(new_rows, digest_sub_id=digest_sub_id, event_sub_id=event_sub_id)
        summary["total_calls"] = len(new_rows)

        yandex = _yandex_aggregate(world)

        digest_delivered = len(world.delivery.for_url(_DIGEST_WEBHOOK_URL))
        event_delivered = len(world.delivery.for_url(_EVENT_WEBHOOK_URL))

        cost_digest = summary["by_subscription"].get(digest_sub_id, 0.0)
        cost_event = summary["by_subscription"].get(event_sub_id, 0.0)
        cost_unattributed = summary["by_subscription"].get("unattributed", 0.0)
        cost_total = round(cost_digest + cost_event + cost_unattributed + yandex["usd"], 6)

        result_payload = {
            "run_id": run_id,
            "items_mode": os.environ.get("E2E_ITEMS_MODE", "avg"),
            "items_per_source_per_day": items_per_day,
            "simulated_start": _SIM_START.isoformat(),
            "simulated_end": end.isoformat(),
            "simulated_days": _SIM_DAYS,
            "digest_sub_id": digest_sub_id,
            "event_sub_id": event_sub_id,
            "digest_source_universe": _DIGEST_SOURCE_UNIVERSE,
            "event_source_universe": _EVENT_SOURCE_UNIVERSE,
            "digest_items_injected": digest_items_total,
            "event_items_injected": event_items_total,
            "digest_deliveries": digest_delivered,
            "event_deliveries": event_delivered,
            "cost_digest_usd": round(cost_digest, 6),
            "cost_event_usd": round(cost_event, 6),
            "cost_unattributed_usd": round(cost_unattributed, 6),
            "cost_yandex_usd": yandex["usd"],
            "cost_total_usd": cost_total,
            "yandex": yandex,
            "cost_summary": summary,
            "ledger_rows": [asdict(r) for r in new_rows],
        }

        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = _RESULTS_DIR / f"e2e_month_v2_{run_id}.json"
        out_path.write_text(json.dumps(result_payload, indent=2, default=str))

        print(f"\n[e2e-month-v2 {run_id}] mode={os.environ.get('E2E_ITEMS_MODE', 'avg')}")
        print(f"[e2e-month-v2 {run_id}] wrote={out_path}")
        print(
            f"[e2e-month-v2 {run_id}] deliveries digest={digest_delivered} event={event_delivered}"
        )
        print(
            f"[e2e-month-v2 {run_id}] cost_digest_usd={cost_digest:.6f} "
            f"cost_event_usd={cost_event:.6f} "
            f"cost_unattributed_usd={cost_unattributed:.6f} "
            f"cost_yandex_usd={yandex['usd']:.6f} "
            f"cost_total_usd={cost_total:.6f}"
        )
        for agent, stats in sorted(summary["by_agent"].items(), key=lambda kv: -kv[1]["usd"]):
            print(
                f"[e2e-month-v2 {run_id}]   {agent:<20s} "
                f"calls={stats['calls']:>5d} usd={stats['usd']:.6f}"
            )

        assert summary["total_calls"] > 0, "ledger captured no LLM calls -- harness wired wrong"
        assert digest_sub_id, "digest onboarding never produced a subscription id"
        assert event_sub_id, "event onboarding never produced a subscription id"
        assert digest_delivered >= 1, (
            f"expected at least one digest delivery across {_SIM_DAYS} days, got {digest_delivered}"
        )
    finally:
        pass
