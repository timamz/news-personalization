"""
S-verifier: coverage-gap path -- verifier catches a missed event the
linked source NEVER covered and queues source discovery alongside the
catch-up delivery.

Seeds one event-mode subscription narrowly scoped to EU CBAM (Carbon
Border Adjustment Mechanism) developments. One auto-discovered source
is linked; it has been polling and has notified on two recent
non-CBAM EU energy items (forming a non-empty notification history).
A separate CBAM Council announcement from two days ago is planted
only in the fake web-search corpus -- none of the subscription's
sources covered it, so the batch assessor had no chance of
surfacing it.

This is the **coverage-gap** scenario: the correct verifier response
is BOTH to emit_missed_event (so the task delivers a catch-up) AND to
trigger_source_discovery (because the user needs a better source for
this topic going forward). The companion test
``test_s_verifier_assessor_miss.py`` covers the opposite case, where
a linked source DID cover the event and the assessor under-flagged
it -- in that case only catch-up delivery should fire and discovery
should stay silent.

``_reflect_event_subscriptions`` is called directly. The real Event
Verifier ADK agent runs, issues a few web searches (budget 5), finds
the CBAM miss in the fake corpus, calls ``fetch_source_items`` on the
linked source to confirm the source did not cover it, then calls
``emit_missed_event`` AND ``trigger_source_discovery``. The
surrounding task inserts a synthetic NewsItem + SentItem, delivers a
catch-up webhook, and dispatches the discovery task via
``celery_app.send_task``.

The fake web-search corpus is shaped to match the query patterns the
prompt's guidelines ("include entity names, official announcement,
date ranges") plus a broad fallback prefix whose token set covers
most reasonable phrasings, so the test does not depend on the
agent choosing an exact wording.

Exercises: ``_reflect_event_subscriptions``, ``run_event_verifier``
(the ADK agent loop with all five tools, including
``trigger_source_discovery``), ``_deliver_and_record_miss``
(synthetic NewsItem + SentItem + webhook), discovery-task dispatch
via ``celery_app.send_task``, the verifier sentinel source, and
``Subscription.last_reflected_at`` stamping.

Out of scope: the "no miss found" negative case and self-throttling
via ``last_reflected_at`` (we initialise it to NULL so the sub is
due).
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select

from news_benchmark.clock import CLOCK
from news_benchmark.fakes.search import SearchResult

from tests.integration._reflector_common import install_discovery_stub

WEBHOOK_URL = "https://bench.invalid/webhook/s-verifier"
SOURCE_URL = "https://brussels-energy-policy.invalid/feed.xml"


USER_SPEC = (
    "# CBAM alerts\n"
    "\n"
    "Notify me immediately when an official EU Carbon Border Adjustment "
    "Mechanism (CBAM) development happens: Council decisions, Commission "
    "implementing or delegated acts, Parliament votes on CBAM files, and "
    "formal guidance notes. Include the institutional source URL.\n"
    "\n"
    "Do NOT notify me about general climate-policy news, opinion pieces, "
    "industry lobbying responses, or anything that isn't a concrete CBAM "
    "regulatory step."
)


def _build_search_results(now):
    """Build the fake-search results keyed to the FakeClock's current time.

    ``MISSED_EVENT`` lands two days before ``now`` so it sits inside the
    7-day verifier lookback window; distractors are generic and
    undated. Hardcoded dates would break the test the moment the
    clock advances.
    """
    miss_date = now - timedelta(days=2)
    date_iso = miss_date.strftime("%Y-%m-%d")
    date_human = miss_date.strftime("%d %B %Y")

    missed_event = SearchResult(
        title=(
            f"Council of the EU adopts CBAM implementing regulation "
            f"on {miss_date.strftime('%d %b %Y')}"
        ),
        url=(
            "https://www.consilium.europa.eu/en/press/press-releases/"
            f"{date_iso}/cbam-implementing-regulation"
        ),
        snippet=(
            f"The Council of the European Union on {date_human} formally "
            "adopted the implementing regulation for the Carbon Border "
            "Adjustment Mechanism. The regulation covers cement, iron "
            "and steel, aluminium, fertilisers, electricity, and hydrogen "
            "imports. Effective application from Q1 of the following year."
        ),
    )

    distractors = [
        SearchResult(
            title="Industry groups urge CBAM simplification ahead of rollout",
            url="https://www.example-industry.invalid/cbam-simplification-call",
            snippet=(
                "European industry associations are pressing for reporting "
                "simplifications ahead of the CBAM transitional phase. The "
                "letter, signed by nineteen trade bodies, asks the "
                "Commission to reduce quarterly paperwork."
            ),
        ),
        SearchResult(
            title="Explainer: how CBAM will work when it enters full force",
            url="https://www.example-explainer.invalid/cbam-how-it-works",
            snippet=(
                "A primer on the EU's Carbon Border Adjustment Mechanism, "
                "covering scope, default emission factors, and the "
                "interaction with the EU Emissions Trading System."
            ),
        ),
        SearchResult(
            title="EU considers broader methane import standards",
            url="https://www.example-methane.invalid/methane-import",
            snippet=(
                "A separate policy file on methane intensity for imported "
                "LNG is moving through the Council. Environmental groups "
                "have called the timetable ambitious."
            ),
        ),
    ]

    return [missed_event, *distractors]


def _build_search_corpus(now) -> dict[str, list[SearchResult]]:
    """Register the same result list under five prefixes so the fake-search
    matcher lands it for any reasonable verifier query shape.

    Primary-prefix matches: ``cbam``, ``carbon``, ``eu``, ``european``,
    ``council``. The final long prefix covers the fallback matcher
    (any whitespace-split token of the prefix is in the query), so
    creative wordings like "what happened with CBAM this week" also
    hit.
    """
    results = _build_search_results(now)
    return {
        "cbam": results,
        "carbon": results,
        "eu": results,
        "european": results,
        "council": results,
        "cbam carbon border adjustment mechanism eu council commission": results,
    }


# Two on-topic-but-non-CBAM items that were already notified on. Their
# existence proves the source is polling and the assessor is running;
# the miss is about CBAM specifically, which none of these cover.
SEEDED_NOTIFIED_ITEMS = [
    {
        "headline": "ENVI committee tightens methane-leak limits for imported LNG",
        "body": (
            "The European Parliament's ENVI committee voted on Thursday to "
            "tighten methane-intensity limits for imported liquefied natural "
            "gas. The amendment extends the Methane Regulation's import "
            "standard to the full upstream value chain."
        ),
    },
    {
        "headline": "Council approves 40 percent renewable electricity target for 2030",
        "body": (
            "The Council of the European Union endorsed a binding 40% "
            "renewable electricity target for the EU-27 by 2030, up from "
            "the current 32%. Trilogues with Parliament are expected in May."
        ),
    },
]


@pytest.mark.asyncio
async def test_s_verifier_coverage_gap_delivers_catchup_and_queues_discovery(world):
    """Verifier finds a CBAM miss with zero source coverage: catch-up + discovery."""
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.failed_task import FailedTask
    from news_service.models.news_item import NewsItem
    from news_service.models.sent_item import SentItem
    from news_service.models.source import Source
    from news_service.models.subscription import Subscription
    from news_service.models.subscription_source import SubscriptionSource
    from news_service.models.user import User

    now = CLOCK.now()
    world.search.corpus.update(_build_search_corpus(now))

    discovery_stub = install_discovery_stub(world)

    user_id = uuid.uuid4()
    sub_id = uuid.uuid4()
    source_id = uuid.uuid4()

    topic_embedding = await embed_text(
        "EU CBAM Carbon Border Adjustment Mechanism Council Commission"
    )

    async with async_session_factory() as s:
        s.add(
            User(
                id=user_id,
                api_key=f"bench-{user_id.hex}",
                language="en",
                timezone="UTC",
                delivery_webhook_url=WEBHOOK_URL,
                has_onboarded=True,
            )
        )
        s.add(
            Source(
                id=source_id,
                url=SOURCE_URL,
                title="Brussels Energy Policy Tracker",
                source_description="EU-level energy and climate policy newswire.",
                subscriber_count=1,
            )
        )
        s.add(
            Subscription(
                id=sub_id,
                user_id=user_id,
                user_spec=USER_SPEC,
                delivery_mode="event",
                digest_language="en",
                delivery_webhook_url=WEBHOOK_URL,
                topic_embedding=topic_embedding,
                is_active=True,
                last_reflected_at=None,
            )
        )
        s.add(
            SubscriptionSource(
                subscription_id=sub_id,
                source_id=source_id,
                is_user_specified=False,
            )
        )

        # Seed notified items a few days before the miss. Pre-backend-fix,
        # the agent inferred "today" from the latest notification and
        # rejected the 2-day-old miss as "future relative to history".
        # Now that the backend prompt includes "Current date/time", the
        # agent has explicit anchoring and these dates no longer need
        # to straddle the miss.
        for idx, item in enumerate(SEEDED_NOTIFIED_ITEMS):
            item_id = uuid.uuid4()
            item_embedding = await embed_text(item["headline"] + "\n" + item["body"][:400])
            item_ts = now - timedelta(days=4 + idx)
            s.add(
                NewsItem(
                    id=item_id,
                    source_id=source_id,
                    headline=item["headline"],
                    body=item["body"],
                    url=f"{SOURCE_URL.rstrip('/')}/seeded-{idx:02d}",
                    source="Brussels Energy Policy Tracker",
                    published_at=item_ts,
                    fetched_at=item_ts,
                    embedding=item_embedding,
                )
            )
            s.add(
                SentItem(
                    subscription_id=sub_id,
                    news_item_id=item_id,
                    sent_at=item_ts,
                )
            )

        await s.commit()

    pre_existing_sent_count = len(SEEDED_NOTIFIED_ITEMS)

    from news_service.tasks.reflect_events import _reflect_event_subscriptions

    result = await _reflect_event_subscriptions()

    await world.celery.drain()

    assert result.get("status") == "done", f"expected status=done, got {result!r}"
    assert result.get("processed", 0) >= 1, (
        f"expected at least one subscription processed, got {result!r}"
    )
    assert result.get("delivered_misses", 0) >= 1, (
        f"expected at least 1 delivered miss, got {result!r}. "
        f"Verifier searches issued: {world.search.call_log!r}"
    )

    assert world.search.call_log, (
        "expected verifier to issue at least one web_search call; call_log is empty."
    )

    captured = world.delivery.for_url(WEBHOOK_URL)
    assert len(captured) == 1, (
        f"expected exactly 1 catch-up webhook for {WEBHOOK_URL}, got {len(captured)}. "
        f"Bodies: {[c.body[:160] for c in captured]}"
    )

    combined = (captured[0].body + " " + (captured[0].subject or "")).lower()
    assert "cbam" in combined or "carbon border" in combined, (
        f"catch-up body should mention CBAM / carbon border. Got: {captured[0].body!r}"
    )

    assert discovery_stub.call_count() >= 1, (
        "coverage-gap scenario: the linked source never covered CBAM, so the "
        "verifier should have queued source discovery alongside the catch-up. "
        f"discovery_stub.calls={discovery_stub.calls!r}. "
        f"Verifier searches issued: {world.search.call_log!r}"
    )

    async with async_session_factory() as s:
        sent_rows = list(
            (await s.execute(select(SentItem).where(SentItem.subscription_id == sub_id)))
            .scalars()
            .all()
        )
    assert len(sent_rows) == pre_existing_sent_count + 1, (
        f"expected {pre_existing_sent_count + 1} SentItem rows after catch-up "
        f"(pre-existing {pre_existing_sent_count} + 1 verifier miss), got "
        f"{len(sent_rows)}"
    )

    async with async_session_factory() as s:
        refreshed = await s.get(Subscription, sub_id)
    assert refreshed is not None
    assert refreshed.last_reflected_at is not None, (
        "expected subscription.last_reflected_at to be stamped after the verifier run"
    )

    async with async_session_factory() as s:
        failed = list((await s.execute(select(FailedTask))).scalars().all())
    assert not failed, f"expected 0 failed_tasks rows, got {len(failed)}: {failed!r}"
