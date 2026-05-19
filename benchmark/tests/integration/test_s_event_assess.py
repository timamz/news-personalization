"""
S-event-assess: event-pipeline smoke test.

One poll cycle pushes four items into one event-mode subscription. Two
items are clearly on-topic (lithium supply chain), two are clearly
off-topic (EV sales, copper). Asserts that exactly the two on-topic
items become webhook deliveries and ``SentItem`` rows; the two
off-topic ones do not.

Exercises: ``poll_feeds._poll_all_feeds`` (fetch + embed + upsert),
``CeleryShim`` inline dispatch of ``deliver_event_notifications_batch``,
``assess_batch_events`` (batch assessor), ``judge_batch_events``
(event judge -- PASS path, no revision needed), ``deliver`` (webhook
capture via FakeDelivery), and ``SentItem`` bookkeeping.

Out of scope: the assessor/judge REVISE loop, cross-batch dedup, and
the conversational agent (S-conv proved that path).
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select

from news_benchmark.clock import CLOCK
from news_benchmark.fakes.adapters import FakeAdapter, ScenarioItem

SOURCE_URL = "https://example-nova-metals.invalid/feed.xml"
WEBHOOK_URL = "https://bench.invalid/webhook/s-event-assess"

USER_SPEC = (
    "# Lithium supply-chain alerts\n"
    "\n"
    "Notify me instantly when news breaks about lithium mining, lithium "
    "refining, battery-grade lithium carbonate or hydroxide pricing, or "
    "regulatory actions affecting the lithium supply chain (mining permits, "
    "export restrictions, royalty changes, pricing floors).\n"
    "\n"
    "Do NOT notify me about: electric-vehicle sales or delivery numbers, "
    "Tesla stock moves, downstream battery-cell or pack news, or other "
    "metals (copper, nickel, cobalt) unless the story is explicitly about "
    "their interaction with lithium supply.\n"
)


@pytest.mark.asyncio
async def test_s_event_assess_relevant_vs_irrelevant(world):
    """Four items in, two on-topic deliveries out."""
    from news_service.db.session import async_session_factory
    from news_service.models.failed_task import FailedTask
    from news_service.models.news_item import NewsItem
    from news_service.models.sent_item import SentItem
    from news_service.models.source import Source
    from news_service.models.subscription import Subscription
    from news_service.models.subscription_source import SubscriptionSource
    from news_service.models.user import User

    user_id = uuid.uuid4()
    sub_id = uuid.uuid4()
    source_id = uuid.uuid4()

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
                title="Nova Metals Daily",
                source_description="Metals supply-chain newswire.",
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
                is_active=True,
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

    now = CLOCK.now()
    items = [
        ScenarioItem(
            fake_ts=now - timedelta(hours=2),
            source_url=SOURCE_URL,
            headline="Chile ministry unveils lithium pricing floor for 2026 contracts",
            body=(
                "Chile's economy ministry published draft regulations that set a "
                "minimum floor price for battery-grade lithium carbonate sold "
                "under long-term contracts starting January 2026. The measure is "
                "aimed at protecting state royalties as spot prices recover from "
                "the 2024 trough. Industry consultations close in May. "
                "SQM and Albemarle, the two largest lithium producers in Chile, "
                "said they are reviewing the draft text."
            ),
        ),
        ScenarioItem(
            fake_ts=now - timedelta(hours=4),
            source_url=SOURCE_URL,
            headline="Albemarle pauses expansion of Kemerton lithium hydroxide refinery",
            body=(
                "Albemarle Corp said on Monday it will pause the third-train "
                "expansion of its Kemerton lithium hydroxide refinery in Western "
                "Australia, citing weaker-than-expected demand for battery-grade "
                "lithium hydroxide through 2026. Commissioned capacity from "
                "trains 1 and 2 remains on stream; the company declined to give "
                "a restart date for the paused expansion."
            ),
        ),
        ScenarioItem(
            fake_ts=now - timedelta(hours=6),
            source_url=SOURCE_URL,
            headline="Tesla reports 22% Q1 delivery growth as Model Y refresh lands",
            body=(
                "Tesla Inc delivered 480,309 vehicles in the first quarter of "
                "2026, a 22% year-on-year increase, led by Model Y refreshes in "
                "North America and Europe. The company maintained its full-year "
                "guidance of 2.2 million vehicles and reiterated that margin "
                "pressure should ease in the second half."
            ),
        ),
        ScenarioItem(
            fake_ts=now - timedelta(hours=8),
            source_url=SOURCE_URL,
            headline="Copper futures hit three-month high on Peru strike fears",
            body=(
                "London copper futures rose 2.1% on Monday to their highest level "
                "in three months after workers at Peru's Las Bambas mine voted to "
                "strike over wage negotiations. The mine accounts for roughly 2% "
                "of global copper supply. Peruvian officials said they were "
                "encouraging both sides to return to the negotiating table."
            ),
        ),
    ]

    world.adapters[SOURCE_URL] = FakeAdapter(source_url=SOURCE_URL, items=items)

    from news_service.tasks.poll_feeds import _poll_all_feeds

    result = await _poll_all_feeds()
    assert result["new_items"] == 4, (
        f"expected 4 new items ingested, got {result['new_items']}: {result!r}"
    )
    assert result["event_notifications_queued"] == 4, (
        f"expected 4 event notifications queued, got "
        f"{result['event_notifications_queued']}: {result!r}"
    )

    await world.celery.drain()

    async with async_session_factory() as s:
        news_rows = (
            (await s.execute(select(NewsItem).where(NewsItem.source_id == source_id)))
            .scalars()
            .all()
        )
    assert len(news_rows) == 4, (
        f"expected 4 news_items persisted, got {len(news_rows)}"
    )

    async with async_session_factory() as s:
        sent_rows = list(
            (await s.execute(select(SentItem).where(SentItem.subscription_id == sub_id)))
            .scalars()
            .all()
        )

    captured = world.delivery.for_url(WEBHOOK_URL)
    captured_bodies_snip = [c.body[:160] for c in captured]

    assert len(sent_rows) == len(captured), (
        f"SentItem count ({len(sent_rows)}) should match webhook delivery count "
        f"({len(captured)}). Bodies: {captured_bodies_snip}"
    )

    assert 1 <= len(captured) <= 2, (
        f"expected 1 or 2 on-topic deliveries, got {len(captured)}. "
        f"Bodies: {captured_bodies_snip}"
    )

    combined = " ".join(cap.body.lower() for cap in captured)
    assert "lithium" in combined, (
        "expected at least one delivery to mention 'lithium' (both on-topic items do). "
        f"Bodies: {captured_bodies_snip}"
    )
    assert "tesla" not in combined, (
        f"unexpected off-topic Tesla delivery. Bodies: {captured_bodies_snip}"
    )
    assert "copper" not in combined, (
        f"unexpected off-topic copper delivery. Bodies: {captured_bodies_snip}"
    )

    async with async_session_factory() as s:
        failed = list((await s.execute(select(FailedTask))).scalars().all())
    assert not failed, f"expected 0 failed_tasks rows, got {len(failed)}: {failed!r}"
