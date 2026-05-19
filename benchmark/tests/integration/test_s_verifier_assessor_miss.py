"""
S-verifier (assessor-miss twin): source covered the event but the
assessor never notified; the correct verifier routing is catch-up
delivered AND discovery NOT queued.

This is the twin of ``test_s_verifier_coverage_gap.py`` (the former
``test_s_verifier.py``). The seeding -- user, source, subscription,
subscription_source, fake-search corpus, and the two seeded notified
items -- is copied verbatim. The only difference is a third NewsItem
planted on the linked source whose headline, body, and timestamp
mirror the CBAM Council adoption in the fake-search corpus. There
is no corresponding SentItem: the assessor had the item available
but failed to flag it.

The Event Verifier prompt tells the agent to call
``fetch_source_items`` on each linked source for every candidate
miss and use the result to distinguish:

- "source did not cover it" -> coverage gap, queue discovery.
- "source covered it but we didn't notify" -> assessor miss,
  deliver catch-up only.

This test exercises the second branch. We therefore assert:

- catch-up webhook is delivered (the user was never notified);
- discovery is NOT queued (the source already covers the topic,
  better sources would not help).

Out of scope: self-throttling via ``last_reflected_at``, and the
positive "discovery queued" routing which the sibling test
``test_s_verifier_coverage_gap.py`` covers.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select

from news_benchmark.clock import CLOCK
from news_benchmark.fakes.search import SearchResult
from tests.integration._reflector_common import install_discovery_stub

WEBHOOK_URL = "https://bench.invalid/webhook/s-verifier-assessor-miss"
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
async def test_s_verifier_assessor_miss_delivers_catchup_without_discovery(world):
    """Source covered the CBAM event but no SentItem exists: catch-up only, no discovery."""
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

        # Seeded notified items: prove the source polls and the
        # assessor is running. Same as the coverage-gap sibling.
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

        # THIRD item -- the assessor-miss plant. Same institution, same
        # date, same event as the fake-search missed_event. Published
        # two days before ``now`` to match the corpus entry. NO
        # SentItem is inserted for it: the linked source ingested the
        # CBAM adoption but the assessor never flagged it, so the user
        # was never notified. The verifier's fetch_source_items call
        # will return this row, letting the agent see that coverage
        # was fine and only the assessor failed.
        cbam_item_date = now - timedelta(days=2)
        cbam_headline = (
            f"Council of the EU adopts CBAM implementing regulation on "
            f"{cbam_item_date.strftime('%d %b %Y')}"
        )
        cbam_body = (
            f"The Council of the European Union on "
            f"{cbam_item_date.strftime('%d %B %Y')} formally adopted the "
            "implementing regulation for the Carbon Border Adjustment "
            "Mechanism. The regulation covers cement, iron and steel, "
            "aluminium, fertilisers, electricity, and hydrogen imports. "
            "Effective application from Q1 of the following year."
        )
        cbam_embedding = await embed_text(cbam_headline + "\n" + cbam_body[:400])
        s.add(
            NewsItem(
                id=uuid.uuid4(),
                source_id=source_id,
                headline=cbam_headline,
                body=cbam_body,
                url=f"{SOURCE_URL.rstrip('/')}/cbam-council-adoption",
                source="Brussels Energy Policy Tracker",
                published_at=cbam_item_date,
                fetched_at=cbam_item_date,
                embedding=cbam_embedding,
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
        f"expected at least 1 delivered miss (user was never notified), got "
        f"{result!r}. Verifier searches issued: {world.search.call_log!r}. "
        f"Discovery stub calls: {discovery_stub.calls!r}"
    )

    assert discovery_stub.call_count() == 0, (
        f"expected 0 discovery triggers -- source covered the event, no "
        f"coverage gap -- got {discovery_stub.call_count()}: "
        f"{discovery_stub.calls!r}. Verifier searches: {world.search.call_log!r}"
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
