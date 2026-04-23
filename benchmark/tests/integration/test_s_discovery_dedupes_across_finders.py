"""
S-discovery-dedupes-across-finders: the Discovery Agent's candidate pool
deduplicates URLs returned under different surface forms by parallel
Source Finders.

Scope
-----
Exercises ``news_service.agents.source_discovery.pipeline.run_source_discovery``
end-to-end through ``run_and_persist_discovery``. The fake web-search
corpus is rigged so that multiple distinct strategy keyword prefixes all
surface the SAME canonical source, but under slightly different surface
forms -- trailing slash plus uppercase hostname -- mimicking the kind of
duplication real Finders produce when they hit mirror pages, cached
directories, or URLs copy-pasted from different curator lists.

The pipeline normalises candidate URLs via ``_normalize_url`` (strip
trailing slash, lowercase) before inserting into ``candidate_pool``, so
even if two Finders score different surface forms of the same endpoint
the pool collapses them to one entry. Downstream, ``ensure_source_by_url``
persists exactly one ``Source`` row and ``run_and_persist_discovery``
inserts exactly one ``SubscriptionSource`` link for that subscription.

Assertions (one logical claim: the URL-normalisation dedupe works end-to-
end from parallel Finders to the subscription_sources table)
  - discovery status is ok
  - exactly one Source row exists whose canonical URL matches either
    surface form (case-insensitive, trailing-slash-insensitive)
  - exactly one SubscriptionSource row exists for this subscription
    pointing at that canonical URL
  - zero FailedTask rows (no tool-arg hallucinations, no tier-1 crash)

Out of scope
------------
Whether the agent also selected other candidates, which Finder strategies
the LLM happened to pick, and the specific surface form that survived
dedup. The pool may legitimately contain extra unrelated candidates.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import func, or_, select


@pytest.mark.asyncio
async def test_s_discovery_dedupes_same_source_across_parallel_finders(world):
    """Two Finder strategies hit the same URL in different surface forms; one row lands."""
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.failed_task import FailedTask
    from news_service.models.source import Source
    from news_service.models.subscription import Subscription
    from news_service.models.subscription_source import SubscriptionSource
    from news_service.models.user import User
    from news_service.tasks.discover_sources import run_and_persist_discovery

    from news_benchmark.fakes.adapters import FakeAdapter, ScenarioItem
    from news_benchmark.fakes.search import SearchResult

    canonical_url = "https://brussels-energy-policy.invalid/feed.xml"
    same_url_v1 = "https://brussels-energy-policy.invalid/feed.xml"
    same_url_v2 = "https://BRUSSELS-ENERGY-POLICY.invalid/feed.xml/"

    from datetime import UTC, datetime, timedelta

    now = datetime(2026, 3, 1, tzinfo=UTC)
    adapter_items = [
        ScenarioItem(
            fake_ts=now - timedelta(days=2),
            source_url=canonical_url,
            headline="ACER adopts new transmission-tariff guideline",
            body=(
                "The Agency for the Cooperation of Energy Regulators approved "
                "a revised tariff methodology for cross-border electricity "
                "transmission, aligning with Commission proposals on "
                "European energy policy and the next Council review."
            ),
        ),
        ScenarioItem(
            fake_ts=now - timedelta(days=1),
            source_url=canonical_url,
            headline="Commission proposes sharper gas-storage targets for 2027",
            body=(
                "Brussels energy policy update: the European Commission "
                "presented a revised directive on minimum gas storage "
                "obligations for member states, with ENTSO-E modelling "
                "support and Council debate scheduled next week."
            ),
        ),
        ScenarioItem(
            fake_ts=now - timedelta(hours=6),
            source_url=canonical_url,
            headline="ENTSO-E publishes winter adequacy outlook",
            body=(
                "ENTSO-E's winter adequacy report covers EU electricity "
                "security and flags grid-tightness risks for the Iberian "
                "peninsula, informing ACER and Commission decisions on "
                "European energy policy reserves."
            ),
        ),
    ]
    world.adapters[canonical_url] = FakeAdapter(
        source_url=canonical_url, items=adapter_items
    )
    for item in adapter_items:
        synth_url = item.to_normalized()["url"]
        world.article_fetch.bodies[str(synth_url)] = item.body

    world.search.corpus.update(
        {
            "best eu energy policy rss feeds": [
                SearchResult(
                    title="Brussels Energy Policy Tracker (official feed)",
                    url=same_url_v1,
                    snippet=(
                        "Primary RSS feed covering EU energy directives, "
                        "ACER decisions, Commission proposals, and ENTSO-E "
                        "outlooks. Recommended by every European energy "
                        "policy listicle."
                    ),
                ),
                SearchResult(
                    title="Awesome EU Energy Policy Sources (GitHub)",
                    url=same_url_v1,
                    snippet=(
                        "Curated list of RSS feeds for EU energy regulation, "
                        "starting with Brussels Energy Policy Tracker."
                    ),
                ),
            ],
            "european commission energy directives rss": [
                SearchResult(
                    title="Brussels Energy Policy Tracker (mirror)",
                    url=same_url_v2,
                    snippet=(
                        "Mirror of the official EU energy directives feed; "
                        "tracks Commission proposals and Council outcomes."
                    ),
                ),
                SearchResult(
                    title="Top EU Energy Directive Feeds 2026",
                    url=same_url_v2,
                    snippet=(
                        "Listicle of European Commission energy-directive "
                        "feeds; Brussels Energy Policy Tracker leads the list."
                    ),
                ),
            ],
            "acer entso-e news rss feeds": [
                SearchResult(
                    title="Brussels Energy Policy Tracker",
                    url=same_url_v2,
                    snippet=(
                        "ACER and ENTSO-E coverage, plus EU Council energy "
                        "policy outcomes."
                    ),
                ),
            ],
            "best rss feeds eu energy regulation": [
                SearchResult(
                    title="Brussels Energy Policy Tracker (primary)",
                    url=same_url_v1,
                    snippet=(
                        "Covers ACER, ENTSO-E and Commission energy policy "
                        "announcements."
                    ),
                ),
            ],
            "telegram channels eu energy policy": [
                SearchResult(
                    title="Brussels Energy Policy Tracker web-feed",
                    url=same_url_v2,
                    snippet=(
                        "No Telegram presence; use the RSS endpoint from "
                        "Brussels Energy Policy Tracker instead."
                    ),
                ),
            ],
            "european council energy policy news feed": [
                SearchResult(
                    title="Brussels Energy Policy Tracker (Council coverage)",
                    url=same_url_v1,
                    snippet=(
                        "EU Council energy-policy outcomes and Commission "
                        "communications."
                    ),
                ),
            ],
        }
    )

    user_id = uuid.uuid4()
    sub_id = uuid.uuid4()
    webhook_url = "https://bench.invalid/webhook/s-discovery-dedupe"

    user_spec = (
        "# EU energy policy digest\n\n"
        "I want news strictly about European Union energy regulation:\n"
        "Council decisions, Commission proposals, ACER rulings and\n"
        "ENTSO-E publications. Skip purely national political coverage\n"
        "and retail-market price reporting.\n"
        "\n"
        "Present as a short bulletised digest in English.\n"
    )
    retrieval_query = (
        "European Union energy policy Commission Council ACER ENTSO-E directives"
    )
    topic_vector = await embed_text(retrieval_query)

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
                topic_embedding=topic_vector,
                user_spec=user_spec,
                delivery_mode="digest",
                schedule_cron="0 8 * * *",
                digest_language="en",
                delivery_webhook_url=webhook_url,
                is_active=True,
            )
        )
        await s.commit()

    async with async_session_factory() as s:
        result = await run_and_persist_discovery(
            s,
            sub_id,
            reason="initial-discovery-for-test",
        )

    assert result.get("status") == "ok", (
        "discovery did not succeed; the rigged corpus should have produced "
        f"at least one scored candidate. Got payload={result!r}"
    )

    canonical_norm = canonical_url.rstrip("/").lower()

    async with async_session_factory() as s:
        source_count_row = await s.execute(
            select(func.count(Source.id)).where(
                or_(
                    Source.url == same_url_v1,
                    Source.url == same_url_v2,
                    func.lower(func.rtrim(Source.url, "/")) == canonical_norm,
                )
            )
        )
        source_count = source_count_row.scalar_one()

        link_count_row = await s.execute(
            select(func.count(SubscriptionSource.subscription_id))
            .join(Source, Source.id == SubscriptionSource.source_id)
            .where(
                SubscriptionSource.subscription_id == sub_id,
                or_(
                    Source.url == same_url_v1,
                    Source.url == same_url_v2,
                    func.lower(func.rtrim(Source.url, "/")) == canonical_norm,
                ),
            )
        )
        link_count = link_count_row.scalar_one()

        matching_rows = await s.execute(
            select(Source.url)
            .join(
                SubscriptionSource, SubscriptionSource.source_id == Source.id
            )
            .where(
                SubscriptionSource.subscription_id == sub_id,
                or_(
                    Source.url == same_url_v1,
                    Source.url == same_url_v2,
                    func.lower(func.rtrim(Source.url, "/")) == canonical_norm,
                ),
            )
        )
        matching_urls = [row[0] for row in matching_rows.all()]

    assert source_count == 1, (
        "expected exactly one Source row whose URL matches the canonical "
        "Brussels Energy Policy Tracker endpoint (case-insensitive, "
        f"trailing-slash-insensitive); got {source_count} rows matching. "
        f"URLs seen via the subscription: {matching_urls!r}. "
        f"Discovery payload: {result!r}"
    )

    assert link_count == 1, (
        "expected exactly one SubscriptionSource row for this subscription "
        "pointing at the canonical Brussels Energy Policy Tracker URL; "
        f"got {link_count} rows. The Discovery Agent's candidate pool "
        "failed to deduplicate distinct surface forms returned by parallel "
        f"Finders. Matching URLs: {matching_urls!r}. "
        f"Discovery payload: {result!r}"
    )

    async with async_session_factory() as s:
        failed_rows = await s.execute(select(FailedTask))
        failed = list(failed_rows.scalars().all())
    assert failed == [], (
        "dedupe test must not produce any FailedTask rows; "
        f"got {len(failed)}: {failed!r}"
    )
