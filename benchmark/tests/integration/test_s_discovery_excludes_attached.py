"""
S-discovery-excludes-attached: the discovery pipeline filters already-attached
sources out of the candidate pool BEFORE the LLM ever gets a chance to pick them.

Scope
-----
Exercises ``news_service.tasks.discover_sources.run_and_persist_discovery``
end-to-end against a subscription that already has TWO auto-discovered
sources attached (simulating a reflector asking for MORE coverage). The fake
web-search deliberately RE-SURFACES one of those attached URLs alongside
two genuinely new candidates, using a compelling title and snippet so an
LLM without the mechanical filter would plausibly pick it. The attached
URL's FakeAdapter is also seeded with on-topic posts so the validator,
if it ran, would score it as a strong candidate.

What this proves
----------------
  * The attached URL never enters the candidate pool. The guard lives
    upstream in ``agents/source_discovery/pipeline.py`` at the top of
    ``spawn_finder`` (``exclude_urls`` -> ``_normalize_url`` dedupe) and
    in ``agents/source_discovery/finder.py`` ``validate_and_score_source``.
  * Discovery attaches ONLY new, non-attached URLs as
    ``SubscriptionSource`` rows.
  * No duplicate ``SubscriptionSource`` row is created for the already-
    attached source; the count of rows pointing at the re-surfaced URL
    stays at exactly 1.

Out of scope
------------
LLM judgment (we're NOT asking the agent to reject the attached URL on
quality grounds); exact wording of spawn_finder strategies; the number
of new sources the agent picks (1 or 2 is acceptable); and the Discovery
Agent's inspect_source path.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from sqlalchemy import select

ATTACHED_URL_1 = "https://acer-decisions.invalid/rss"
ATTACHED_URL_2 = "https://entso-e-news.invalid/feed"
FRESH_URL_1 = "https://brussels-energy-policy.invalid/feed.xml"
FRESH_URL_2 = "https://eu-energy-watch.invalid/atom.xml"


@pytest.mark.asyncio
async def test_s_discovery_does_not_reattach_already_attached_source_even_when_resurfaced(
    world,
):
    """Already-attached source is filtered out upstream; no duplicate link created."""
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

    now = datetime(2026, 4, 20, tzinfo=UTC)

    enticing_results = [
        SearchResult(
            title="ACER Official Decisions RSS",
            url=ATTACHED_URL_1,
            snippet=(
                "Primary source of ACER decisions on EU-wide energy regulation. "
                "Daily updates on cross-border interconnector rulings, gas market "
                "rules, and ENTSO-E coordination."
            ),
        ),
        SearchResult(
            title="Brussels Energy Policy Tracker",
            url=FRESH_URL_1,
            snippet=(
                "Tracker for EU Council decisions, Commission energy directives, "
                "and carbon pricing rules. Updated weekdays."
            ),
        ),
        SearchResult(
            title="EU Energy Watch (Atom feed)",
            url=FRESH_URL_2,
            snippet=(
                "Weekly digest covering gas market rules, interconnector "
                "decisions, and ACER-ENTSO-E coordination across EU member states."
            ),
        ),
    ]
    for prefix in [
        "eu energy policy",
        "best eu energy policy feeds",
        "list of rss feeds for eu energy policy",
        "eu energy policy rss",
        "best rss feeds eu energy policy",
        "top eu energy policy sources",
        "eu energy regulation",
        "european union energy policy",
        "acer entso-e",
        "best rss feeds",
        "curated eu energy policy",
    ]:
        world.search.corpus[prefix] = list(enticing_results)

    on_topic_posts = [
        ScenarioItem(
            fake_ts=now,
            source_url=ATTACHED_URL_1,
            headline="ACER ruling on cross-border interconnector capacity",
            body=(
                "The European Union Agency for the Cooperation of Energy "
                "Regulators (ACER) adopted a new methodology for cross-border "
                "interconnector capacity allocation across EU member states."
            ),
        ),
        ScenarioItem(
            fake_ts=now,
            source_url=ATTACHED_URL_1,
            headline="Commission energy directive on carbon pricing",
            body=(
                "The European Commission today proposed a revision to the "
                "carbon pricing framework targeting EU-wide electricity markets "
                "and harmonising Emissions Trading System coverage."
            ),
        ),
        ScenarioItem(
            fake_ts=now,
            source_url=ATTACHED_URL_1,
            headline="ENTSO-E winter outlook for EU electricity markets",
            body=(
                "ENTSO-E published its winter outlook assessing supply risks "
                "across EU electricity markets, coordinating with ACER on gas "
                "market rules and interconnector capacity."
            ),
        ),
    ]
    world.adapters[ATTACHED_URL_1] = FakeAdapter(
        source_url=ATTACHED_URL_1,
        items=list(on_topic_posts),
    )
    world.adapters[FRESH_URL_1] = FakeAdapter(
        source_url=FRESH_URL_1,
        items=[
            ScenarioItem(
                fake_ts=now,
                source_url=FRESH_URL_1,
                headline="EU Council adopts revised energy market directive",
                body=(
                    "The Council of the European Union adopted the revised "
                    "electricity market directive covering capacity mechanisms "
                    "and cross-border trade within the EU internal energy market."
                ),
            ),
            ScenarioItem(
                fake_ts=now,
                source_url=FRESH_URL_1,
                headline="Commission proposes new rules for EU carbon pricing",
                body=(
                    "The European Commission unveiled a proposal to tighten "
                    "EU-wide carbon pricing rules under the Emissions Trading "
                    "System, with direct implications for energy producers."
                ),
            ),
            ScenarioItem(
                fake_ts=now,
                source_url=FRESH_URL_1,
                headline="ACER opinion on national regulatory frameworks",
                body=(
                    "ACER issued an opinion on national regulatory frameworks "
                    "for electricity and gas networks, urging greater harmonisation "
                    "across EU member state regulators."
                ),
            ),
        ],
    )
    world.adapters[FRESH_URL_2] = FakeAdapter(
        source_url=FRESH_URL_2,
        items=[
            ScenarioItem(
                fake_ts=now,
                source_url=FRESH_URL_2,
                headline="ENTSO-E publishes gas market interconnector report",
                body=(
                    "ENTSO-E published a report on interconnector usage across "
                    "EU gas markets, analysing ACER-coordinated rulings on "
                    "network codes and balancing rules."
                ),
            ),
            ScenarioItem(
                fake_ts=now,
                source_url=FRESH_URL_2,
                headline="EU Council reaches deal on renewable targets",
                body=(
                    "EU energy ministers reached a political agreement on the "
                    "revised renewable energy directive, raising the bloc-wide "
                    "renewable target for 2030."
                ),
            ),
            ScenarioItem(
                fake_ts=now,
                source_url=FRESH_URL_2,
                headline="Commission flags grid investment gap",
                body=(
                    "The European Commission flagged a significant grid "
                    "investment gap across the EU, urging member states to "
                    "accelerate cross-border transmission projects."
                ),
            ),
        ],
    )

    user_id = uuid.uuid4()
    sub_id = uuid.uuid4()
    attached_source_1_id = uuid.uuid4()
    attached_source_2_id = uuid.uuid4()
    webhook_url = "https://bench.invalid/webhook/s-discovery-excludes-attached"

    user_spec = (
        "# EU energy policy digest\n\n"
        "I want news about European Union energy regulation, especially\n"
        "Council decisions, Commission proposals, and ACER / ENTSO-E\n"
        "announcements. Keep it policy-flavoured; skip purely national\n"
        "political coverage and retail-market price reporting.\n"
        "\n"
        "Present as a short bulletised digest in English.\n"
    )
    retrieval_query = (
        "EU energy policy Council Commission ACER ENTSO-E regulation"
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
        s.add(
            Source(
                id=attached_source_1_id,
                url=ATTACHED_URL_1,
                title="ACER Official Decisions RSS",
                source_description="ACER decisions on EU-wide energy regulation.",
                is_active=True,
            )
        )
        s.add(
            Source(
                id=attached_source_2_id,
                url=ATTACHED_URL_2,
                title="ENTSO-E News",
                source_description="ENTSO-E bulletins on EU electricity markets.",
                is_active=True,
            )
        )
        s.add(
            SubscriptionSource(
                subscription_id=sub_id,
                source_id=attached_source_1_id,
                is_user_specified=False,
            )
        )
        s.add(
            SubscriptionSource(
                subscription_id=sub_id,
                source_id=attached_source_2_id,
                is_user_specified=False,
            )
        )
        await s.commit()

    async with async_session_factory() as s:
        result = await run_and_persist_discovery(
            s,
            sub_id,
            reason=(
                "Reflector asked for more EU energy policy sources; current "
                "coverage of Commission proposals and Council decisions is thin."
            ),
        )

    assert result.get("status") in {"ok", "no_sources_found"}, (
        "discovery must return a known status; got unexpected payload: "
        f"{result!r}"
    )

    async with async_session_factory() as s:
        link_rows = await s.execute(
            select(SubscriptionSource, Source)
            .join(Source, Source.id == SubscriptionSource.source_id)
            .where(SubscriptionSource.subscription_id == sub_id)
        )
        rows = list(link_rows.all())

    all_urls = [src.url for _, src in rows]
    count_attached_1 = sum(1 for u in all_urls if u == ATTACHED_URL_1)
    count_attached_2 = sum(1 for u in all_urls if u == ATTACHED_URL_2)

    assert count_attached_1 == 1, (
        "re-surfaced already-attached URL must never be re-linked; expected "
        f"exactly 1 SubscriptionSource row for {ATTACHED_URL_1!r}, got "
        f"{count_attached_1}. Full url list: {all_urls!r}. Discovery "
        f"payload: {result!r}"
    )
    assert count_attached_2 == 1, (
        "pre-existing attached URL must not be duplicated; expected exactly "
        f"1 SubscriptionSource row for {ATTACHED_URL_2!r}, got "
        f"{count_attached_2}. Full url list: {all_urls!r}. Discovery "
        f"payload: {result!r}"
    )

    new_urls = [u for u in all_urls if u not in {ATTACHED_URL_1, ATTACHED_URL_2}]
    for new_url in new_urls:
        assert new_url in {FRESH_URL_1, FRESH_URL_2}, (
            "discovery attached a URL we did not seed into the search corpus; "
            f"unexpected url {new_url!r}. Full url list: {all_urls!r}. "
            f"Discovery payload: {result!r}"
        )

    async with async_session_factory() as s:
        failed_rows = await s.execute(select(FailedTask))
        failed = list(failed_rows.scalars().all())
    assert failed == [], (
        "discovery run must not produce any FailedTask rows; "
        f"got {len(failed)}: {failed!r}. Discovery payload: {result!r}"
    )
