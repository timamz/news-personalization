"""
S-discovery-reuses-existing-source: pgvector dedupe / Source reuse check.

Scope
-----
Covers the path where a ``Source`` row is already persisted in the DB with
a ``source_description_embedding`` close to a new subscription's topic
embedding. When the Discovery pipeline runs for the new subscription, the
Source Finder's ``search_existing_sources`` pgvector cosine query (or, as
a fallback, a web-search result that happens to point at the same URL)
must find the pre-existing source. The orchestrator submits it,
``ensure_source_by_url`` looks it up by URL in the ``sources`` table,
finds the existing row, increments ``subscriber_count``, and does NOT
create a duplicate ``Source`` row -- it only creates a new
``SubscriptionSource`` link to the existing source for the new sub.

This guards a specific correctness invariant of the ingest architecture:
``sources.url`` is unique and is the canonical dedupe key. Discovering a
semantically-matching URL already known to the system must reuse it, not
fork a parallel row.

Assertions (one logical claim folded into a single assert)
----------------------------------------------------------
  * pipeline returned ``status=ok``,
  * exactly one ``Source`` row exists with the pre-seeded URL (no duplicate),
  * exactly one ``SubscriptionSource`` link exists for the new sub pointing
    at the pre-seeded source id,
  * the pre-seeded source's ``subscriber_count`` was incremented past 0,
  * zero ``FailedTask`` rows.

Out of scope
------------
Scoring, orchestrator strategy choice, whether the discovery agent went
via ``search_existing_sources`` or via ``tool_search_web`` -- either path
is fine. The assertion is about post-run DB state.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from sqlalchemy import func, select

EXISTING_URL = "https://brussels-energy-policy.invalid/feed.xml"

EXISTING_SOURCE_TITLE = "Brussels Energy Policy Tracker"

EXISTING_SOURCE_DESCRIPTION = (
    "RSS feed covering EU energy policy: Council of the European Union decisions, "
    "European Commission proposals, ACER and ENTSO-E regulatory publications, "
    "EUR-Lex directives, Parliament ENVI and ITRE committee votes, methane "
    "regulation, gas storage rules, renewable targets, LNG import standards, "
    "electricity balancing network codes, cross-border interconnector financing."
)

RETRIEVAL_QUERY = (
    "EU energy policy news: Council of the European Union decisions, European "
    "Commission proposals, ACER and ENTSO-E regulatory publications, EUR-Lex "
    "directives, Parliament ENVI and ITRE committee votes, methane regulation, "
    "gas storage rules, renewable targets, LNG import standards, electricity "
    "balancing network codes, cross-border interconnector financing."
)

USER_SPEC = (
    "# EU energy policy news\n"
    "\n"
    "I want English-language coverage of EU-level energy and climate policy: "
    "Council decisions, Commission proposals, ACER and ENTSO-E regulatory "
    "publications, EUR-Lex directives, and Parliament committee votes "
    "(ENVI, ITRE). Focus on policy and regulation, not retail market prices. "
    "Prefer RSS-style feeds. Skip EV-sales, stock coverage, and national "
    "political horse-trading.\n"
)


EXISTING_SOURCE_POSTS: list[tuple[str, str]] = [
    (
        "Council of the EU adopts emergency gas storage directive for 2027 winter",
        (
            "The Council of the European Union formally adopted a directive requiring "
            "Member States to fill underground gas storage to 90 percent of capacity "
            "by 1 November each year starting in 2027. The text replaces the temporary "
            "regulation expiring at the end of 2026 and makes the storage obligation "
            "permanent under EU energy policy."
        ),
    ),
    (
        "Commission tables proposal on cross-border interconnector financing",
        (
            "The European Commission tabled a draft regulation on joint financing for "
            "cross-border electricity interconnectors. The proposal touches on ACER's "
            "oversight role and the network code on capacity allocation. It will go "
            "through the ordinary legislative procedure with Council and Parliament."
        ),
    ),
    (
        "Parliament ITRE committee endorses tighter methane import standard",
        (
            "The European Parliament's ITRE committee endorsed a tightened methane- "
            "intensity standard for imported LNG, extending the Methane Regulation's "
            "scope upstream. The trilogue with Council is expected within weeks under "
            "the EU energy policy legislative agenda."
        ),
    ),
]


FALLBACK_URL = "https://eu-energy-watch.invalid/atom.xml"

FALLBACK_POSTS: list[tuple[str, str]] = [
    (
        "EUR-Lex publishes directive on accelerated offshore wind permitting",
        (
            "A new directive accelerating permitting procedures for offshore wind "
            "installations was published on EUR-Lex and enters force in 20 days. The "
            "text caps permitting timelines inside 'renewable acceleration areas' at "
            "24 months, one of the flagship files of EU energy policy this session."
        ),
    ),
    (
        "Commission unveils 40 percent renewable electricity target for 2030",
        (
            "The European Commission unveiled a proposal to set a binding 40 percent "
            "renewable electricity target for the EU-27 by 2030, up from 32 percent in "
            "the current Renewable Energy Directive. The file will go through Council "
            "and Parliament under the EU energy policy legislative agenda."
        ),
    ),
]


def _seed_fake_search(world) -> None:
    """Populate the fake search corpus with results pointing mainly at EXISTING_URL.

    The production Source Finder is instructed to call ``search_existing_sources``
    first, but if it opts for web search instead we still want the agent to see
    ``EXISTING_URL`` as the most enticing candidate. We therefore seed a broad
    set of curator-query prefixes with EXISTING_URL listed first and FALLBACK_URL
    second, so validation + selection converge on the pre-existing source.
    """
    from news_benchmark.fakes.search import SearchResult

    rows = [
        SearchResult(
            title=EXISTING_SOURCE_TITLE,
            url=EXISTING_URL,
            snippet=(
                "RSS feed covering EU energy policy: Council decisions, Commission "
                "proposals, ACER and ENTSO-E announcements, EUR-Lex directives."
            ),
        ),
        SearchResult(
            title="EU Energy Watch",
            url=FALLBACK_URL,
            snippet=(
                "Independent watch on EU energy policy: Commission directives, "
                "Parliament ENVI and ITRE committee votes, methane regulation, "
                "LNG import rules, renewable targets."
            ),
        ),
    ]
    corpus_keys = [
        "eu energy policy rss feeds",
        "best rss feeds for eu energy policy",
        "european energy regulation rss",
        "best energy policy feeds",
        "acer entso-e news",
        "eu electricity market feeds",
        "european commission energy news rss",
        "best rss feeds",
        "list of rss feeds for eu energy policy",
        "awesome eu energy sources",
        "top rss feeds european energy",
        "best energy news rss",
        "best eu energy sources",
        "eu energy",
        "energy policy rss",
        "european energy news feeds",
        "top energy policy sources",
        "best european energy policy news sources",
    ]
    for key in corpus_keys:
        world.search.corpus[key] = rows


def _seed_fake_adapters(world) -> None:
    """Install FakeAdapters for EXISTING_URL and FALLBACK_URL so validation scores.

    ``validate_and_score_source`` pulls real posts through ``fetch_source_posts``,
    which the world harness routes to ``FakeAdapter`` by URL. Each candidate
    URL gets 2-3 topic-dense posts so the finder can validate and score them.
    """
    from news_benchmark.clock import CLOCK
    from news_benchmark.fakes.adapters import FakeAdapter, ScenarioItem

    now = CLOCK.now()

    for idx, (url, posts) in enumerate(
        ((EXISTING_URL, EXISTING_SOURCE_POSTS), (FALLBACK_URL, FALLBACK_POSTS))
    ):
        items: list[ScenarioItem] = []
        for post_idx, (headline, body) in enumerate(posts):
            items.append(
                ScenarioItem(
                    fake_ts=now - timedelta(hours=6 + idx * 5 + post_idx * 3),
                    source_url=url,
                    headline=headline,
                    body=body,
                )
            )
        world.adapters[url] = FakeAdapter(
            source_url=url,
            items=sorted(items, key=lambda x: x.fake_ts),
        )
        for item in items:
            synth_url = str(item.to_normalized()["url"])
            world.article_fetch.bodies[synth_url] = item.body


@pytest.mark.asyncio
async def test_s_discovery_reuses_existing_source_without_duplicating_source_row(world):
    """Discovery reuses a pre-existing Source row instead of creating a duplicate."""
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.failed_task import FailedTask
    from news_service.models.source import Source
    from news_service.models.subscription import Subscription
    from news_service.models.subscription_source import SubscriptionSource
    from news_service.models.user import User
    from news_service.tasks.discover_sources import run_and_persist_discovery

    _seed_fake_search(world)
    _seed_fake_adapters(world)

    existing_source_id = uuid.uuid4()
    user_id = uuid.uuid4()
    sub_id = uuid.uuid4()

    existing_description_embedding = await embed_text(EXISTING_SOURCE_DESCRIPTION)
    topic_embedding = await embed_text(RETRIEVAL_QUERY)

    async with async_session_factory() as s:
        pre_source_count = (await s.execute(select(func.count(Source.id)))).scalar_one()

    async with async_session_factory() as s:
        s.add(
            Source(
                id=existing_source_id,
                url=EXISTING_URL,
                title=EXISTING_SOURCE_TITLE,
                source_description=EXISTING_SOURCE_DESCRIPTION,
                source_description_embedding=existing_description_embedding,
                is_active=True,
                subscriber_count=0,
            )
        )
        s.add(
            User(
                id=user_id,
                api_key=f"bench-{user_id.hex}",
                language="en",
                timezone="UTC",
                has_onboarded=True,
            )
        )
        s.add(
            Subscription(
                id=sub_id,
                user_id=user_id,
                user_spec=USER_SPEC,
                topic_embedding=topic_embedding,
                delivery_mode="digest",
                schedule_cron="0 9 * * *",
                digest_language="en",
                is_active=True,
            )
        )
        await s.commit()

    async with async_session_factory() as s:
        result = await run_and_persist_discovery(
            s, sub_id, reason="initial-discovery-for-test"
        )

    async with async_session_factory() as s:
        sources_with_existing_url = (
            await s.execute(
                select(func.count(Source.id)).where(Source.url == EXISTING_URL)
            )
        ).scalar_one()

        link_rows = list(
            (
                await s.execute(
                    select(SubscriptionSource).where(
                        SubscriptionSource.subscription_id == sub_id,
                        SubscriptionSource.source_id == existing_source_id,
                    )
                )
            ).scalars().all()
        )

        refreshed_existing = (
            await s.execute(select(Source).where(Source.id == existing_source_id))
        ).scalar_one_or_none()

        failed = list((await s.execute(select(FailedTask))).scalars().all())

    existing_source_is_linked = len(link_rows) == 1
    source_row_is_unique = sources_with_existing_url == 1
    subscriber_count_incremented = (
        refreshed_existing is not None and refreshed_existing.subscriber_count >= 1
    )

    reuse_succeeded = (
        result.get("status") == "ok"
        and existing_source_is_linked
        and source_row_is_unique
        and subscriber_count_incremented
        and not failed
    )
    assert reuse_succeeded, (
        "discovery did not reuse the pre-existing Source row: expected "
        "status=ok, exactly one SubscriptionSource link from the new sub to the "
        "pre-seeded source id, exactly one Source row with the pre-seeded URL "
        "(no duplicates), subscriber_count incremented past 0, and zero "
        "FailedTask rows. "
        f"Got result={result!r}; "
        f"pre_source_count={pre_source_count!r}; "
        f"sources_with_existing_url={sources_with_existing_url!r}; "
        f"link_rows_count={len(link_rows)!r}; "
        f"refreshed_existing_subscriber_count="
        f"{(refreshed_existing.subscriber_count if refreshed_existing else None)!r}; "
        f"failed_tasks={failed!r}."
    )
