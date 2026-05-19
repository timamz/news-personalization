"""
S-discovery-inspect-then-submit: prove the Discovery Agent uses inspect_source.

Scope
-----
The Discovery Agent prompt instructs the orchestrator to call
``inspect_source(url)`` when the cosine score alone is not enough to
decide on a candidate (borderline scores, suspicious titles, possible
off-topic feeds). This benchmark integration test seeds the orchestrator
pool with three candidates -- two that are clearly strong (tight
on-topic titles, dense on-topic snippets and post bodies) and one that
is genuinely borderline (ambiguous name, generic snippet, posts that mix
EU energy with broader finance/geopolitics). We then run
``run_and_persist_discovery`` end-to-end with the real Discovery Agent
LLM and assert that ``inspect_source`` was invoked at least once.

How the spy works
-----------------
``inspect_source`` is a closure defined inside ``run_source_discovery``
and is only observable via its one external dependency:
``news_service.agents.source_discovery.pipeline.fetch_source_posts``.
The ``world`` fixture already replaces that module attribute with its
fake. We wrap that fake again for the duration of the test and record
every call. This specifically captures orchestrator calls because the
finder's ``validate_and_score_source`` uses
``news_service.services.relevance.fetch_source_posts``, which is a
distinct module-level reference.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select

STRONG_URL_1 = "https://brussels-energy-policy.invalid/feed.xml"
STRONG_URL_2 = "https://acer-decisions.invalid/rss"
BORDERLINE_URL = "https://energydaily-markets.invalid/feed"

CANDIDATE_URLS = [STRONG_URL_1, STRONG_URL_2, BORDERLINE_URL]

CANDIDATE_TITLES = {
    STRONG_URL_1: "Brussels Energy Policy Tracker",
    STRONG_URL_2: "ACER Decisions Wire",
    BORDERLINE_URL: "EnergyDaily: Markets & Policy",
}

CANDIDATE_SNIPPETS = {
    STRONG_URL_1: (
        "RSS feed covering EU energy policy: Council decisions, Commission proposals, "
        "ACER and ENTSO-E announcements, EUR-Lex directives on gas and electricity markets."
    ),
    STRONG_URL_2: (
        "Official RSS for the EU Agency for the Cooperation of Energy Regulators (ACER): "
        "network codes, cross-border balancing rules, interconnector decisions."
    ),
    BORDERLINE_URL: (
        "Daily commentary on energy markets, regulation, and geopolitics. Covers oil, "
        "gas, power, and commodities with a global lens."
    ),
}


CANDIDATE_POSTS: dict[str, list[tuple[str, str]]] = {
    STRONG_URL_1: [
        (
            "Council of the EU adopts emergency gas storage directive for 2027 winter",
            (
                "The Council of the European Union formally adopted a directive requiring "
                "Member States to fill underground gas storage to 90 percent of capacity by "
                "1 November each year starting in 2027. The text replaces the temporary "
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
                "The European Parliament's ITRE committee endorsed a tightened methane-intensity "
                "standard for imported LNG, extending the Methane Regulation's scope upstream. "
                "The trilogue with Council is expected within weeks under the EU energy policy "
                "legislative agenda."
            ),
        ),
    ],
    STRONG_URL_2: [
        (
            "ACER publishes final network code on electricity balancing markets",
            (
                "The EU Agency for the Cooperation of Energy Regulators (ACER) published the "
                "final text of the amended network code on electricity balancing. The code "
                "harmonises cross-border balancing platforms and tightens imbalance-settlement "
                "timelines across European energy regulators."
            ),
        ),
        (
            "ACER decision on interconnector capacity allocation for Iberian peninsula",
            (
                "ACER issued a binding decision on capacity allocation across the Spain-France "
                "interconnector. The decision clarifies how transmission system operators must "
                "implement the EU electricity market regulation and align with ENTSO-E methods."
            ),
        ),
        (
            "ACER consultation opens on gas tariff harmonisation under NC TAR",
            (
                "ACER opened a public consultation on updates to the network code on tariff "
                "structures for gas transmission (NC TAR). The consultation addresses "
                "cost-allocation principles and cross-border gas market rules under EU energy "
                "regulation."
            ),
        ),
    ],
    BORDERLINE_URL: [
        (
            "Brent slips below 78 as OPEC+ signals patience on output cuts",
            (
                "Brent crude drifted below 78 dollars per barrel after OPEC+ delegates signaled "
                "they are in no hurry to unwind voluntary production cuts. Traders are watching "
                "Chinese refinery runs and US shale breakeven costs. Analysts at several banks "
                "revised fourth-quarter price forecasts in both directions. The piece closes "
                "with a brief note on EU gas storage levels heading into shoulder season."
            ),
        ),
        (
            "Middle East tensions keep a geopolitical premium in LNG spot cargoes",
            (
                "Spot LNG cargoes in Asia and Europe carry a geopolitical premium as tensions "
                "in the Middle East persist. The commentary reviews shipping routes, insurance "
                "rates, and how traders are hedging with TTF futures. A closing paragraph "
                "mentions that Brussels has floated tighter methane-intensity rules for imported "
                "cargoes but does not dive into the regulatory text."
            ),
        ),
        (
            "Week ahead: central banks, earnings, and a smattering of energy data",
            (
                "This week's agenda includes Fed and ECB speakers, tech earnings, and US crude "
                "and natural gas inventories. The newsletter flags a few EU regulatory items on "
                "the calendar -- a Commission energy update, a Parliament committee vote -- but "
                "focuses mostly on rates, equities, and FX positioning going into month-end."
            ),
        ),
    ],
}


USER_SPEC = (
    "# EU energy policy news\n"
    "\n"
    "I want English-language coverage of EU-level energy and climate policy: "
    "Council decisions, Commission proposals, ACER and ENTSO-E regulatory "
    "publications, EUR-Lex directives, and Parliament committee votes "
    "(ENVI, ITRE). Focus on policy and regulation, not retail market prices "
    "or commodity trading commentary. Prefer RSS-style feeds. Skip EV-sales, "
    "stock coverage, and national political horse-trading.\n"
)

RETRIEVAL_QUERY = (
    "EU energy policy: Council decisions, Commission proposals, ACER, ENTSO-E, "
    "EUR-Lex directives, methane regulation, gas storage, renewable targets, "
    "LNG import rules, electricity balancing network code"
)


def _build_search_result(url: str):
    from news_benchmark.fakes.search import SearchResult

    return SearchResult(
        title=CANDIDATE_TITLES[url],
        url=url,
        snippet=CANDIDATE_SNIPPETS[url],
    )


def _seed_search_corpus(world) -> None:
    rows = [_build_search_result(url) for url in CANDIDATE_URLS]
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
        "energy markets commentary rss",
        "daily energy news feed",
        "global energy markets rss",
    ]
    for key in corpus_keys:
        world.search.corpus[key] = rows


def _seed_adapters_and_bodies(world) -> None:
    from news_benchmark.clock import CLOCK
    from news_benchmark.fakes.adapters import FakeAdapter, ScenarioItem

    now = CLOCK.now()
    for idx, url in enumerate(CANDIDATE_URLS):
        items: list[ScenarioItem] = []
        for post_idx, (headline, body) in enumerate(CANDIDATE_POSTS[url]):
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


def _normalize(url: str) -> str:
    return url.rstrip("/").lower()


@pytest.mark.asyncio
async def test_s_discovery_agent_uses_inspect_source_for_borderline_candidate(world):
    """The orchestrator previews a borderline candidate before submitting."""
    from news_service.agents.source_discovery import pipeline as disc_pipeline_mod
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.failed_task import FailedTask
    from news_service.models.source import Source
    from news_service.models.subscription import Subscription
    from news_service.models.subscription_source import SubscriptionSource
    from news_service.models.user import User
    from news_service.tasks.discover_sources import run_and_persist_discovery

    _seed_search_corpus(world)
    _seed_adapters_and_bodies(world)

    user_id = uuid.uuid4()
    sub_id = uuid.uuid4()
    topic_embedding = await embed_text(RETRIEVAL_QUERY)

    async with async_session_factory() as s:
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

    inspect_calls: list[tuple[str, str, list[str]]] = []
    original_fetch = disc_pipeline_mod.fetch_source_posts

    async def _spy(url: str, source_kind: str):
        posts = await original_fetch(url, source_kind)
        sample_texts: list[str] = []
        for post in posts:
            text = getattr(post, "text", "")
            if text:
                sample_texts.append(text[:200])
        inspect_calls.append((url, source_kind, sample_texts))
        return posts

    disc_pipeline_mod.fetch_source_posts = _spy  # type: ignore[assignment]
    try:
        async with async_session_factory() as s:
            result = await run_and_persist_discovery(
                s, sub_id, reason="initial-discovery-for-test"
            )
    finally:
        disc_pipeline_mod.fetch_source_posts = original_fetch  # type: ignore[assignment]

    async with async_session_factory() as s:
        attached_rows = list(
            (
                await s.execute(
                    select(SubscriptionSource, Source)
                    .join(Source, Source.id == SubscriptionSource.source_id)
                    .where(SubscriptionSource.subscription_id == sub_id)
                )
            ).all()
        )
        failed = list((await s.execute(select(FailedTask))).scalars().all())

    attached_urls = [src.url for _link, src in attached_rows]
    inspected_urls = [_normalize(u) for (u, _k, _t) in inspect_calls]
    inspected_borderline = _normalize(BORDERLINE_URL) in inspected_urls
    borderline_in_selection = _normalize(BORDERLINE_URL) in [_normalize(u) for u in attached_urls]

    non_empty_previews = [
        (u, texts) for (u, _k, texts) in inspect_calls if any(t.strip() for t in texts)
    ]

    ok = (
        result.get("status") == "ok"
        and len(inspect_calls) >= 1
        and len(non_empty_previews) >= 1
        and (inspected_borderline or borderline_in_selection)
        and len(attached_rows) >= 1
        and not failed
    )
    assert ok, (
        "discovery inspect-then-submit benchmark failed: expected status=ok, "
        "at least one inspect_source call with a non-empty preview, the borderline "
        "candidate either inspected or in the final selection, at least one "
        "persisted SubscriptionSource, and zero FailedTask rows. "
        f"Got result={result!r}; "
        f"inspect_calls_count={len(inspect_calls)}; "
        f"inspected_urls={inspected_urls!r}; "
        f"inspected_borderline={inspected_borderline!r}; "
        f"borderline_in_selection={borderline_in_selection!r}; "
        f"non_empty_preview_count={len(non_empty_previews)}; "
        f"attached_urls={attached_urls!r}; "
        f"failed_tasks={failed!r}."
    )
