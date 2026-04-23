"""
S-discovery-happy-path: the source-discovery pipeline end-to-end on a fresh sub.

Scope
-----
Covers ``news_service.tasks.discover_sources.run_and_persist_discovery``
invoked directly (no conversational agent, no HTTP layer) against a
subscription that has zero attached sources. The real Discovery Agent
and real Source Finder LLMs run, the fake search returns curated
energy-policy feed URLs, the fake adapters serve on-topic posts behind
each URL so ``validate_and_score_source`` produces a real cosine score,
and the pipeline must persist at least one auto-discovered
``SubscriptionSource`` row.

Why ``run_and_persist_discovery`` directly
------------------------------------------
The conversational plumbing test ``test_s_conv_trigger_source_discovery``
stubs ``_run_inline_discovery`` because it only cares that the agent
routes the tool call. Here we test the pipeline logic itself: agent
orchestration, finder ReAct loop, score/validation wiring, selection
persistence. Going through the conversational agent would add another
LLM hop that has nothing to do with the behavior under test.

Happy path
----------
A fresh subscription about EU energy policy with no attached sources.
The finder's ``tool_search_web`` is seeded with realistic curator-query
results pointing at four plausible EU energy-policy feed URLs. Each URL
has a ``FakeAdapter`` with 3 on-topic posts whose bodies share the
subscription topic vocabulary. The agent spawns finders, the finders
search, harvest, and validate, the orchestrator submits, and the
pipeline writes the rows. One logical claim: at least one auto
``SubscriptionSource`` row ends up attached to the seeded subscription,
and every attached URL comes from the seeded candidate set.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select


CANDIDATE_URLS = [
    "https://brussels-energy-policy.invalid/feed.xml",
    "https://acer-decisions.invalid/rss",
    "https://entso-e-news.invalid/feed",
    "https://eu-energy-watch.invalid/atom.xml",
]

CANDIDATE_TITLES = {
    "https://brussels-energy-policy.invalid/feed.xml": "Brussels Energy Policy Tracker",
    "https://acer-decisions.invalid/rss": "ACER Decisions Wire",
    "https://entso-e-news.invalid/feed": "ENTSO-E News Room",
    "https://eu-energy-watch.invalid/atom.xml": "EU Energy Watch",
}

CANDIDATE_SNIPPETS = {
    "https://brussels-energy-policy.invalid/feed.xml": (
        "RSS feed covering EU energy policy: Council decisions, Commission proposals, "
        "ACER and ENTSO-E announcements, EUR-Lex directives on gas and electricity markets."
    ),
    "https://acer-decisions.invalid/rss": (
        "Official RSS for the EU Agency for the Cooperation of Energy Regulators (ACER): "
        "network codes, cross-border balancing rules, interconnector decisions."
    ),
    "https://entso-e-news.invalid/feed": (
        "ENTSO-E news feed: winter outlooks, transmission capacity assessments, "
        "grid code revisions, pan-European electricity market reports."
    ),
    "https://eu-energy-watch.invalid/atom.xml": (
        "Independent watch on EU energy policy: Commission directives, Parliament ENVI "
        "and ITRE committee votes, methane regulation, LNG import rules, renewable targets."
    ),
}


CANDIDATE_POSTS: dict[str, list[tuple[str, str]]] = {
    "https://brussels-energy-policy.invalid/feed.xml": [
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
    "https://acer-decisions.invalid/rss": [
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
                "implement the EU electricity market regulation and aligns with ENTSO-E methodology."
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
    "https://entso-e-news.invalid/feed": [
        (
            "ENTSO-E warns of winter capacity shortfall in Central Europe",
            (
                "The European Network of Transmission System Operators for Electricity "
                "(ENTSO-E) warned in its winter outlook that Central European grids may face "
                "capacity shortfalls during cold snaps. Germany, Austria and Czechia are "
                "most exposed under current EU energy policy planning."
            ),
        ),
        (
            "ENTSO-E publishes ten-year network development plan draft",
            (
                "ENTSO-E released the draft Ten-Year Network Development Plan (TYNDP) covering "
                "cross-border interconnection needs across the EU. The plan feeds into "
                "Commission's list of Projects of Common Interest and informs ACER's assessment "
                "of EU energy infrastructure priorities."
            ),
        ),
        (
            "ENTSO-E flags ancillary-services gap ahead of 2027 balancing go-live",
            (
                "ENTSO-E flagged an ancillary-services gap ahead of the 2027 go-live of the "
                "pan-European balancing platforms. National regulators and ACER are working on "
                "a joint response consistent with the electricity market regulation and EU "
                "energy policy objectives."
            ),
        ),
    ],
    "https://eu-energy-watch.invalid/atom.xml": [
        (
            "ENVI committee tightens methane-leak limits for imported LNG",
            (
                "The European Parliament's Committee on the Environment (ENVI) voted to tighten "
                "methane-intensity limits for imported liquefied natural gas. The amendment "
                "extends the Methane Regulation's import standard across the upstream value "
                "chain under EU energy and climate policy."
            ),
        ),
        (
            "EUR-Lex publishes directive on accelerated offshore wind permitting",
            (
                "A new directive accelerating permitting procedures for offshore wind "
                "installations was published on EUR-Lex and enters force in 20 days. The text "
                "caps permitting timelines inside 'renewable acceleration areas' at 24 months, "
                "one of the flagship files of EU energy policy this session."
            ),
        ),
        (
            "Commission unveils 40 percent renewable electricity target for 2030",
            (
                "The European Commission unveiled a proposal to set a binding 40 percent "
                "renewable electricity target for the EU-27 by 2030, up from 32 percent in the "
                "current Renewable Energy Directive. The file will go through Council and "
                "Parliament under the EU energy policy legislative agenda."
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
    "(ENVI, ITRE). Focus on policy and regulation, not retail market prices. "
    "Prefer RSS-style feeds. Skip EV-sales, stock coverage, and national "
    "political horse-trading.\n"
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
    """Populate the fake search with many phrasings aimed at the finder.

    The finder is prompted to use natural language curator queries and is
    budgeted to at most 2 phrasings per strategy. Seeding a broad set of
    prefix keys ensures whatever phrasing the LLM picks lands on results.
    """
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
    ]
    for key in corpus_keys:
        world.search.corpus[key] = rows


def _seed_adapters_and_bodies(world) -> None:
    """Populate FakeAdapter for each candidate URL and article bodies for synth items.

    ``validate_and_score_source`` calls ``fetch_source_posts`` which the
    ``world`` monkey-patch routes to ``World.fake_fetch_source_posts``.
    That helper reads ``world.adapters`` by URL. We build one
    ``FakeAdapter`` per candidate URL with 3 recent, topic-dense posts.
    We also mirror the item bodies into ``world.article_fetch.bodies``
    so any follow-up ``fetch_page`` on the synth item URL resolves.
    """
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


@pytest.mark.asyncio
async def test_s_discovery_happy_path_attaches_auto_discovered_sources(world):
    """Running discovery on a fresh sub attaches at least one auto-discovered source."""
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

    async with async_session_factory() as s:
        result = await run_and_persist_discovery(
            s, sub_id, reason="initial-discovery-for-test"
        )

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
    user_specified_flags = [link.is_user_specified for link, _src in attached_rows]
    unknown_urls = [u for u in attached_urls if u not in CANDIDATE_URLS]

    ok = (
        result.get("status") == "ok"
        and len(attached_rows) >= 1
        and all(flag is False for flag in user_specified_flags)
        and not unknown_urls
        and not failed
    )
    assert ok, (
        "discovery happy path failed: expected status=ok with at least one "
        "auto-discovered SubscriptionSource whose URL comes from the seeded "
        "candidate set, no user-specified flags set, and zero FailedTask rows. "
        f"Got result={result!r}; "
        f"attached_urls={attached_urls!r}; "
        f"user_specified_flags={user_specified_flags!r}; "
        f"unknown_urls={unknown_urls!r}; "
        f"failed_tasks={failed!r}."
    )
