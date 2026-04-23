"""
S-digest-writer-fetch-page: the Digest Writer reaches for
``fetch_page_bounded`` when a search snippet is too thin to carry the
concrete fact the user_spec demands, but the full page body does.

Setup. One digest-mode subscription whose user_spec insists every item
cite the exact ENTSO-E document code (format ``ENTSO-E-YYYY-NNNN``).
Six candidate NewsItems are seeded on the general ENTSO-E / EU
grid-operations topic, but NONE of them names a document code -- the
premise is that the codes live outside the polled feed.

Three search results are configured under five query prefixes; ALL
three snippets are deliberately thin: "The filing details were
published on the agency site." -- no code, no date, no specific
reference. The writer therefore cannot satisfy the spec by reading
snippets alone. Only ONE of the three URLs, when fetched as a full
page via ``fetch_page_bounded``, returns text containing the concrete
code ``ENTSO-E-2026-0412``. The other two URLs return generic
uninformative page bodies.

Fake strategy. No new fake is introduced. The existing
``FakeArticleFetch`` at ``news_benchmark.fakes.article_fetch`` is
already wired through ``World.install()`` onto
``news_service.agents.web_tools.fetch_article_text``, which is what
``fetch_page`` (aliased as ``_fetch_page`` in the writer and exposed
to the ADK agent as ``fetch_page_bounded``) calls. Populating
``world.article_fetch.bodies[url] = ...`` per URL gives per-URL
response control, and ``world.article_fetch.call_log`` gives call
counting and URL capture for free.

Assertions (one behavioral claim: "writer calls fetch_page to pull
a snippet-absent fact from the full page").

  1. ``_deliver_digest(sub_id)`` returns ``status == "delivered"``.
  2. ``world.search.call_log`` has at least one entry (writer searched).
  3. ``world.article_fetch.call_log`` contains at least one URL AND
     contains the URL that carries the code -- proof the writer
     clicked through to the right page.
  4. Exactly one webhook fires for the test URL.
  5. The delivered body contains the exact code ``ENTSO-E-2026-0412``
     -- this string lives in NO candidate body and in NO search
     snippet, so its presence in the digest proves the writer
     pulled it out of the full-page text via ``fetch_page_bounded``.
  6. Zero ``FailedTask`` rows.

Variance note. We cannot *force* a low-temperature ADK agent to
call ``fetch_page_bounded`` -- but the user_spec makes composing
without a code obviously out of compliance, and every snippet is
deliberately vague, so searching is not enough. If the writer still
refuses to fetch, the final-body-contains-code assertion fails
explicitly with the delivered body and the fetch log dumped in the
failure message, so the failure mode is legible.

Out of scope: fetch-budget exhaustion, Judge REVISE loops, the exact
number of fetches performed, which URL the writer tries first.
"""

from __future__ import annotations

import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select

from news_benchmark.clock import CLOCK
from news_benchmark.fakes.search import SearchResult

WEBHOOK_URL = "https://bench.invalid/webhook/s-digest-fetch"
SOURCE_URL = "https://entsoe-grid-tracker.invalid/feed.xml"

FACT_BEARING_URL = (
    "https://www.entsoe.eu/publications/filings/2026-03-14-market-coupling-notice"
)
GENERIC_URL_A = (
    "https://www.entsoe.eu/news/2026-03-14-press-release-index"
)
GENERIC_URL_B = (
    "https://www.consilium.europa.eu/press/2026-03-14/grid-operations-update"
)

TARGET_CODE = "ENTSO-E-2026-0412"


USER_SPEC = (
    "# ENTSO-E filings daily digest\n"
    "\n"
    "Topic: ENTSO-E publications, market-coupling notices, balancing "
    "platform filings, and cross-border capacity allocations published "
    "by the EU TSO association.\n"
    "\n"
    "STRICT REQUIREMENTS for every item in the digest:\n"
    "- MUST cite the exact ENTSO-E document code when the item refers "
    "to an ENTSO-E filing. Codes are formatted ``ENTSO-E-YYYY-NNNN`` "
    "(for example ``ENTSO-E-2025-0007``). Without a code an item "
    "cannot be included.\n"
    "- 3 to 5 items, short paragraphs, plain text, no markdown bold.\n"
    "\n"
    "If the candidate items do not carry the code, you MUST search the "
    "web and, when a search snippet is too brief to contain the code, "
    "fetch the full page to extract it before submitting the digest."
)

RETRIEVAL_QUERY = (
    "ENTSO-E filing document code market coupling balancing platform "
    "cross-border capacity allocation TSO"
)


STARVED_ITEMS: list[tuple[str, str]] = [
    (
        "ENTSO-E releases new market-coupling notice",
        (
            "The European Network of Transmission System Operators for "
            "Electricity released a new market-coupling notice this week. "
            "The document covers day-ahead cross-border allocation "
            "procedures and introduces adjustments to the harmonised "
            "allocation rules."
        ),
    ),
    (
        "Balancing platform roll-out enters second phase",
        (
            "The cross-border balancing platform roll-out has entered its "
            "second phase according to the TSO association. Additional "
            "control areas will connect over the coming months under the "
            "EU network code on electricity balancing."
        ),
    ),
    (
        "TSOs coordinate winter adequacy planning",
        (
            "Transmission System Operators across the EU have begun "
            "coordinating their winter adequacy planning in line with the "
            "ENTSO-E seasonal outlook methodology. Contingency reserves "
            "are being benchmarked against last year's profile."
        ),
    ),
    (
        "Harmonised allocation rules consultation closes",
        (
            "Public consultation on amendments to the harmonised "
            "allocation rules closed this week. Market participants "
            "submitted responses on collateral, firmness, and curtailment "
            "compensation provisions for long-term transmission rights."
        ),
    ),
    (
        "Cross-border capacity allocation audit completed",
        (
            "A scheduled audit of cross-border capacity allocation "
            "procedures was completed in the quarter. Findings focus on "
            "transparency of remedial-action costs and the treatment of "
            "internal congestion in flow-based calculations."
        ),
    ),
    (
        "ENTSO-E publishes TYNDP stakeholder workshop summary",
        (
            "A summary of the Ten-Year Network Development Plan "
            "stakeholder workshop was published by ENTSO-E. The write-up "
            "covers offshore hybrid projects and integration of "
            "hydrogen-ready corridors in future scenarios."
        ),
    ),
]


def _build_thin_snippet_corpus() -> dict[str, list[SearchResult]]:
    """Three results under five prefixes; every snippet is vague.

    No snippet mentions the document code or any other concrete
    identifier -- the writer must click through to a full page to
    satisfy the spec.
    """
    results = [
        SearchResult(
            title="ENTSO-E market coupling notice -- filing page",
            url=FACT_BEARING_URL,
            snippet="The filing details were published on the agency site.",
        ),
        SearchResult(
            title="ENTSO-E press release index",
            url=GENERIC_URL_A,
            snippet="Recent announcements from the European TSO association.",
        ),
        SearchResult(
            title="Council press briefing on grid operations",
            url=GENERIC_URL_B,
            snippet="Summary of grid operations discussed during the week.",
        ),
    ]
    return {
        "entso-e": results,
        "market coupling": results,
        "balancing platform": results,
        "tso filing": results,
        "cross-border capacity": results,
    }


def _build_fetch_bodies() -> dict[str, str]:
    """Only one URL carries the code; the other two are deliberately bland.

    The fact-bearing body names ``ENTSO-E-2026-0412`` explicitly so a
    writer that fetches this URL cannot miss it. The two generic
    bodies contain no code of any shape.
    """
    fact_body = (
        "ENTSO-E FILING -- MARKET COUPLING NOTICE\n\n"
        "Publication date: 14 March 2026.\n\n"
        "Document code: ENTSO-E-2026-0412.\n\n"
        "This notice covers amendments to the day-ahead market coupling "
        "procedure across the core region. The filing revises the "
        "harmonised allocation rules for cross-border transmission "
        "capacity and supersedes earlier guidance on remedial-action "
        "cost allocation. Transmission System Operators in the affected "
        "bidding zones are instructed to implement the updated procedure "
        "within 60 days of publication. The document code "
        "ENTSO-E-2026-0412 should be cited in any downstream reference."
    )
    generic_a = (
        "ENTSO-E PRESS RELEASE INDEX\n\n"
        "This page lists recent announcements from the European Network "
        "of Transmission System Operators for Electricity. Entries cover "
        "a broad range of topics including seasonal adequacy, stakeholder "
        "workshops, and network development planning. Please follow the "
        "individual publication links for filing-specific details."
    )
    generic_b = (
        "COUNCIL GRID OPERATIONS UPDATE\n\n"
        "The Council press service summarised recent discussions on grid "
        "operations during the week. Topics included winter preparedness, "
        "balancing reserves, and coordination between Member State TSOs. "
        "No specific regulatory instruments were adopted during the "
        "briefing."
    )
    return {
        FACT_BEARING_URL: fact_body,
        GENERIC_URL_A: generic_a,
        GENERIC_URL_B: generic_b,
    }


async def _seed_subscription_and_items(world, *, embedding_fn) -> uuid.UUID:
    """Insert user, source, subscription, and six code-free items.

    Every seeded body is deliberately stripped of any document code;
    the target code lives only in the fact-bearing page body.
    """
    from news_service.db.session import async_session_factory
    from news_service.models.news_item import NewsItem
    from news_service.models.source import Source
    from news_service.models.subscription import Subscription
    from news_service.models.subscription_source import SubscriptionSource
    from news_service.models.user import User

    user_id = uuid.uuid4()
    sub_id = uuid.uuid4()
    source_id = uuid.uuid4()

    topic_embedding = await embedding_fn(RETRIEVAL_QUERY)

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
                title="ENTSO-E Grid Operations Tracker",
                source_description=(
                    "EU TSO association publications: market coupling, "
                    "balancing, cross-border capacity."
                ),
            )
        )
        s.add(
            Subscription(
                id=sub_id,
                user_id=user_id,
                user_spec=USER_SPEC,
                delivery_mode="digest",
                schedule_cron="0 8 * * *",
                digest_language="en",
                delivery_webhook_url=WEBHOOK_URL,
                topic_embedding=topic_embedding,
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

        now = CLOCK.now()
        for idx, (headline, body) in enumerate(STARVED_ITEMS):
            item_embedding = await embedding_fn(headline + "\n" + body[:400])
            s.add(
                NewsItem(
                    id=uuid.uuid4(),
                    source_id=source_id,
                    headline=headline,
                    body=body,
                    url=f"{SOURCE_URL.rstrip('/')}/item-{idx:02d}",
                    source="ENTSO-E Grid Operations Tracker",
                    published_at=now - timedelta(hours=4 + idx * 2),
                    fetched_at=now,
                    embedding=item_embedding,
                )
            )

        await s.commit()

    world.adapters[SOURCE_URL] = None

    return sub_id


@pytest.mark.asyncio
async def test_s_digest_writer_uses_fetch_page_when_snippets_are_thin(world):
    """Thin snippets + a code only in the full page -> writer calls fetch_page."""
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.failed_task import FailedTask
    from news_service.tasks.deliver_digest import _deliver_digest

    world.search.corpus.update(_build_thin_snippet_corpus())
    world.article_fetch.bodies.update(_build_fetch_bodies())

    for _headline, body in STARVED_ITEMS:
        assert TARGET_CODE not in body, (
            f"seed body unexpectedly contains the target code; the test premise "
            f"requires candidates to lack it. Body: {body!r}"
        )
    for rows in _build_thin_snippet_corpus().values():
        for r in rows:
            assert TARGET_CODE not in r.snippet, (
                f"search snippet unexpectedly contains the target code; the test "
                f"premise requires snippets to be thin. Snippet: {r.snippet!r}"
            )

    sub_id = await _seed_subscription_and_items(world, embedding_fn=embed_text)

    result = await _deliver_digest(sub_id)
    assert result.get("status") == "delivered", (
        f"expected delivered status, got {result!r}"
    )

    await world.celery.drain()

    search_calls = list(world.search.call_log)
    fetch_calls = list(world.article_fetch.call_log)
    captured = world.delivery.for_url(WEBHOOK_URL)
    delivered_body = captured[0].body if captured else "<no delivery>"

    assert len(search_calls) >= 1, (
        f"expected writer to call search_web at least once because no candidate "
        f"body carries the ENTSO-E document code the user_spec demands; "
        f"search_calls={search_calls!r}. Delivered body: {delivered_body!r}"
    )

    assert len(fetch_calls) >= 1 and FACT_BEARING_URL in fetch_calls, (
        f"expected writer to call fetch_page_bounded at least once AND to fetch "
        f"{FACT_BEARING_URL!r} -- every search snippet is deliberately vague, "
        f"and the target code {TARGET_CODE!r} lives only in that URL's full "
        f"page. search_calls={search_calls!r}, fetch_calls={fetch_calls!r}. "
        f"Delivered body: {delivered_body!r}"
    )

    assert len(captured) == 1, (
        f"expected exactly 1 digest webhook for {WEBHOOK_URL}, got "
        f"{len(captured)}. Bodies: {[c.body[:160] for c in captured]}"
    )

    assert TARGET_CODE in delivered_body, (
        f"expected the delivered body to contain {TARGET_CODE!r}; this exact "
        f"string exists in no candidate body and in no search snippet, so its "
        f"presence in the digest is the only proof the writer actually read "
        f"the fact-bearing page via fetch_page_bounded. "
        f"search_calls={search_calls!r}, fetch_calls={fetch_calls!r}. "
        f"Delivered body:\n{delivered_body}"
    )

    async with async_session_factory() as s:
        failed = list((await s.execute(select(FailedTask))).scalars().all())
    assert not failed, (
        f"expected 0 failed_tasks rows, got {len(failed)}: {failed!r}"
    )
