"""
S-digest-writer-search-web: the Digest Writer reaches for ``search_web``
when the candidate items are too information-starved to satisfy a
user_spec that demands facts (dates, EUR-Lex reference numbers) which
simply are not present in any candidate body.

Setup. One digest-mode subscription whose user_spec declares strict
requirements: every item MUST cite (a) a date in YYYY-MM-DD format and
(b) a EUR-Lex reference number shaped like "Regulation (EU) YYYY/NNNN"
or "Directive YYYY/NN/EU". Five candidate NewsItems are seeded, each
with a short body (under 200 chars) that names an EU energy-policy
event but carries NO dates and NO reference numbers. The facts that
would let the writer meet the spec live only in
``world.search.corpus``: five prefixes ("cbam", "eu energy", "council",
"regulation", "eur-lex") map to the same four SearchResult rows, each
of which carries a specific date and a EUR-Lex reference.

Assertions.

  1. ``_deliver_digest(sub_id)`` returns ``status == "delivered"``.
  2. ``world.search.call_log`` has at least one entry: the writer
     searched. (``world`` already routes ``digest_writer._search_web``
     through ``FakeSearch.search_web`` which appends every query --
     no extra spy needed.)
  3. The delivered body contains at least one date token matching
     ``\\b20\\d{2}-\\d{2}-\\d{2}\\b`` OR at least one EUR-Lex
     reference token matching ``Regulation \\(EU\\) \\d{4}/\\d+`` or
     ``Directive \\d{4}/\\d+/EU``. These tokens exist only in the
     search corpus, not in any candidate body, so their presence in
     the digest proves the writer actually used the search results.
  4. Exactly one webhook fires for the test URL.
  5. Zero ``FailedTask`` rows.

Variance note. We cannot *force* a temperature-0.1 ADK agent to call
``search_web`` -- but the user_spec makes composing from candidates
alone obviously insufficient (no item can be written without facts
that are absent). If the writer still refuses to search, the
final-body-contains-fact assertion fails explicitly with the body
shown in the failure message so the failure mode is legible.

Out of scope: search-result quality (``FakeSearch`` returns the
corpus verbatim), fetch-page behaviour, the writer <-> judge REVISE
loop (a REVISE on the last draft still delivers the unreviewed draft).
"""

from __future__ import annotations

import re
import uuid
from datetime import timedelta

import pytest
from sqlalchemy import select

from news_benchmark.clock import CLOCK
from news_benchmark.fakes.search import SearchResult

WEBHOOK_URL = "https://bench.invalid/webhook/s-digest-search"
SOURCE_URL = "https://brussels-energy-policy.invalid/search-feed.xml"


USER_SPEC = (
    "# EU energy policy daily digest\n"
    "\n"
    "Topic: EU-level energy and climate policy news.\n"
    "\n"
    "STRICT REQUIREMENTS for every item in the digest:\n"
    "- MUST cite the exact date of the Council decision / Commission "
    "proposal / adoption in YYYY-MM-DD format.\n"
    "- MUST cite the EUR-Lex regulation or directive reference number "
    '(format "Regulation (EU) YYYY/NNNN" or "Directive YYYY/NN/EU").\n'
    "- Drafts missing any date or reference number for any item are "
    "invalid.\n"
    "- 5 items, short paragraphs of 2-3 sentences, plain text, no "
    "markdown bold.\n"
    "\n"
    "If the candidate items do not carry those facts, you MUST search "
    "the web for the missing dates and EUR-Lex references before "
    "submitting the digest."
)

RETRIEVAL_QUERY = (
    "EU energy policy Council Commission EUR-Lex regulation directive "
    "CBAM renewable electricity methane offshore wind"
)


STARVED_ITEMS: list[tuple[str, str]] = [
    (
        "Council adopts CBAM implementing regulation",
        (
            "The Council of the European Union adopted a new CBAM "
            "implementing regulation this week, according to officials. "
            "Industry response has been mixed."
        ),
    ),
    (
        "Commission proposes higher renewable electricity target",
        (
            "The European Commission unveiled a proposal this week to "
            "raise the binding renewable electricity target for the "
            "EU-27. Member States reacted cautiously."
        ),
    ),
    (
        "ENVI committee tightens methane import rules",
        (
            "The Parliament's ENVI committee voted to tighten methane "
            "intensity rules for imported LNG. Trilogue talks are "
            "expected shortly."
        ),
    ),
    (
        "EUR-Lex publishes offshore wind permitting directive",
        (
            "A new directive accelerating offshore wind permitting was "
            "published on EUR-Lex. Member States will transpose the "
            "measure in coming months."
        ),
    ),
    (
        "Council signs off on gas storage directive",
        (
            "The Council of the European Union signed off on a new "
            "gas storage directive, making the winter-fill obligation "
            "permanent beyond the current temporary regulation."
        ),
    ),
]


def _build_search_corpus() -> dict[str, list[SearchResult]]:
    """Return the same four fact-rich results under five prefixes.

    Every result carries both a concrete YYYY-MM-DD date and a
    EUR-Lex reference number, neither of which appears in any
    candidate body. Five prefixes cover the likely query shapes the
    writer will try.
    """
    results = [
        SearchResult(
            title="Council of the EU adopts CBAM implementing regulation",
            url=(
                "https://www.consilium.europa.eu/en/press/press-releases/"
                "2026-02-17/cbam-implementing-regulation"
            ),
            snippet=(
                "On 2026-02-17 the Council of the European Union formally "
                "adopted Regulation (EU) 2026/412 implementing the Carbon "
                "Border Adjustment Mechanism across cement, steel, "
                "aluminium, fertilisers, electricity, and hydrogen imports."
            ),
        ),
        SearchResult(
            title="Commission proposal on renewable electricity target",
            url=("https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:52026PC0081"),
            snippet=(
                "Proposal tabled 2026-01-29 to amend Directive 2023/55/EU "
                "with a binding 40 percent renewable electricity target "
                "for the EU-27 by 2030. Accompanying impact assessment "
                "references Regulation (EU) 2026/88."
            ),
        ),
        SearchResult(
            title="ENVI committee vote on imported-LNG methane limits",
            url=("https://www.europarl.europa.eu/news/en/press-room/2026-03-05-envi-methane"),
            snippet=(
                "Committee vote held 2026-03-05 on amendments extending "
                "Regulation (EU) 2024/1787 to cover the full upstream "
                "value chain of imported LNG."
            ),
        ),
        SearchResult(
            title="Offshore wind permitting directive published on EUR-Lex",
            url=("https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32026L0045"),
            snippet=(
                "Directive 2026/45/EU on accelerated permitting for "
                "offshore wind installations published on 2026-02-03. "
                "Transposition deadline is 18 months from entry into "
                "force."
            ),
        ),
    ]
    return {
        "cbam": results,
        "eu energy": results,
        "council": results,
        "regulation": results,
        "eur-lex": results,
    }


async def _seed_subscription_and_items(world, *, embedding_fn) -> uuid.UUID:
    """Insert user, source, subscription, and five info-starved items.

    Returns the subscription id. Every body is deliberately stripped
    of dates and EUR-Lex references; those live only in the search
    corpus.
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
                title="Brussels Energy Policy Tracker",
                source_description="EU-level energy and climate policy newswire.",
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
                    source="Brussels Energy Policy Tracker",
                    published_at=now - timedelta(hours=4 + idx * 2),
                    fetched_at=now,
                    embedding=item_embedding,
                )
            )

        await s.commit()

    world.adapters[SOURCE_URL] = None

    return sub_id


_DATE_RE = re.compile(r"\b20\d{2}-\d{2}-\d{2}\b")
_REG_RE = re.compile(r"Regulation \(EU\) \d{4}/\d+")
_DIR_RE = re.compile(r"Directive \d{4}/\d+/EU")


@pytest.mark.asyncio
async def test_s_digest_writer_uses_search_web_when_candidates_lack_facts(world):
    """Info-starved candidates + fact-demanding spec -> writer calls search_web."""
    from news_service.db.session import async_session_factory
    from news_service.db.vector_store import embed_text
    from news_service.models.failed_task import FailedTask
    from news_service.tasks.deliver_digest import _deliver_digest

    world.search.corpus.update(_build_search_corpus())

    for _headline, body in STARVED_ITEMS:
        assert not _DATE_RE.search(body), (
            f"seed body unexpectedly contains a date token; the test premise "
            f"requires candidates to lack dates. Body: {body!r}"
        )
        assert not _REG_RE.search(body) and not _DIR_RE.search(body), (
            f"seed body unexpectedly contains a EUR-Lex reference; the test "
            f"premise requires candidates to lack references. Body: {body!r}"
        )

    sub_id = await _seed_subscription_and_items(world, embedding_fn=embed_text)

    result = await _deliver_digest(sub_id)
    assert result.get("status") == "delivered", f"expected delivered status, got {result!r}"

    await world.celery.drain()

    search_calls = list(world.search.call_log)
    captured = world.delivery.for_url(WEBHOOK_URL)

    assert len(search_calls) >= 1, (
        f"expected writer to call search_web at least once because candidate "
        f"bodies lack the dates and EUR-Lex references the user_spec demands; "
        f"search_calls={search_calls!r}. "
        f"Delivered body: "
        f"{(captured[0].body if captured else '<no delivery>')!r}"
    )

    assert len(captured) == 1, (
        f"expected exactly 1 digest webhook for {WEBHOOK_URL}, got "
        f"{len(captured)}. Bodies: {[c.body[:160] for c in captured]}"
    )

    body = captured[0].body
    has_date = bool(_DATE_RE.search(body))
    has_reference = bool(_REG_RE.search(body) or _DIR_RE.search(body))
    assert has_date or has_reference, (
        f"expected the delivered body to contain at least one search-only "
        f"token (YYYY-MM-DD date OR EUR-Lex reference); candidate bodies "
        f"carry neither, so a fact in the digest proves the writer actually "
        f"used search_web output. search_calls={search_calls!r}. "
        f"Delivered body:\n{body}"
    )

    async with async_session_factory() as s:
        failed = list((await s.execute(select(FailedTask))).scalars().all())
    assert not failed, f"expected 0 failed_tasks rows, got {len(failed)}: {failed!r}"
