"""
Shared digest-test fixtures: seeds one digest-mode sub plus a pool of
eight pre-embedded news items for the writer to draw from.

Kept outside any ``test_*.py`` module so pytest does not collect it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import timedelta

SOURCE_URL = "https://brussels-energy-policy.invalid/feed.xml"
WEBHOOK_URL = "https://bench.invalid/webhook/s-digest"


USER_SPEC = (
    "# EU energy & climate policy daily digest\n"
    "\n"
    "I want a daily round-up of EU-level energy and climate policy news: "
    "Council decisions, Commission proposals, ENTSO-E / ACER regulatory "
    "publications, directives entering force via EUR-Lex, and Parliament "
    "committee votes (ENVI, ITRE). Focus on policy and regulation, not "
    "market prices. English language, 5-8 items, short paragraph per "
    "item, no bold markdown.\n"
    "\n"
    "Skip: Tesla / EV sales news, sports, celebrity coverage, and generic "
    "tech-industry commentary."
)

RETRIEVAL_QUERY = (
    "EU energy and climate policy: Council decisions, Commission "
    "proposals, ENTSO-E, ACER, EUR-Lex directives, ENVI ITRE committees, "
    "gas storage, renewable targets, methane, wind, LNG"
)


@dataclass(frozen=True)
class SeedItem:
    """One pre-embedded news item plus the signature terms that identify it."""

    headline: str
    body: str
    signatures: tuple[str, ...]
    is_on_topic: bool
    age_hours: int


SEED_ITEMS: list[SeedItem] = [
    SeedItem(
        headline="Council of the EU adopts emergency gas storage directive for 2027 winter",
        body=(
            "The Council of the European Union on Tuesday formally adopted a "
            "directive requiring Member States to fill underground gas storage "
            "to 90% of capacity by 1 November each year starting in 2027. The "
            "text was approved by qualified majority after negotiations with "
            "the European Parliament. The directive replaces the temporary "
            "regulation that expires at the end of 2026 and makes the storage "
            "obligation permanent. Commission officials said the measure is "
            "designed to guard against the kind of supply shock Europe faced "
            "in 2022."
        ),
        signatures=("council", "gas", "storage"),
        is_on_topic=True,
        age_hours=4,
    ),
    SeedItem(
        headline="European Commission proposes 40% renewable electricity target by 2030",
        body=(
            "The European Commission on Wednesday unveiled a new proposal to "
            "set a binding 40% renewable electricity target for the EU-27 by "
            "2030, up from the 32% currently enshrined in the Renewable "
            "Energy Directive. Commissioner for Energy Kadri Simson said the "
            "higher ceiling reflects falling costs for solar and wind "
            "generation. The proposal will now go to the Council and "
            "Parliament for the ordinary legislative procedure."
        ),
        signatures=("commission", "renewable"),
        is_on_topic=True,
        age_hours=8,
    ),
    SeedItem(
        headline="ACER publishes final network code on electricity balancing markets",
        body=(
            "The EU Agency for the Cooperation of Energy Regulators (ACER) on "
            "Thursday published the final text of the amended network code on "
            "electricity balancing. The code harmonises cross-border "
            "balancing platforms and tightens imbalance-settlement timelines. "
            "National regulators have six months to transpose the changes. "
            "ACER director Christian Zinglersen called the publication a "
            "milestone for the internal electricity market."
        ),
        signatures=("acer", "balancing"),
        is_on_topic=True,
        age_hours=12,
    ),
    SeedItem(
        headline="ENTSO-E warns of winter capacity shortfall in Central Europe",
        body=(
            "The European Network of Transmission System Operators for "
            "Electricity (ENTSO-E) warned in its winter outlook published on "
            "Friday that Central European grids may face capacity shortfalls "
            "during cold snaps in December and January. Germany, Austria and "
            "the Czech Republic are most exposed. ENTSO-E urged national "
            "operators to finalise contingency reserves before November and "
            "flagged France's nuclear availability as the single biggest "
            "swing factor."
        ),
        signatures=("entso-e", "capacity"),
        is_on_topic=True,
        age_hours=16,
    ),
    SeedItem(
        headline="EUR-Lex publishes directive on accelerated offshore wind permitting",
        body=(
            "A new directive accelerating permitting procedures for offshore "
            "wind installations was published on EUR-Lex today and will enter "
            "force 20 days after publication. The text caps permitting "
            "timelines for projects located in pre-designated 'renewable "
            "acceleration areas' at 24 months. The Commission estimates the "
            "measure could unlock up to 32 GW of additional offshore "
            "capacity by 2030. Transposition deadline is 18 months."
        ),
        signatures=("offshore", "wind"),
        is_on_topic=True,
        age_hours=20,
    ),
    SeedItem(
        headline="ENVI committee tightens methane-leak limits for imported LNG",
        body=(
            "The European Parliament's Committee on the Environment, Public "
            "Health and Food Safety (ENVI) voted on Thursday to tighten "
            "methane-intensity limits for imported liquefied natural gas. "
            "The amendment extends the Methane Regulation's import standard "
            "to cover the full upstream value chain. The trilogue with the "
            "Council is expected in June. Environmental groups called the "
            "vote a significant strengthening of the text."
        ),
        signatures=("methane", "lng"),
        is_on_topic=True,
        age_hours=24,
    ),
    SeedItem(
        headline="Tesla reports 22% Q1 delivery growth as Model Y refresh lands",
        body=(
            "Tesla Inc delivered 480,309 vehicles in the first quarter of "
            "2026, a 22% year-on-year increase, led by Model Y refreshes in "
            "North America and Europe. The company maintained its full-year "
            "guidance of 2.2 million vehicles and reiterated that margin "
            "pressure should ease in the second half as input costs fall."
        ),
        signatures=("tesla",),
        is_on_topic=False,
        age_hours=6,
    ),
    SeedItem(
        headline="Premier League: Arsenal top the table after Liverpool draw",
        body=(
            "Arsenal moved back to the top of the Premier League table on "
            "Sunday after Liverpool were held to a 1-1 draw at Everton. "
            "Manchester City remain third, two points adrift, with a game in "
            "hand. The title race is expected to go down to the final "
            "weekend for the third consecutive season."
        ),
        signatures=("premier", "league"),
        is_on_topic=False,
        age_hours=10,
    ),
]


async def seed_digest_world(world, *, embedding_fn):
    """Create one digest-mode subscription, one source, eight news items.

    ``embedding_fn`` is a coroutine ``str -> list[float]`` the caller
    must supply so this helper never imports ``news_service`` at module
    load time. Returns ``(user_id, sub_id, source_id)``.
    """
    from news_service.db.session import async_session_factory
    from news_service.models.news_item import NewsItem
    from news_service.models.source import Source
    from news_service.models.subscription import Subscription
    from news_service.models.subscription_source import SubscriptionSource
    from news_service.models.user import User

    from news_benchmark.clock import CLOCK

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
        for idx, seed in enumerate(SEED_ITEMS):
            item_embedding = await embedding_fn(seed.headline + "\n" + seed.body[:400])
            s.add(
                NewsItem(
                    id=uuid.uuid4(),
                    source_id=source_id,
                    headline=seed.headline,
                    body=seed.body,
                    url=f"{SOURCE_URL.rstrip('/')}/item-{idx:02d}",
                    source="Brussels Energy Policy Tracker",
                    published_at=now - timedelta(hours=seed.age_hours),
                    fetched_at=now,
                    embedding=item_embedding,
                )
            )

        await s.commit()

    world.adapters[SOURCE_URL] = None  # No polling in digest tests; only seeded items.

    return user_id, sub_id, source_id


def count_covered_items(digest_body: str) -> tuple[int, int]:
    """Return ``(on_topic_hits, off_topic_hits)`` by signature-term match.

    An item counts as covered iff every one of its signature tokens
    appears (case-insensitive substring) somewhere in the digest body.
    This survives paraphrase because the signatures are chosen as
    distinctive proper-noun or topic-keyword tokens any faithful
    paraphrase must preserve.
    """
    body_low = digest_body.lower()
    on = 0
    off = 0
    for seed in SEED_ITEMS:
        if all(sig in body_low for sig in seed.signatures):
            if seed.is_on_topic:
                on += 1
            else:
                off += 1
    return on, off
