"""
Shared fixtures for the four reflector tests.

Each reflector test seeds one digest-mode subscription plus a source
pool with a single "bad" source that trips one of the four reflector
triggers (drift, staleness, contribution-streak, REVISE-after-max).
The helpers below cover the boilerplate so each test focuses on the
trigger-specific seeding and the assertions.

Kept outside any ``test_*.py`` module so pytest does not collect it.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

WEBHOOK_URL = "https://bench.invalid/webhook/s-reflector"


SUB_USER_SPEC = (
    "# EU energy & climate policy daily digest\n"
    "\n"
    "Daily round-up of EU-level energy and climate policy news: Council "
    "decisions, Commission proposals, ENTSO-E / ACER regulatory "
    "publications, directives entering force via EUR-Lex, and Parliament "
    "committee votes. Focus on policy and regulation, not market prices. "
    "English, 5-8 items, short paragraph per item, no bold markdown."
)

SUB_RETRIEVAL_QUERY = (
    "EU energy and climate policy: Council decisions, Commission "
    "proposals, ENTSO-E, ACER, EUR-Lex directives, ENVI ITRE committees, "
    "gas storage, renewable targets, methane, wind, LNG"
)


# Six on-topic EU energy-policy items with distinctive signature tokens,
# reused by any reflector test that wants the happy digest path to
# produce a real draft while a second "bad" source trips the trigger.
ON_TOPIC_ITEMS: list[dict[str, Any]] = [
    {
        "headline": "Council of the EU adopts emergency gas storage directive for 2027 winter",
        "body": (
            "The Council of the European Union on Tuesday adopted a directive "
            "requiring Member States to fill underground gas storage to 90% of "
            "capacity by 1 November each year starting in 2027. The text was "
            "approved by qualified majority. Commission officials said the "
            "measure guards against supply shocks like those seen in 2022."
        ),
    },
    {
        "headline": "European Commission proposes 40% renewable electricity target by 2030",
        "body": (
            "The European Commission on Wednesday proposed a binding 40% "
            "renewable electricity target for the EU-27 by 2030, up from 32%. "
            "Commissioner Kadri Simson said the higher ceiling reflects falling "
            "costs for solar and wind generation."
        ),
    },
    {
        "headline": "ACER publishes final network code on electricity balancing markets",
        "body": (
            "The EU Agency for the Cooperation of Energy Regulators published "
            "the final text of the amended network code on electricity "
            "balancing. National regulators have six months to transpose."
        ),
    },
    {
        "headline": "ENTSO-E warns of winter capacity shortfall in Central Europe",
        "body": (
            "ENTSO-E's winter outlook warns that Central European grids may "
            "face capacity shortfalls in December and January. Germany, Austria "
            "and the Czech Republic are most exposed."
        ),
    },
    {
        "headline": "EUR-Lex publishes directive on accelerated offshore wind permitting",
        "body": (
            "A new directive accelerating permitting for offshore wind "
            "installations was published on EUR-Lex and will enter force 20 "
            "days after publication. Transposition deadline is 18 months."
        ),
    },
    {
        "headline": "ENVI committee tightens methane-leak limits for imported LNG",
        "body": (
            "ENVI voted to tighten methane-intensity limits for imported LNG. "
            "The amendment extends the Methane Regulation's import standard to "
            "the full upstream value chain."
        ),
    },
]


@dataclass
class SourceSpec:
    """One source plus how its row should be shaped for a trigger test."""

    label: str
    url: str
    description_text: str
    is_user_specified: bool = False
    items: list[dict[str, Any]] = field(default_factory=list)
    items_age_hours: int = 6
    # Optional overrides on the SubscriptionSource row:
    digests_since_last_contribution: int = 0
    contribution_rate: float = 0.5
    contributed_last_30_digests: int = 5


async def seed_reflector_world(
    world,
    *,
    embedding_fn: Callable[[str], Awaitable[list[float]]],
    sources: list[SourceSpec],
) -> tuple[uuid.UUID, uuid.UUID, dict[str, uuid.UUID]]:
    """Seed one digest sub plus the given sources.

    Returns ``(user_id, sub_id, label -> source_id)``.

    ``sources`` is an ordered list; each SourceSpec describes one
    linked source with its embedding text (embedded via
    ``embedding_fn`` to compute ``source_description_embedding``) and
    an optional pre-seeded item set.
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

    topic_embedding = await embedding_fn(SUB_RETRIEVAL_QUERY)
    source_id_by_label: dict[str, uuid.UUID] = {}

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
            Subscription(
                id=sub_id,
                user_id=user_id,
                user_spec=SUB_USER_SPEC,
                delivery_mode="digest",
                schedule_cron="0 8 * * *",
                digest_language="en",
                delivery_webhook_url=WEBHOOK_URL,
                topic_embedding=topic_embedding,
                is_active=True,
            )
        )

        now = CLOCK.now()

        for spec in sources:
            source_id = uuid.uuid4()
            source_id_by_label[spec.label] = source_id

            description_embedding = await embedding_fn(spec.description_text)
            s.add(
                Source(
                    id=source_id,
                    url=spec.url,
                    title=spec.label,
                    source_description=spec.description_text[:500],
                    source_description_embedding=description_embedding,
                    subscriber_count=1,
                )
            )
            s.add(
                SubscriptionSource(
                    subscription_id=sub_id,
                    source_id=source_id,
                    is_user_specified=spec.is_user_specified,
                    digests_since_last_contribution=spec.digests_since_last_contribution,
                    contribution_rate=spec.contribution_rate,
                    contributed_last_30_digests=spec.contributed_last_30_digests,
                )
            )

            for idx, item in enumerate(spec.items):
                item_body = item["body"]
                item_embedding = await embedding_fn(
                    item["headline"] + "\n" + item_body[:400]
                )
                s.add(
                    NewsItem(
                        id=uuid.uuid4(),
                        source_id=source_id,
                        headline=item["headline"],
                        body=item_body,
                        url=f"{spec.url.rstrip('/')}/item-{idx:02d}",
                        source=spec.label,
                        published_at=now - timedelta(hours=spec.items_age_hours),
                        fetched_at=now,
                        embedding=item_embedding,
                    )
                )

        await s.commit()

    return user_id, sub_id, source_id_by_label


@dataclass
class DiscoveryStub:
    """Captures every discovery-task dispatch; lets tests assert count."""

    calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = field(default_factory=list)

    def call_count(self) -> int:
        return len(self.calls)


def install_discovery_stub(world) -> DiscoveryStub:
    """Replace CeleryShim's discover-task registry entry with a counter stub.

    Must be called AFTER ``world.install()`` so the registry has already
    been built. Returns a ``DiscoveryStub`` the test can inspect.
    """
    from news_benchmark.fakes.celery_shim import _TaskSpec  # type: ignore[import-not-found]

    stub = DiscoveryStub()

    async def _stub_impl(*args: Any, **kwargs: Any) -> dict[str, Any]:
        stub.calls.append((args, kwargs))
        return {"status": "stubbed", "reason": "reflector-test bypass"}

    task_name = "news_service.tasks.discover_sources.discover_sources_for_subscription"
    world.celery._registry[task_name] = _TaskSpec(impl=_stub_impl, coercions=())  # noqa: SLF001
    return stub


async def read_source_removal_log_for(sub_id):
    """Read every SourceRemovalLog row for this subscription."""
    from news_service.db.session import async_session_factory
    from news_service.models.source_removal_log import SourceRemovalLog
    from sqlalchemy import select

    async with async_session_factory() as s:
        rows = await s.execute(
            select(SourceRemovalLog).where(SourceRemovalLog.subscription_id == sub_id)
        )
        return list(rows.scalars().all())


async def read_subscription_sources(sub_id):
    """Return every SubscriptionSource row currently linked to the sub."""
    from news_service.db.session import async_session_factory
    from news_service.models.subscription_source import SubscriptionSource
    from sqlalchemy import select

    async with async_session_factory() as s:
        rows = await s.execute(
            select(SubscriptionSource).where(SubscriptionSource.subscription_id == sub_id)
        )
        return list(rows.scalars().all())
