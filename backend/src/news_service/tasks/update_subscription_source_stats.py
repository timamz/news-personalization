"""Celery task that refreshes per (subscription, source) digest-pool stats.

Runs daily. For each active subscription with a topic embedding, the task
loads every linked source's recent item embeddings and sent-item history,
then populates six stats on ``subscription_sources``:

- ``item_cosine_p50``, ``item_cosine_p90``, ``item_cosine_std``: a three-
  number summary of how closely the source's recent items match the
  subscription topic, giving the Reflector a distribution shape rather
  than a single smoothed mean.
- ``contributed_last_30_digests``: how many of the last 30 delivered
  digests included an item from this source.
- ``contribution_rate``: ratio of digest-included items to recently
  published items -- catches sources that publish a lot but rarely hit
  the digest.
- ``digests_since_last_contribution``: streak counter used by the
  Reflector's contribution-streak trigger; reset to 0 when the source
  contributes to a digest.

All stats are per (subscription, source) because they depend on the
subscription's topic vector and sent-item history. Source-global signals
(publish volume, fetch health) live on ``sources`` and are maintained
elsewhere.
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.core.config import get_settings
from news_service.db.session import get_task_session
from news_service.models.news_item import NewsItem
from news_service.models.sent_item import SentItem
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.services.relevance import cosine_similarity
from news_service.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)
settings = get_settings()

UPDATE_SUBSCRIPTION_SOURCE_STATS_TASK = (
    "news_service.tasks.update_subscription_source_stats.update_all_subscription_source_stats"
)

_COSINE_WINDOW_DAYS = 30
_DIGEST_WINDOW = 30


@dataclass(slots=True)
class _DigestRecord:
    """A digest delivery: its timestamp and the source_ids that contributed."""

    sent_at: datetime
    contributing_source_ids: set


@celery_app.task(name=UPDATE_SUBSCRIPTION_SOURCE_STATS_TASK)
def update_all_subscription_source_stats() -> dict:
    """Entry point invoked by Celery Beat."""
    return asyncio.run(_update_all())


async def _update_all() -> dict:
    updated = 0
    skipped = 0
    async with get_task_session() as session:
        subs = (
            (
                await session.execute(
                    select(Subscription).where(
                        Subscription.is_active.is_(True),
                        Subscription.paused_at.is_(None),
                    )
                )
            )
            .scalars()
            .all()
        )

        for sub in subs:
            if sub.topic_embedding is None:
                skipped += 1
                continue
            topic = list(sub.topic_embedding)
            links = (
                (
                    await session.execute(
                        select(SubscriptionSource).where(
                            SubscriptionSource.subscription_id == sub.id
                        )
                    )
                )
                .scalars()
                .all()
            )
            if not links:
                continue

            digest_records = await _recent_digests(session, sub.id, limit=_DIGEST_WINDOW)
            now = datetime.now(UTC)

            for link in links:
                stats = await _compute_link_stats(
                    session=session,
                    subscription_id=sub.id,
                    source_id=link.source_id,
                    topic_embedding=topic,
                    digest_records=digest_records,
                    now=now,
                )
                link.contributed_last_30_digests = stats.contributed_count
                link.contribution_rate = stats.contribution_rate
                link.digests_since_last_contribution = stats.streak
                link.item_cosine_p50 = stats.p50
                link.item_cosine_p90 = stats.p90
                link.item_cosine_std = stats.std
                link.stats_updated_at = now
                updated += 1

        await session.commit()

    logger.info(
        "Subscription-source stats refresh: updated=%d skipped_subscriptions=%d",
        updated,
        skipped,
    )
    return {"status": "ok", "updated": updated, "skipped_subscriptions": skipped}


@dataclass(slots=True)
class _LinkStats:
    contributed_count: int
    contribution_rate: float
    streak: int
    p50: float | None
    p90: float | None
    std: float | None


async def _recent_digests(
    session: AsyncSession, subscription_id, limit: int
) -> list[_DigestRecord]:
    """Load the most recent ``limit`` digests (grouped by sent_at) for a subscription.

    Two items with the same ``sent_at`` belong to the same digest batch. We
    fetch enough rows to reconstruct at least ``limit`` distinct batches plus
    their contributing source ids.
    """
    stmt = (
        select(SentItem.sent_at, NewsItem.source_id)
        .join(NewsItem, NewsItem.id == SentItem.news_item_id)
        .where(SentItem.subscription_id == subscription_id)
        .order_by(SentItem.sent_at.desc())
    )
    rows = (await session.execute(stmt)).all()

    records: list[_DigestRecord] = []
    current: _DigestRecord | None = None
    for sent_at, source_id in rows:
        if current is None or sent_at != current.sent_at:
            if len(records) == limit:
                break
            current = _DigestRecord(sent_at=sent_at, contributing_source_ids=set())
            records.append(current)
        current.contributing_source_ids.add(source_id)
    return records


async def _compute_link_stats(
    *,
    session: AsyncSession,
    subscription_id,
    source_id,
    topic_embedding: list[float],
    digest_records: list[_DigestRecord],
    now: datetime,
) -> _LinkStats:
    cutoff = now - timedelta(days=_COSINE_WINDOW_DAYS)
    embedding_rows = (
        await session.execute(
            select(NewsItem.embedding).where(
                NewsItem.source_id == source_id,
                NewsItem.embedding.is_not(None),
                NewsItem.published_at.is_not(None),
                NewsItem.published_at >= cutoff,
            )
        )
    ).all()
    cosines = [
        cosine_similarity(list(emb), topic_embedding)
        for (emb,) in embedding_rows
        if emb is not None
    ]
    p50, p90, std = _distribution_stats(cosines)
    published_count = len(cosines)

    items_in_digests = (
        await session.execute(
            select(func.count())
            .select_from(SentItem)
            .join(NewsItem, NewsItem.id == SentItem.news_item_id)
            .where(
                SentItem.subscription_id == subscription_id,
                NewsItem.source_id == source_id,
                SentItem.sent_at >= cutoff,
            )
        )
    ).scalar_one()

    contribution_rate = items_in_digests / published_count if published_count > 0 else 0.0

    contributed_count = sum(
        1 for record in digest_records if source_id in record.contributing_source_ids
    )

    streak = 0
    for record in digest_records:
        if source_id in record.contributing_source_ids:
            break
        streak += 1

    return _LinkStats(
        contributed_count=contributed_count,
        contribution_rate=min(contribution_rate, 1.0),
        streak=streak,
        p50=p50,
        p90=p90,
        std=std,
    )


def _distribution_stats(values: list[float]) -> tuple[float | None, float | None, float | None]:
    n = len(values)
    if n == 0:
        return None, None, None
    if n == 1:
        return values[0], values[0], 0.0

    ordered = sorted(values)
    p50 = _percentile(ordered, 0.5)
    p90 = _percentile(ordered, 0.9)
    mean = sum(ordered) / n
    variance = sum((v - mean) ** 2 for v in ordered) / n
    std = variance**0.5
    return p50, p90, std


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile on an already-sorted list."""
    if not sorted_values:
        raise ValueError("cannot compute percentile of empty list")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    low = int(pos)
    high = min(low + 1, len(sorted_values) - 1)
    frac = pos - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * frac
