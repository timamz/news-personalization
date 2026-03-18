import logging
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.core.config import get_settings
from news_service.db.vector_store import embed_text
from news_service.models.rss_feed import RssFeed
from news_service.models.sent_item import SentItem
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource

logger = logging.getLogger(__name__)

settings = get_settings()


def _effective_prompt(subscription: Subscription) -> str:
    canonical_prompt = getattr(subscription, "canonical_prompt", "")
    return canonical_prompt.strip() or subscription.raw_prompt


async def generate_digest(session: AsyncSession, subscription: Subscription) -> str | None:
    sent_result = await session.execute(
        select(SentItem.news_item_id, SentItem.sent_at).where(
            SentItem.subscription_id == subscription.id
        )
    )
    sent_rows = list(sent_result.all())
    sent_ids: set[uuid.UUID] = {news_item_id for news_item_id, _ in sent_rows}
    last_sent_at = max((sent_at for _, sent_at in sent_rows), default=None)

    source_feed_ids = await _source_feed_ids_for_digest(session, subscription)
    if not source_feed_ids:
        logger.warning(
            "No fixed sources configured for subscription %s",
            subscription.id,
            extra={"subscription_id": str(subscription.id)},
        )
        return None

    query_embedding = getattr(subscription, "canonical_prompt_embedding", None)
    if query_embedding is None:
        query_text = _effective_prompt(subscription) or subscription.prompt_summary.strip()
        query_embedding = await embed_text(query_text)
        subscription.canonical_prompt_embedding = query_embedding

    from news_service.agents.digest_curator import run_digest_curator

    try:
        result = await run_digest_curator(
            session=session,
            query_embedding=query_embedding,
            exclude_ids=sent_ids,
            allowed_feed_ids=source_feed_ids,
            published_after=_published_after_for_digest(last_sent_at),
            format_instructions=subscription.format_instructions,
            digest_language=subscription.digest_language,
        )
    except Exception:
        logger.exception(
            "Digest curator agent failed for subscription %s",
            subscription.id,
            extra={"subscription_id": str(subscription.id)},
        )
        return None

    if result is None:
        logger.info(
            "No usable news items for subscription %s",
            subscription.id,
            extra={"subscription_id": str(subscription.id)},
        )
        return None

    used_ids = [uuid.UUID(item_id) for item_id in result.used_item_ids]
    await _mark_as_sent(session, subscription.id, used_ids)

    logger.info(
        "Generated digest with %d items for subscription %s",
        len(used_ids),
        subscription.id,
        extra={"subscription_id": str(subscription.id)},
    )
    return result.digest_text


def _published_after_for_digest(last_sent_at: datetime | None) -> datetime:
    if last_sent_at is not None:
        return last_sent_at
    return datetime.now(UTC) - timedelta(days=settings.news_item_max_age_days)


async def _source_feed_ids_for_digest(
    session: AsyncSession,
    subscription: Subscription,
) -> set[uuid.UUID]:
    source_result = await session.execute(
        select(RssFeed.id)
        .join(SubscriptionSource, SubscriptionSource.feed_id == RssFeed.id)
        .where(SubscriptionSource.subscription_id == subscription.id)
    )
    return {row[0] for row in source_result.all()}


async def _mark_as_sent(
    session: AsyncSession,
    subscription_id: uuid.UUID,
    news_item_ids: list[uuid.UUID],
) -> None:
    for item_id in news_item_ids:
        session.add(SentItem(subscription_id=subscription_id, news_item_id=item_id))
    await session.flush()
