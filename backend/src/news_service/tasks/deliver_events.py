import asyncio
import logging
import uuid

from sqlalchemy import select

from news_service.agents.event import assess_and_compose_event_notification
from news_service.core.config import get_settings
from news_service.db.session import get_task_session
from news_service.models.news_item import NewsItem
from news_service.models.sent_item import SentItem
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.services.delivery import get_delivery_channel
from news_service.services.event_notifications import load_recent_notification_history
from news_service.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)
settings = get_settings()

_HEADLINE_DUPLICATE_THRESHOLD = 0.95


def _headline_is_obvious_duplicate(
    headline: str,
    history: list[str],
) -> bool:
    normalized = headline.strip().casefold()
    if not normalized:
        return False
    for entry in history:
        entry_normalized = entry.strip().casefold()
        if not entry_normalized:
            continue
        shorter = min(len(normalized), len(entry_normalized))
        longer = max(len(normalized), len(entry_normalized))
        if shorter == 0:
            continue
        common = 0
        for a, b in zip(normalized, entry_normalized, strict=False):
            if a == b:
                common += 1
        overlap = common / longer
        if overlap >= _HEADLINE_DUPLICATE_THRESHOLD:
            return True
    return False


@celery_app.task(name="news_service.tasks.deliver_events.deliver_event_notifications")
def deliver_event_notifications(news_item_id: str) -> dict:
    return asyncio.run(_deliver_event_notifications(uuid.UUID(news_item_id)))


async def _deliver_event_notifications(news_item_id: uuid.UUID) -> dict:
    async with get_task_session() as session:
        item = await session.get(NewsItem, news_item_id)
        if item is None:
            logger.warning("News item %s not found for event delivery", news_item_id)
            return {"status": "skipped", "reason": "news_item_not_found"}

        result = await session.execute(
            select(Subscription)
            .join(
                SubscriptionSource,
                SubscriptionSource.subscription_id == Subscription.id,
            )
            .where(
                Subscription.is_active.is_(True),
                Subscription.delivery_mode == "event",
                SubscriptionSource.feed_id == item.feed_id,
            )
        )
        subscriptions = list(result.scalars().all())
        if not subscriptions:
            return {"status": "skipped", "reason": "no_matching_subscriptions"}

        subscription_ids = [subscription.id for subscription in subscriptions]
        sent_result = await session.execute(
            select(SentItem.subscription_id).where(
                SentItem.news_item_id == item.id,
                SentItem.subscription_id.in_(subscription_ids),
            )
        )
        sent_subscription_ids = set(sent_result.scalars().all())

        delivered = 0
        failed = 0
        for subscription in subscriptions:
            if subscription.id in sent_subscription_ids:
                continue

            history = await load_recent_notification_history(session, subscription.id)
            history_headlines = [entry.title for entry in history]
            if _headline_is_obvious_duplicate(item.headline, history_headlines):
                logger.info(
                    "Event %s skipped for subscription %s: deterministic headline duplicate",
                    item.id,
                    subscription.id,
                    extra={
                        "subscription_id": str(subscription.id),
                        "news_item_id": str(item.id),
                    },
                )
                continue

            history_strings = [
                f"Title: {entry.title}\nSummary: {entry.summary}\n"
                f"Source: {entry.source}\nShown at: {entry.sent_at.isoformat()}"
                for entry in history
            ]

            raw_prompt = getattr(subscription, "canonical_prompt", "") or subscription.raw_prompt
            try:
                assessment = await assess_and_compose_event_notification(
                    headline=item.headline,
                    body=item.body,
                    published_at=item.published_at,
                    raw_prompt=raw_prompt,
                    target_language=subscription.digest_language,
                    event_matching_mode=subscription.event_matching_mode,
                    recent_notification_history=history_strings,
                    max_history_chars=settings.llm_max_context_chars,
                )
            except Exception:
                logger.exception(
                    "Failed to assess event for subscription %s",
                    subscription.id,
                    extra={
                        "subscription_id": str(subscription.id),
                        "news_item_id": str(item.id),
                    },
                )
                continue

            if not assessment.is_relevant_event:
                logger.info(
                    "Event %s not relevant for subscription %s: %s",
                    item.id,
                    subscription.id,
                    assessment.reason,
                    extra={
                        "subscription_id": str(subscription.id),
                        "news_item_id": str(item.id),
                    },
                )
                continue

            channel = get_delivery_channel(subscription.delivery_webhook_url)
            try:
                await channel.send("", assessment.notification_body)
            except Exception:
                failed += 1
                logger.exception(
                    "Failed to deliver event notification for subscription %s",
                    subscription.id,
                    extra={
                        "subscription_id": str(subscription.id),
                        "news_item_id": str(item.id),
                    },
                )
                continue

            session.add(SentItem(subscription_id=subscription.id, news_item_id=item.id))
            await session.commit()
            delivered += 1

        if delivered == 0 and failed == 0:
            return {"status": "skipped", "reason": "already_sent"}

        status = "delivered"
        if delivered > 0 and failed > 0:
            status = "partial"
        elif delivered == 0 and failed > 0:
            status = "failed"

        return {
            "status": status,
            "delivered": delivered,
            "failed": failed,
            "news_item_id": str(news_item_id),
        }
