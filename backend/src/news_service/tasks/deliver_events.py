import asyncio
import logging
import uuid
from datetime import UTC, datetime

from sqlalchemy import select

from news_service.db.session import get_task_session
from news_service.models.news_item import NewsItem
from news_service.models.sent_item import SentItem
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.services.delivery import get_delivery_channel
from news_service.services.event_notifications import (
    RecentNotificationEntry,
    build_event_notification,
    load_recent_notification_history,
    notification_history_entry_from_item,
    notification_was_already_shown,
    subscription_matches_event,
)
from news_service.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="news_service.tasks.deliver_events.deliver_event_notifications")
def deliver_event_notifications(news_item_id: str) -> dict:
    return asyncio.run(_deliver_event_notifications(uuid.UUID(news_item_id)))


async def _deliver_event_notifications(news_item_id: uuid.UUID) -> dict:
    async with get_task_session() as session:
        item = await session.get(NewsItem, news_item_id)
        if item is None:
            logger.warning("News item %s not found for event delivery", news_item_id)
            return {"status": "skipped", "reason": "news_item_not_found"}
        if not item.event_title:
            return {"status": "skipped", "reason": "not_event"}

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
        history_cache: dict[uuid.UUID, list[RecentNotificationEntry]] = {}
        notification_cache: dict[str, tuple[str, str]] = {}
        for subscription in subscriptions:
            if subscription.id in sent_subscription_ids:
                continue
            try:
                if not await subscription_matches_event(subscription, item):
                    continue
            except Exception:
                logger.exception(
                    "Failed to evaluate event match for subscription %s",
                    subscription.id,
                    extra={
                        "subscription_id": str(subscription.id),
                        "news_item_id": str(item.id),
                    },
                )
                continue

            if subscription.id not in history_cache:
                try:
                    history_cache[subscription.id] = await load_recent_notification_history(
                        session,
                        subscription.id,
                    )
                except Exception:
                    logger.exception(
                        "Failed to load recent notification history for subscription %s",
                        subscription.id,
                        extra={
                            "subscription_id": str(subscription.id),
                            "news_item_id": str(item.id),
                        },
                    )
                    history_cache[subscription.id] = []

            try:
                if await notification_was_already_shown(item, history_cache[subscription.id]):
                    continue
            except Exception:
                logger.exception(
                    "Failed to evaluate duplicate notification status for subscription %s",
                    subscription.id,
                    extra={
                        "subscription_id": str(subscription.id),
                        "news_item_id": str(item.id),
                    },
                )

            channel = get_delivery_channel(subscription.delivery_webhook_url)
            notification = notification_cache.get(subscription.digest_language)
            if notification is None:
                notification = await build_event_notification(subscription.digest_language, item)
                notification_cache[subscription.digest_language] = notification
            subject, body = notification
            try:
                await channel.send(subject, body)
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
            history_cache[subscription.id].insert(
                0,
                notification_history_entry_from_item(item, sent_at=datetime.now(UTC)),
            )
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
