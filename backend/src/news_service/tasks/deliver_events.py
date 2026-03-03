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
from news_service.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

_LABELS = {
    "en": {
        "subject": "Upcoming event",
        "event": "Event",
        "when": "When",
        "source": "Source",
    },
    "ru": {
        "subject": "Predstoyashchee sobytie",
        "event": "Sobytiye",
        "when": "Kogda",
        "source": "Istochnik",
    },
}


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
        for subscription in subscriptions:
            if subscription.id in sent_subscription_ids:
                continue

            channel = get_delivery_channel(subscription.delivery_webhook_url)
            subject, body = _build_notification(subscription.digest_language, item)
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


def _build_notification(digest_language: str, item: NewsItem) -> tuple[str, str]:
    labels = _labels_for(digest_language)
    subject = f"{labels['subject']}: {item.event_title}"
    lines = [f"{labels['event']}: {item.event_title}"]
    if item.event_starts_at is not None:
        lines.append(f"{labels['when']}: {_format_event_time(item.event_starts_at)}")

    summary = item.event_summary or item.headline
    if summary:
        lines.extend(["", summary])

    lines.extend(["", f"{labels['source']}: {item.source}", item.url])
    return subject, "\n".join(lines)


def _labels_for(digest_language: str) -> dict[str, str]:
    normalized = digest_language.lower().split("-", maxsplit=1)[0]
    return _LABELS.get(normalized, _LABELS["en"])


def _format_event_time(value: datetime) -> str:
    value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return value.strftime("%Y-%m-%d %H:%M UTC")
