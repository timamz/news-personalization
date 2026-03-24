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
from news_service.services.event_notifications import (
    load_recent_notification_history,
)
from news_service.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)
settings = get_settings()


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
                SubscriptionSource.source_id == item.source_id,
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

        pending = [s for s in subscriptions if s.id not in sent_subscription_ids]
        if not pending:
            return {"status": "skipped", "reason": "already_sent"}

        sem = asyncio.Semaphore(settings.recent_event_match_concurrency)

        async def _process_subscription(
            subscription: Subscription,
        ) -> str:
            """Process a single subscription. Returns 'delivered', 'failed', or 'skipped'."""
            async with sem:
                history = await load_recent_notification_history(session, subscription.id)

                history_strings = [
                    f"Title: {entry.title}\nSummary: {entry.summary}\n"
                    f"Source: {entry.source}\nShown at: {entry.sent_at.isoformat()}"
                    for entry in history
                ]

                raw_prompt = subscription.canonical_prompt
                try:
                    assessment = await assess_and_compose_event_notification(
                        headline=item.headline,
                        body=item.body,
                        url=item.url,
                        published_at=item.published_at,
                        raw_prompt=raw_prompt,
                        target_language=subscription.digest_language,
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
                    return "failed"

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
                    return "skipped"

                channel = get_delivery_channel(subscription.delivery_webhook_url)
                try:
                    await channel.send("", assessment.notification_body)
                except Exception:
                    logger.exception(
                        "Failed to deliver event notification for subscription %s",
                        subscription.id,
                        extra={
                            "subscription_id": str(subscription.id),
                            "news_item_id": str(item.id),
                        },
                    )
                    return "failed"

                session.add(SentItem(subscription_id=subscription.id, news_item_id=item.id))
                await session.commit()
                return "delivered"

        results = await asyncio.gather(
            *[_process_subscription(s) for s in pending],
            return_exceptions=True,
        )

        delivered = 0
        failed = 0
        for r in results:
            if isinstance(r, BaseException):
                failed += 1
            elif r == "delivered":
                delivered += 1
            elif r == "failed":
                failed += 1

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
