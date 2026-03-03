import asyncio
import logging
import uuid

from sqlalchemy import select

from news_service.agents.digest import generate_digest
from news_service.db.session import get_task_session
from news_service.models.subscription import Subscription
from news_service.services.delivery import get_delivery_channel
from news_service.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="news_service.tasks.deliver_digest.deliver_digest")
def deliver_digest(subscription_id: str, notify_if_empty: bool = False) -> dict:
    return asyncio.run(_deliver_digest(uuid.UUID(subscription_id), notify_if_empty))


async def _deliver_digest(subscription_id: uuid.UUID, notify_if_empty: bool = False) -> dict:
    async with get_task_session() as session:
        result = await session.execute(
            select(Subscription).where(Subscription.id == subscription_id)
        )
        subscription = result.scalar_one_or_none()

        if subscription is None or not subscription.is_active:
            logger.warning("Subscription %s not found or inactive", subscription_id)
            return {"status": "skipped", "reason": "not_found_or_inactive"}
        if subscription.delivery_mode != "digest":
            logger.warning(
                "Subscription %s is not a digest subscription",
                subscription_id,
                extra={"subscription_id": str(subscription_id)},
            )
            return {"status": "skipped", "reason": "wrong_delivery_mode"}

        digest_text = await generate_digest(session, subscription)
        if digest_text is None:
            if notify_if_empty:
                channel = get_delivery_channel(subscription.delivery_webhook_url)
                await channel.send(
                    "No new updates right now",
                    "No new articles since your last digest. Try again a bit later.",
                )
                return {"status": "notified", "reason": "no_new_items"}
            return {"status": "skipped", "reason": "no_new_items"}
        channel = get_delivery_channel(subscription.delivery_webhook_url)
        topic_summary = ", ".join(subscription.topics[:3])
        subject = f"Your News Digest: {topic_summary}"
        await channel.send(subject, digest_text)

        await session.commit()
        return {"status": "delivered", "subscription_id": str(subscription_id)}
