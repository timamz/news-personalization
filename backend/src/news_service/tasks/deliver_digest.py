import asyncio
import logging
import uuid

from sqlalchemy import select

from news_service.agents.digest import generate_digest
from news_service.db.session import async_session_factory
from news_service.models.subscription import Subscription
from news_service.services.delivery import get_delivery_channel
from news_service.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="news_service.tasks.deliver_digest.deliver_digest")
def deliver_digest(subscription_id: str) -> dict:
    return asyncio.run(_deliver_digest(uuid.UUID(subscription_id)))


async def _deliver_digest(subscription_id: uuid.UUID) -> dict:
    async with async_session_factory() as session:
        result = await session.execute(
            select(Subscription).where(Subscription.id == subscription_id)
        )
        subscription = result.scalar_one_or_none()

        if subscription is None or not subscription.is_active:
            logger.warning("Subscription %s not found or inactive", subscription_id)
            return {"status": "skipped", "reason": "not_found_or_inactive"}

        digest_text = await generate_digest(session, subscription)
        if digest_text is None:
            return {"status": "skipped", "reason": "no_new_items"}

        await session.commit()

    channel = get_delivery_channel(subscription.delivery_webhook_url)
    topic_summary = ", ".join(subscription.topics[:3])
    subject = f"Your News Digest: {topic_summary}"
    await channel.send(subject, digest_text)

    return {"status": "delivered", "subscription_id": str(subscription_id)}
