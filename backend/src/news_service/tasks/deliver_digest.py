import asyncio
import logging
import uuid

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from news_service.agents.digest import generate_digest
from news_service.core.exceptions import DigestPipelineError
from news_service.core.llm_usage import subscription_tag
from news_service.core.provider_errors import ProviderLimitError
from news_service.db.session import get_task_session
from news_service.models.subscription import Subscription
from news_service.services.delivery import deliver
from news_service.tasks.celery_app import celery_app
from news_service.tasks.retry_policy import retry_on_provider_limit

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="news_service.tasks.deliver_digest.deliver_digest")
def deliver_digest(self, subscription_id: str, notify_if_empty: bool = False) -> dict:
    sub_uuid = uuid.UUID(subscription_id)
    with subscription_tag(sub_uuid):
        try:
            return asyncio.run(_deliver_digest(sub_uuid, notify_if_empty))
        except ProviderLimitError as exc:
            raise retry_on_provider_limit(self, exc) from exc


async def _deliver_digest(subscription_id: uuid.UUID, notify_if_empty: bool = False) -> dict:
    async with get_task_session() as session:
        result = await session.execute(
            select(Subscription)
            .options(selectinload(Subscription.user))
            .where(Subscription.id == subscription_id)
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

        try:
            digest_text = await generate_digest(session, subscription)
        except DigestPipelineError:
            logger.exception(
                "Digest pipeline failed for subscription %s",
                subscription_id,
                extra={"subscription_id": str(subscription_id)},
            )
            return {"status": "failed", "reason": "pipeline_error"}

        if digest_text is None:
            if notify_if_empty:
                webhook_url = subscription.delivery_webhook_url
                if webhook_url is None and subscription.user is not None:
                    webhook_url = subscription.user.delivery_webhook_url
                await deliver(
                    webhook_url,
                    "No new updates right now",
                    "No new articles since your last digest. Try again a bit later.",
                )
                return {"status": "notified", "reason": "no_new_items"}
            return {"status": "skipped", "reason": "no_new_items"}
        webhook_url = subscription.delivery_webhook_url
        if webhook_url is None and subscription.user is not None:
            webhook_url = subscription.user.delivery_webhook_url
        await deliver(webhook_url, "", digest_text)

        await session.commit()
        return {"status": "delivered", "subscription_id": str(subscription_id)}
