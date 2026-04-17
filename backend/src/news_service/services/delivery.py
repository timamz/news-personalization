"""Deliver a payload to the configured frontend webhook.

When the subscription has no webhook URL (local development, unconfigured
frontend), the payload is logged at INFO instead of being sent over HTTP.
"""

import logging

import httpx

from news_service.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def deliver(webhook_url: str | None, subject: str, body: str) -> None:
    if not webhook_url:
        logger.info("Delivery without webhook_url [%s]:\n%s", subject, body[:500])
        return
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
            response = await client.post(webhook_url, json={"subject": subject, "body": body})
            response.raise_for_status()
        logger.info("Webhook delivered to %s: %s", webhook_url, subject)
    except httpx.HTTPError:
        logger.exception("Webhook delivery failed for %s", webhook_url)
        raise
