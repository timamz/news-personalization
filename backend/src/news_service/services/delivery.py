import logging
from abc import ABC, abstractmethod

import httpx

logger = logging.getLogger(__name__)


class DeliveryChannel(ABC):
    @abstractmethod
    async def send(self, subject: str, body: str) -> None:
        pass


class WebhookChannel(DeliveryChannel):
    def __init__(self, url: str) -> None:
        self.url = url

    async def send(self, subject: str, body: str) -> None:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(self.url, json={"subject": subject, "body": body})
                response.raise_for_status()
            logger.info("Webhook delivered to %s: %s", self.url, subject)
        except httpx.HTTPError:
            logger.exception("Webhook delivery failed for %s", self.url)
            raise


class LogChannel(DeliveryChannel):
    async def send(self, subject: str, body: str) -> None:
        logger.info("Digest delivery [%s]:\n%s", subject, body[:500])


def get_delivery_channel(webhook_url: str | None = None) -> DeliveryChannel:
    if webhook_url:
        return WebhookChannel(webhook_url)
    return LogChannel()
