import logging
from abc import ABC, abstractmethod
from email.message import EmailMessage

import aiosmtplib
import httpx

from news_service.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


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


class EmailChannel(DeliveryChannel):
    def __init__(self, recipient: str) -> None:
        self.recipient = recipient

    async def send(self, subject: str, body: str) -> None:
        message = EmailMessage()
        message["From"] = settings.smtp_from
        message["To"] = self.recipient
        message["Subject"] = subject
        message.set_content(body)

        try:
            await aiosmtplib.send(
                message,
                hostname=settings.smtp_host,
                port=settings.smtp_port,
                username=settings.smtp_user or None,
                password=settings.smtp_password or None,
                start_tls=True,
            )
            logger.info("Email sent to %s: %s", self.recipient, subject)
        except aiosmtplib.SMTPException:
            logger.exception("Failed to send email to %s", self.recipient)
            raise


class LogChannel(DeliveryChannel):
    async def send(self, subject: str, body: str) -> None:
        logger.info("Digest delivery [%s]:\n%s", subject, body[:500])


def get_delivery_channel(webhook_url: str | None = None) -> DeliveryChannel:
    if webhook_url:
        return WebhookChannel(webhook_url)
    return LogChannel()
