import logging
from abc import ABC, abstractmethod
from email.message import EmailMessage

import aiosmtplib

from news_service.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class DeliveryChannel(ABC):
    @abstractmethod
    async def send(self, recipient: str, subject: str, body: str) -> None:
        pass


class EmailChannel(DeliveryChannel):
    async def send(self, recipient: str, subject: str, body: str) -> None:
        message = EmailMessage()
        message["From"] = settings.smtp_from
        message["To"] = recipient
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
            logger.info("Email sent to %s: %s", recipient, subject)
        except aiosmtplib.SMTPException:
            logger.exception("Failed to send email to %s", recipient)
            raise


class LogChannel(DeliveryChannel):
    """Delivery channel that logs the digest. Useful for development and testing."""

    async def send(self, recipient: str, subject: str, body: str) -> None:
        logger.info(
            "Digest delivery [%s] to %s:\n%s",
            subject,
            recipient,
            body[:500],
        )


def get_delivery_channel() -> DeliveryChannel:
    if settings.smtp_user:
        return EmailChannel()
    return LogChannel()
