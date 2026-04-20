"""
FakeDelivery captures every webhook POST the backend attempts.

Replaces news_service.services.delivery.deliver. Instead of sending HTTP,
it appends a CapturedWebhook record to an in-memory log, tagged with the
current virtual timestamp. Assertions and metrics read from this log.

The log is intentionally keyed by delivery_webhook_url rather than by
subscription, because the backend passes the URL verbatim — the
scenario pre-registers synthetic URLs like
`https://bench.invalid/sub/<sub_id>/digest` per subscription, so
matching is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from news_benchmark.clock import CLOCK


@dataclass
class CapturedWebhook:
    url: str | None
    subject: str
    body: str
    fake_clock: datetime


@dataclass
class FakeDelivery:
    """Stub webhook delivery that records payloads instead of sending them."""

    captured: list[CapturedWebhook] = field(default_factory=list)

    async def deliver(self, webhook_url: str | None, subject: str, body: str) -> None:
        self.captured.append(
            CapturedWebhook(
                url=webhook_url,
                subject=subject,
                body=body,
                fake_clock=CLOCK.now(),
            )
        )

    def for_url(self, url: str) -> list[CapturedWebhook]:
        return [c for c in self.captured if c.url == url]

    def clear(self) -> None:
        self.captured.clear()
