"""Operator-only alerts for provider failures.

When an LLM or search provider returns a usage-limit error (balance,
auth, persistent rate-limit), only the configured admin user should
hear about it. End users do not need a notification for every quota
blip, and a worker hung by an outage was producing far too many.

The path here is intentionally minimal: classify the error upstream
(``core.provider_errors``), then call ``notify_provider_limit`` from
the LLM / search wrapper sites. Delivery is HTTP POST to
``admin_alert_webhook_url`` -- typically the tgbot's regular
``/deliver/{token}/{chat_id}`` endpoint pointed at the admin's chat.
Throttle keys live in Redis (one alert per (provider, kind) per
``admin_alert_throttle_seconds``) so a sustained outage does not
flood the admin chat.

Failures here are best-effort: if Redis is down or the bot is
unreachable, log and continue -- the alert is the lowest-priority
side effect of the failing call.
"""

from __future__ import annotations

import logging

import httpx

from news_service.core.config import get_settings
from news_service.core.provider_errors import ProviderKind, ProviderLimitError
from news_service.core.redis import get_redis_client

logger = logging.getLogger(__name__)


_MESSAGE_SNIPPET_MAX = 800


async def notify_provider_limit(err: ProviderLimitError) -> None:
    """Send an admin alert for a provider limit error, throttled per (provider, kind)."""
    await _notify(provider=err.provider, kind=err.kind, message=err.message)


async def _notify(*, provider: str, kind: ProviderKind, message: str) -> None:
    settings = get_settings()
    webhook_url = settings.admin_alert_webhook_url
    if not webhook_url:
        logger.warning(
            "Provider limit hit but ADMIN_ALERT_WEBHOOK_URL is unset; "
            "provider=%s kind=%s message=%s",
            provider,
            kind,
            message[:_MESSAGE_SNIPPET_MAX],
        )
        return

    throttle_key = f"alert:provider_limit:{provider}:{kind}"
    try:
        acquired = await get_redis_client().set(
            throttle_key, "1", ex=settings.admin_alert_throttle_seconds, nx=True
        )
    except Exception:
        logger.exception("Redis throttle check failed for admin alert; sending anyway")
        acquired = True

    if not acquired:
        logger.info(
            "Suppressed duplicate admin alert (within throttle window): provider=%s kind=%s",
            provider,
            kind,
        )
        return

    subject = f"Provider limit: {kind} ({provider})"
    body = _format_admin_body(provider=provider, kind=kind, message=message)
    try:
        async with httpx.AsyncClient(timeout=settings.http_timeout_seconds) as client:
            response = await client.post(webhook_url, json={"subject": subject, "body": body})
            response.raise_for_status()
    except Exception:
        logger.exception("Failed to deliver admin alert to %s", webhook_url)


def _format_admin_body(*, provider: str, kind: ProviderKind, message: str) -> str:
    snippet = message.strip()
    if len(snippet) > _MESSAGE_SNIPPET_MAX:
        snippet = snippet[:_MESSAGE_SNIPPET_MAX] + "..."
    explanation = _KIND_EXPLANATION.get(kind, "")
    lines = [
        f"Provider: {provider}",
        f"Kind: {kind}",
    ]
    if explanation:
        lines.append(explanation)
    lines.append("")
    lines.append("Background tasks will retry every 30 min for the next 24 h.")
    lines.append("Original error:")
    lines.append(snippet)
    return "\n".join(lines)


_KIND_EXPLANATION: dict[ProviderKind, str] = {
    "balance": "Insufficient balance or quota -- top up the provider account.",
    "rate_limit": "Persistent rate limit -- provider is throttling our key.",
    "auth": "Authentication failure -- key revoked, expired, or missing role.",
    "timeout": "Provider call timed out.",
    "connection": "Could not reach provider.",
    "unknown": "Unclassified provider failure.",
}
