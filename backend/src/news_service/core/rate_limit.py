"""Per-user fixed-window rate limiting on top of Redis.

A single ``INCR`` + first-time ``EXPIRE`` per window. Cheap and accurate
enough for "do not let one user or stolen API key burn the LLM budget".
Not a smooth-sliding limiter and not strict-once-per-second; for those
we'd reach for a Lua script or a token bucket. The fixed window is fine
for the budgets we set (tens to hundreds per hour / day per user).

Usage:

    from news_service.core.rate_limit import check_rate_limit

    await check_rate_limit(
        scope="conversation",
        subject_id=str(user.id),
        limit=settings.rate_limit_conversation_per_hour,
        window_seconds=3600,
    )

Raises ``RateLimitExceeded`` with the remaining seconds for the caller
to surface as a 429 or a clear chat message.
"""

from __future__ import annotations

import logging

from news_service.core.redis import get_redis_client

logger = logging.getLogger(__name__)


class RateLimitExceeded(Exception):
    """Raised when a per-user limit is hit. Carries the retry-after window."""

    def __init__(self, scope: str, subject_id: str, limit: int, retry_after_seconds: int) -> None:
        super().__init__(
            f"rate limit exceeded for scope={scope!r} subject={subject_id!r} "
            f"limit={limit} retry_after={retry_after_seconds}s"
        )
        self.scope = scope
        self.subject_id = subject_id
        self.limit = limit
        self.retry_after_seconds = retry_after_seconds


def _key(scope: str, subject_id: str) -> str:
    return f"ratelimit:{scope}:{subject_id}"


async def check_rate_limit(
    *,
    scope: str,
    subject_id: str,
    limit: int,
    window_seconds: int,
) -> None:
    """Increment the counter for (scope, subject) and raise if over limit.

    Counter expires at ``window_seconds`` from the first hit in the
    window. Subsequent hits within the same window do not bump the TTL,
    so the window is strictly fixed.

    Fail-open on Redis errors: a transient Redis outage must not lock
    every user out of the API. Operational alerting on Redis health
    is the right place to catch that, not the request path.
    """
    client = get_redis_client()
    key = _key(scope, subject_id)
    try:
        new_value = await client.incr(key)
        if new_value == 1:
            await client.expire(key, window_seconds)
        if new_value > limit:
            ttl = await client.ttl(key)
            retry = ttl if isinstance(ttl, int) and ttl > 0 else window_seconds
            raise RateLimitExceeded(
                scope=scope,
                subject_id=subject_id,
                limit=limit,
                retry_after_seconds=retry,
            )
    except RateLimitExceeded:
        raise
    except Exception:
        logger.exception(
            "Rate limit check failed for scope=%s subject=%s; allowing request",
            scope,
            subject_id,
        )
