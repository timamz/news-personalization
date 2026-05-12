"""Celery task retry policy for upstream provider limit failures.

When a Celery task fails because an LLM or search provider is out of
balance / rate-limited / unauthenticated, we do not want the task to
land in the dead-letter table immediately. The operator gets a single
throttled alert from ``services.admin_alerts``; this module lets the
task itself defer execution by 30 minutes and keep retrying for up to
24 hours. That window covers a normal "I'll top the balance up later
today" recovery without losing work.

Usage::

    @celery_app.task(bind=True, name=...)
    def my_task(self, ...):
        try:
            return asyncio.run(_my_impl(...))
        except ProviderLimitError as exc:
            raise retry_on_provider_limit(self, exc) from exc

The helper centralizes the retry kwargs so every task picks them up
from settings without duplication.
"""

from __future__ import annotations

import logging

from celery import Task
from celery.exceptions import Retry

from news_service.core.config import get_settings
from news_service.core.provider_errors import ProviderLimitError

logger = logging.getLogger(__name__)


def retry_on_provider_limit(task: Task, exc: ProviderLimitError) -> Retry:
    """Schedule the bound task to retry after the configured countdown.

    Returns the ``Retry`` instance produced by ``task.retry`` so the
    caller can ``raise`` it (Celery uses it as the actual retry signal).
    If ``max_retries`` has already been exhausted, ``task.retry`` raises
    ``MaxRetriesExceededError`` which we let propagate to the failure
    handler -- the dead-letter table is the right destination after
    24 hours of trying.
    """
    settings = get_settings()
    countdown = settings.provider_failure_retry_countdown_seconds
    max_retries = settings.provider_failure_retry_max_attempts
    attempt = getattr(task.request, "retries", 0) or 0
    logger.warning(
        "Provider limit hit in task %s (provider=%s kind=%s); scheduling retry %d/%d in %ds",
        task.name,
        exc.provider,
        exc.kind,
        attempt + 1,
        max_retries,
        countdown,
    )
    return task.retry(exc=exc, countdown=countdown, max_retries=max_retries)
