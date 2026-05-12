"""Periodic event verifier task -- runs the Event Verifier per active event subscription.

Schedule:
- Beat fires daily; the task self-throttles per subscription via
  ``last_reflected_at`` so each event sub is actually verified once per
  ``event_reflector_interval_days``.

Per subscription:
1. Load user_spec, recent notification history, and per-source context.
2. Run the Event Verifier agent.
3. For each missed_event the agent recorded:
   - Insert a synthetic NewsItem (source = the shared "_verifier" sentinel
     Source row) + a SentItem pointing to it so the next verifier window's history
     includes the catch-up.
   - Deliver the catch-up via the sub's webhook.
4. For each discovery_reason the agent queued, dispatch
   DISCOVER_SOURCES_TASK via celery_app.send_task.
5. Stamp subscription.last_reflected_at.

Per CLAUDE.md error-handling tier 3 (non-blocking): per-sub failures are
logged and swallowed so one bad sub does not abort siblings.
"""

import asyncio
import logging
import uuid
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from news_service.agents.event.verifier import (
    MissedEvent,
    VerifierSourceContext,
    run_event_verifier,
)
from news_service.core.config import get_settings
from news_service.core.llm_usage import subscription_tag
from news_service.core.provider_errors import ProviderLimitError
from news_service.db.session import get_task_session
from news_service.models.news_item import NewsItem
from news_service.models.sent_item import SentItem
from news_service.models.source import Source
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.services.delivery import deliver
from news_service.services.event_notifications import load_recent_notification_history
from news_service.tasks.celery_app import celery_app
from news_service.tasks.retry_policy import retry_on_provider_limit

logger = logging.getLogger(__name__)
settings = get_settings()

VERIFIER_SENTINEL_SOURCE_URL = "https://verifier.internal/synthetic"
VERIFIER_SENTINEL_SOURCE_TITLE = "_verifier"


@celery_app.task(bind=True, name="news_service.tasks.reflect_events.reflect_event_subscriptions")
def reflect_event_subscriptions(self) -> dict:
    try:
        return asyncio.run(_reflect_event_subscriptions())
    except ProviderLimitError as exc:
        raise retry_on_provider_limit(self, exc) from exc


async def _reflect_event_subscriptions() -> dict:
    """Select due event subscriptions and run the verifier for each."""
    async with get_task_session() as session:
        cutoff = datetime.now(UTC) - timedelta(days=settings.event_reflector_interval_days)
        result = await session.execute(
            select(Subscription).where(
                Subscription.is_active.is_(True),
                Subscription.delivery_mode == "event",
                or_(
                    Subscription.last_reflected_at.is_(None),
                    Subscription.last_reflected_at < cutoff,
                ),
            )
        )
        due_subs = list(result.scalars().all())

    if not due_subs:
        return {"status": "skipped", "reason": "no_due_subscriptions"}

    sem = asyncio.Semaphore(settings.recent_event_match_concurrency)
    tasks = [_verify_one(sub_id=sub.id, sem=sem) for sub in due_subs]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    processed = 0
    failed = 0
    total_misses = 0
    total_discovery_triggers = 0
    for sub, outcome in zip(due_subs, outcomes, strict=True):
        if isinstance(outcome, ProviderLimitError):
            raise outcome
        if isinstance(outcome, BaseException):
            failed += 1
            logger.exception(
                "Event verifier failed for subscription %s: %s",
                sub.id,
                outcome,
                extra={"subscription_id": str(sub.id)},
            )
            continue
        processed += 1
        total_misses += outcome["delivered_misses"]
        total_discovery_triggers += outcome["discovery_queued"]

    return {
        "status": "done",
        "processed": processed,
        "failed": failed,
        "delivered_misses": total_misses,
        "discovery_queued": total_discovery_triggers,
    }


async def _verify_one(*, sub_id: uuid.UUID, sem: asyncio.Semaphore) -> dict:
    """Run the verifier for a single subscription in its own DB session."""
    with subscription_tag(sub_id):
        return await _verify_one_tagged(sub_id=sub_id, sem=sem)


async def _verify_one_tagged(*, sub_id: uuid.UUID, sem: asyncio.Semaphore) -> dict:
    async with sem, get_task_session() as session:
        sub = (
            await session.execute(
                select(Subscription)
                .options(selectinload(Subscription.user))
                .where(Subscription.id == sub_id)
            )
        ).scalar_one()

        history = await load_recent_notification_history(
            session,
            sub.id,
            lookback_days=settings.event_verifier_lookback_days,
        )
        history_strings = [
            f"Title: {entry.title}\nSummary: {entry.summary}\n"
            f"Source: {entry.source}\nShown at: {entry.sent_at.isoformat()}"
            for entry in history
        ]

        source_contexts = await _load_source_contexts(session, sub.id)

        state = await run_event_verifier(
            db_session=session,
            subscription=sub,
            user_spec=sub.user_spec,
            history_strings=history_strings,
            source_contexts=source_contexts,
            lookback_days=settings.event_verifier_lookback_days,
        )

        delivered_misses = 0
        for miss in state["missed_events"]:
            try:
                await _deliver_and_record_miss(
                    session=session,
                    subscription=sub,
                    miss=miss,
                )
                delivered_misses += 1
            except Exception:
                logger.exception(
                    "Failed to deliver catch-up for subscription %s miss=%s",
                    sub.id,
                    miss.title[:80],
                    extra={"subscription_id": str(sub.id)},
                )

        for reason in state["discovery_reasons"]:
            celery_app.send_task(
                "news_service.tasks.discover_sources.discover_sources_for_subscription",
                args=[str(sub.id), reason],
            )
            logger.info(
                "Event verifier queued discovery for subscription %s: %s",
                sub.id,
                reason[:100],
            )

        for status in state["status_messages"]:
            logger.info(
                "Event verifier status for subscription %s: %s",
                sub.id,
                status[:200],
            )

        sub.last_reflected_at = datetime.now(UTC)
        await session.commit()

        return {
            "delivered_misses": delivered_misses,
            "discovery_queued": len(state["discovery_reasons"]),
        }


async def _load_source_contexts(
    session: AsyncSession,
    subscription_id: uuid.UUID,
) -> list[VerifierSourceContext]:
    """Gather per-source metadata the verifier uses before searching."""
    result = await session.execute(
        select(Source, SubscriptionSource.is_user_specified)
        .join(SubscriptionSource, SubscriptionSource.source_id == Source.id)
        .where(SubscriptionSource.subscription_id == subscription_id)
    )
    rows = list(result.all())
    if not rows:
        return []

    window_cutoff = datetime.now(UTC) - timedelta(days=settings.event_verifier_lookback_days)
    contexts: list[VerifierSourceContext] = []
    for source, is_user_specified in rows:
        window_count = (
            await session.execute(
                select(func.count(NewsItem.id)).where(
                    NewsItem.source_id == source.id,
                    NewsItem.published_at.is_not(None),
                    NewsItem.published_at >= window_cutoff,
                )
            )
        ).scalar_one()
        last_published_at = (
            await session.execute(
                select(func.max(NewsItem.published_at)).where(
                    NewsItem.source_id == source.id,
                    NewsItem.published_at.is_not(None),
                )
            )
        ).scalar_one()
        contexts.append(
            VerifierSourceContext(
                source_id=source.id,
                url=source.url,
                title=source.title,
                is_user_specified=bool(is_user_specified),
                last_published_at=last_published_at,
                items_in_window=int(window_count),
            )
        )
    return contexts


async def _deliver_and_record_miss(
    *,
    session: AsyncSession,
    subscription: Subscription,
    miss: MissedEvent,
) -> None:
    """Record a catch-up NewsItem + SentItem and deliver via the sub webhook."""
    sentinel = await _get_or_create_verifier_source(session)

    now = datetime.now(UTC)
    existing = (
        await session.execute(select(NewsItem).where(NewsItem.url == miss.source_url))
    ).scalar_one_or_none()
    if existing is not None:
        news_item = existing
    else:
        news_item = NewsItem(
            source_id=sentinel.id,
            headline=miss.title,
            body=miss.summary,
            url=miss.source_url,
            source=VERIFIER_SENTINEL_SOURCE_TITLE,
            published_at=now,
            fetched_at=now,
        )
        session.add(news_item)
        await session.flush()

    already_sent = (
        await session.execute(
            select(SentItem).where(
                SentItem.subscription_id == subscription.id,
                SentItem.news_item_id == news_item.id,
            )
        )
    ).scalar_one_or_none()
    if already_sent is not None:
        return

    body_text = _format_catch_up_body(miss)
    webhook_url = subscription.delivery_webhook_url
    if webhook_url is None and subscription.user is not None:
        webhook_url = subscription.user.delivery_webhook_url
    await deliver(webhook_url, "", body_text)
    session.add(SentItem(subscription_id=subscription.id, news_item_id=news_item.id))
    await session.flush()


async def _get_or_create_verifier_source(session: AsyncSession) -> Source:
    existing = (
        await session.execute(select(Source).where(Source.url == VERIFIER_SENTINEL_SOURCE_URL))
    ).scalar_one_or_none()
    if existing is not None:
        return existing
    sentinel = Source(
        url=VERIFIER_SENTINEL_SOURCE_URL,
        title=VERIFIER_SENTINEL_SOURCE_TITLE,
        source_description=(
            "Synthetic container for event-verifier catch-up deliveries. Never polled."
        ),
        is_active=False,
        subscriber_count=0,
    )
    session.add(sentinel)
    await session.flush()
    return sentinel


def _format_catch_up_body(miss: MissedEvent) -> str:
    """Render a short plain-text catch-up notification body."""
    lines = [miss.title.strip()]
    if miss.summary:
        lines.append(miss.summary.strip())
    if miss.happened_at and miss.happened_at != "unknown":
        lines.append(f"When: {miss.happened_at.strip()}")
    lines.append(miss.source_url.strip())
    return "\n".join(line for line in lines if line)
