"""Batch event notification delivery.

Receives a batch of news item IDs from a single polling cycle.
Groups items by subscription and runs one LLM call per subscription
(batch assessment) instead of one per item.
"""

import asyncio
import logging
import uuid

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from news_service.agents.event.batch_assessor import (
    BatchAssessmentResult,
    assess_batch_events,
)
from news_service.agents.event.judge import judge_batch_events
from news_service.core.config import get_settings
from news_service.core.guardrails import validate_notification_body
from news_service.db.session import get_task_session
from news_service.models.news_item import NewsItem
from news_service.models.sent_item import SentItem
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.services.delivery import deliver
from news_service.services.event_notifications import load_recent_notification_history
from news_service.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)
settings = get_settings()


@celery_app.task(name="news_service.tasks.deliver_events.deliver_event_notifications_batch")
def deliver_event_notifications_batch(news_item_ids: list[str]) -> dict:
    return asyncio.run(
        _deliver_event_notifications_batch([uuid.UUID(nid) for nid in news_item_ids])
    )


@celery_app.task(name="news_service.tasks.deliver_events.deliver_event_notifications")
def deliver_event_notifications(news_item_id: str) -> dict:
    """Single-item entry point for backward compatibility."""
    return asyncio.run(_deliver_event_notifications_batch([uuid.UUID(news_item_id)]))


async def _deliver_event_notifications_batch(news_item_ids: list[uuid.UUID]) -> dict:
    """Process a batch of new items, grouped by subscription."""
    if not news_item_ids:
        return {"status": "skipped", "reason": "empty_batch"}

    async with get_task_session() as session:
        items_result = await session.execute(select(NewsItem).where(NewsItem.id.in_(news_item_ids)))
        items = list(items_result.scalars().all())
        if not items:
            return {"status": "skipped", "reason": "no_items_found"}

        source_ids = {item.source_id for item in items}
        sub_result = await session.execute(
            select(Subscription)
            .options(selectinload(Subscription.user))
            .join(SubscriptionSource, SubscriptionSource.subscription_id == Subscription.id)
            .where(
                Subscription.is_active.is_(True),
                Subscription.delivery_mode == "event",
                SubscriptionSource.source_id.in_(list(source_ids)),
            )
            .distinct()
        )
        subscriptions = list(sub_result.scalars().all())
        if not subscriptions:
            return {"status": "skipped", "reason": "no_matching_subscriptions"}

        sem = asyncio.Semaphore(settings.recent_event_match_concurrency)
        total_delivered = 0
        total_failed = 0

        async def _process_subscription(subscription: Subscription) -> tuple[int, int]:
            async with sem:
                return await _assess_and_deliver_for_subscription(session, subscription, items)

        results = await asyncio.gather(
            *[_process_subscription(s) for s in subscriptions],
            return_exceptions=True,
        )

        for r in results:
            if isinstance(r, BaseException):
                total_failed += 1
                logger.exception("Subscription processing failed: %s", r)
            else:
                delivered, failed = r
                total_delivered += delivered
                total_failed += failed

        status = "delivered"
        if total_delivered == 0 and total_failed > 0:
            status = "failed"
        elif total_delivered > 0 and total_failed > 0:
            status = "partial"
        elif total_delivered == 0:
            status = "skipped"

        return {
            "status": status,
            "delivered": total_delivered,
            "failed": total_failed,
            "items_in_batch": len(items),
            "subscriptions_processed": len(subscriptions),
        }


async def _assess_and_deliver_for_subscription(
    session,
    subscription: Subscription,
    all_items: list[NewsItem],
) -> tuple[int, int]:
    """Run batch assessment for one subscription and deliver relevant notifications.

    Returns (delivered_count, failed_count).
    """
    sub_source_result = await session.execute(
        select(SubscriptionSource.source_id).where(
            SubscriptionSource.subscription_id == subscription.id
        )
    )
    sub_source_ids = {row[0] for row in sub_source_result.all()}

    matching_items = [item for item in all_items if item.source_id in sub_source_ids]
    if not matching_items:
        return 0, 0

    sent_result = await session.execute(
        select(SentItem.news_item_id).where(
            SentItem.subscription_id == subscription.id,
            SentItem.news_item_id.in_([item.id for item in matching_items]),
        )
    )
    already_sent = set(sent_result.scalars().all())
    pending_items = [item for item in matching_items if item.id not in already_sent]
    if not pending_items:
        return 0, 0

    history = await load_recent_notification_history(session, subscription.id)
    history_strings = [
        f"Title: {entry.title}\nSummary: {entry.summary}\n"
        f"Source: {entry.source}\nShown at: {entry.sent_at.isoformat()}"
        for entry in history
    ]

    user_spec = subscription.user_spec

    items_for_llm = [
        {
            "item_id": str(item.id),
            "headline": item.headline,
            "body": item.body or "",
            "url": item.url,
            "published_at": item.published_at.isoformat() if item.published_at else "unknown",
        }
        for item in pending_items
    ]

    try:
        batch_result = await assess_batch_events(
            items=items_for_llm,
            user_spec=user_spec,
            target_language=subscription.digest_language,
            recent_notification_history=history_strings,
            max_history_chars=settings.llm_max_context_chars,
        )
    except Exception:
        logger.exception(
            "Batch assessment failed for subscription %s",
            subscription.id,
            extra={"subscription_id": str(subscription.id)},
        )
        return 0, 1

    relevant_batch = BatchAssessmentResult(
        assessments=[a for a in batch_result.assessments if a.is_relevant]
    )
    if not relevant_batch.assessments:
        return 0, 0

    relevant_batch, dropped_item_ids = await _judge_and_revise(
        assessment=relevant_batch,
        items_for_llm=items_for_llm,
        user_spec=user_spec,
        target_language=subscription.digest_language,
        history_strings=history_strings,
        subscription_id=subscription.id,
    )

    items_by_id = {str(item.id): item for item in pending_items}
    delivered = 0
    failed = 0

    for assessment in relevant_batch.assessments:
        if assessment.item_id in dropped_item_ids:
            logger.info(
                "Dropping item %s from delivery after %d failed revisions",
                assessment.item_id,
                settings.event_judge_max_revisions,
            )
            continue

        item = items_by_id.get(assessment.item_id)
        if item is None:
            logger.warning("Assessment referenced unknown item_id %s", assessment.item_id)
            continue

        if not assessment.notification_body:
            logger.warning("Relevant item %s has empty notification body", assessment.item_id)
            continue

        validated_body = validate_notification_body(
            assessment.notification_body,
            is_relevant=assessment.is_relevant,
        )
        if validated_body is None:
            continue

        try:
            webhook_url = subscription.delivery_webhook_url
            if webhook_url is None and subscription.user is not None:
                webhook_url = subscription.user.delivery_webhook_url
            await deliver(webhook_url, "", validated_body)
            session.add(SentItem(subscription_id=subscription.id, news_item_id=item.id))
            await session.commit()
            delivered += 1
        except Exception:
            logger.exception(
                "Failed to deliver event for subscription %s, item %s",
                subscription.id,
                item.id,
            )
            failed += 1

    return delivered, failed


async def _judge_and_revise(
    *,
    assessment: BatchAssessmentResult,
    items_for_llm: list[dict],
    user_spec: str,
    target_language: str,
    history_strings: list[str],
    subscription_id: uuid.UUID,
) -> tuple[BatchAssessmentResult, set[str]]:
    """Run the judge critic loop over the assessor's output.

    Expects only relevant items. Runs up to ``event_judge_max_revisions``
    revision turns; on each turn, re-invokes the assessor only for items
    the judge flagged REVISE and replaces the current relevant-only batch
    with the revised one.

    Returns the final assessment and the set of item_ids that remain REVISE
    after the loop and must be dropped from delivery. Judge exceptions are
    swallowed (tier-2 fail-open per CLAUDE.md) and the unreviewed assessment
    is returned with no dropped items.
    """
    if not assessment.assessments:
        return assessment, set()

    items_by_id = {item["item_id"]: item for item in items_for_llm}
    current = assessment

    for revision in range(settings.event_judge_max_revisions + 1):
        try:
            verdict = await judge_batch_events(
                assessment=current,
                user_spec=user_spec,
                recent_notification_history=history_strings,
                max_history_chars=settings.llm_max_context_chars,
            )
        except Exception:
            logger.exception(
                "Batch judge failed for subscription %s revision=%d; "
                "falling through with unreviewed assessor output",
                subscription_id,
                revision,
                extra={"subscription_id": str(subscription_id)},
            )
            return current, set()

        if verdict.overall == "PASS":
            return current, set()

        revise_ids = {v.item_id for v in verdict.per_item if v.verdict == "REVISE"}
        if revision == settings.event_judge_max_revisions:
            return current, revise_ids

        feedback = {v.item_id: v.feedback for v in verdict.per_item if v.verdict == "REVISE"}
        revise_items = [items_by_id[iid] for iid in revise_ids if iid in items_by_id]
        if not revise_items:
            return current, revise_ids

        try:
            revised = await assess_batch_events(
                items=revise_items,
                user_spec=user_spec,
                target_language=target_language,
                recent_notification_history=history_strings,
                max_history_chars=settings.llm_max_context_chars,
                critic_feedback_per_item=feedback,
            )
        except Exception:
            logger.exception(
                "Revision assessor call failed for subscription %s revision=%d",
                subscription_id,
                revision,
                extra={"subscription_id": str(subscription_id)},
            )
            return current, revise_ids

        kept = [a for a in current.assessments if a.item_id not in revise_ids]
        current = BatchAssessmentResult(assessments=kept + revised.assessments)

    return current, set()
