import asyncio
import logging
import uuid
from datetime import UTC, datetime

from pydantic import ValidationError
from sqlalchemy import select

from news_service.agents.event import parse_event_constraint_values
from news_service.db.session import get_task_session
from news_service.models.news_item import NewsItem
from news_service.models.sent_item import SentItem
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.schemas.subscription import EventConstraint
from news_service.services.delivery import get_delivery_channel
from news_service.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

_LABELS = {
    "en": {
        "subject": "Upcoming event",
        "event": "Event",
        "when": "When",
        "source": "Source",
    },
    "ru": {
        "subject": "Predstoyashchee sobytie",
        "event": "Sobytiye",
        "when": "Kogda",
        "source": "Istochnik",
    },
}


@celery_app.task(name="news_service.tasks.deliver_events.deliver_event_notifications")
def deliver_event_notifications(news_item_id: str) -> dict:
    return asyncio.run(_deliver_event_notifications(uuid.UUID(news_item_id)))


async def _deliver_event_notifications(news_item_id: uuid.UUID) -> dict:
    async with get_task_session() as session:
        item = await session.get(NewsItem, news_item_id)
        if item is None:
            logger.warning("News item %s not found for event delivery", news_item_id)
            return {"status": "skipped", "reason": "news_item_not_found"}
        if not item.event_title:
            return {"status": "skipped", "reason": "not_event"}

        result = await session.execute(
            select(Subscription)
            .join(
                SubscriptionSource,
                SubscriptionSource.subscription_id == Subscription.id,
            )
            .where(
                Subscription.is_active.is_(True),
                Subscription.delivery_mode == "event",
                SubscriptionSource.feed_id == item.feed_id,
            )
        )
        subscriptions = list(result.scalars().all())
        if not subscriptions:
            return {"status": "skipped", "reason": "no_matching_subscriptions"}

        subscription_ids = [subscription.id for subscription in subscriptions]
        sent_result = await session.execute(
            select(SentItem.subscription_id).where(
                SentItem.news_item_id == item.id,
                SentItem.subscription_id.in_(subscription_ids),
            )
        )
        sent_subscription_ids = set(sent_result.scalars().all())

        delivered = 0
        failed = 0
        for subscription in subscriptions:
            if subscription.id in sent_subscription_ids:
                continue
            try:
                if not await _subscription_matches_event(subscription, item):
                    continue
            except Exception:
                logger.exception(
                    "Failed to evaluate event constraints for subscription %s",
                    subscription.id,
                    extra={
                        "subscription_id": str(subscription.id),
                        "news_item_id": str(item.id),
                    },
                )
                continue

            channel = get_delivery_channel(subscription.delivery_webhook_url)
            subject, body = _build_notification(subscription.digest_language, item)
            try:
                await channel.send(subject, body)
            except Exception:
                failed += 1
                logger.exception(
                    "Failed to deliver event notification for subscription %s",
                    subscription.id,
                    extra={
                        "subscription_id": str(subscription.id),
                        "news_item_id": str(item.id),
                    },
                )
                continue

            session.add(SentItem(subscription_id=subscription.id, news_item_id=item.id))
            await session.commit()
            delivered += 1

        if delivered == 0 and failed == 0:
            return {"status": "skipped", "reason": "already_sent"}

        status = "delivered"
        if delivered > 0 and failed > 0:
            status = "partial"
        elif delivered == 0 and failed > 0:
            status = "failed"

        return {
            "status": status,
            "delivered": delivered,
            "failed": failed,
            "news_item_id": str(news_item_id),
        }


async def _subscription_matches_event(subscription: Subscription, item: NewsItem) -> bool:
    if subscription.event_matching_mode != "strict_with_prefilter":
        return True

    constraints = _load_constraints(subscription.event_constraints)
    if not constraints:
        logger.warning(
            "Strict event subscription %s has no valid constraints",
            subscription.id,
            extra={"subscription_id": str(subscription.id)},
        )
        return False

    if not _passes_prefilter(item, constraints):
        logger.info(
            "Event %s failed cheap prefilter for subscription %s",
            item.id,
            subscription.id,
            extra={"subscription_id": str(subscription.id), "news_item_id": str(item.id)},
        )
        return False

    parsed_values = await parse_event_constraint_values(
        headline=item.headline,
        body=item.body,
        published_at=item.published_at,
        raw_prompt=subscription.raw_prompt,
        constraints=constraints,
    )
    matches = _all_constraints_match(constraints, parsed_values)
    if not matches:
        logger.info(
            "Event %s failed strict constraint match for subscription %s",
            item.id,
            subscription.id,
            extra={"subscription_id": str(subscription.id), "news_item_id": str(item.id)},
        )
    return matches


def _load_constraints(raw_constraints: list[dict[str, object]] | None) -> list[EventConstraint]:
    constraints: list[EventConstraint] = []
    for raw_constraint in raw_constraints or []:
        try:
            constraints.append(EventConstraint.model_validate(raw_constraint))
        except ValidationError:
            logger.exception("Invalid stored event constraint payload: %s", raw_constraint)
    return constraints


def _passes_prefilter(item: NewsItem, constraints: list[EventConstraint]) -> bool:
    searchable = _normalize_text(
        "\n".join(
            part
            for part in [
                item.headline,
                item.body,
                item.event_title or "",
                item.event_summary or "",
            ]
            if part
        )
    )
    prefilter_groups: list[list[str]] = []
    for constraint in constraints:
        normalized_terms = [
            _normalize_text(term)
            for term in _prefilter_terms_for_constraint(constraint)
            if term.strip()
        ]
        if normalized_terms:
            prefilter_groups.append(normalized_terms)

    if not prefilter_groups:
        return True

    return any(any(term in searchable for term in terms) for terms in prefilter_groups)


def _prefilter_terms_for_constraint(constraint: EventConstraint) -> list[str]:
    if constraint.prefilter_terms:
        return constraint.prefilter_terms
    if constraint.required_string:
        return [constraint.required_string]
    if constraint.required_list:
        return constraint.required_list
    return []


def _all_constraints_match(
    constraints: list[EventConstraint],
    parsed_values: dict[str, str | bool | list[str] | None],
) -> bool:
    return all(
        _constraint_matches(constraint, parsed_values.get(constraint.key))
        for constraint in constraints
    )


def _constraint_matches(
    constraint: EventConstraint,
    actual_value: str | bool | list[str] | None,
) -> bool:
    if actual_value is None:
        return False

    if constraint.value_type == "boolean":
        return isinstance(actual_value, bool) and actual_value == constraint.required_boolean

    if constraint.value_type == "string":
        if not isinstance(actual_value, str) or constraint.required_string is None:
            return False
        actual_text = _normalize_text(actual_value)
        required_text = _normalize_text(constraint.required_string)
        if constraint.match_mode == "exact":
            return actual_text == required_text
        if constraint.match_mode == "contains":
            return required_text in actual_text
        return False

    if not isinstance(actual_value, list):
        return False
    actual_list = [_normalize_text(str(value)) for value in actual_value]
    required_list = [_normalize_text(value) for value in constraint.required_list]
    if constraint.match_mode == "intersects":
        return any(value in actual_list for value in required_list)
    if constraint.match_mode == "exact":
        return sorted(actual_list) == sorted(required_list)
    return False


def _build_notification(digest_language: str, item: NewsItem) -> tuple[str, str]:
    labels = _labels_for(digest_language)
    subject = f"{labels['subject']}: {item.event_title}"
    lines = [f"{labels['event']}: {item.event_title}"]
    if item.event_starts_at is not None:
        lines.append(f"{labels['when']}: {_format_event_time(item.event_starts_at)}")

    summary = item.event_summary or item.headline
    if summary:
        lines.extend(["", summary])

    lines.extend(["", f"{labels['source']}: {item.source}", item.url])
    return subject, "\n".join(lines)


def _labels_for(digest_language: str) -> dict[str, str]:
    normalized = digest_language.lower().split("-", maxsplit=1)[0]
    return _LABELS.get(normalized, _LABELS["en"])


def _format_event_time(value: datetime) -> str:
    value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return value.strftime("%Y-%m-%d %H:%M UTC")


def _normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())
