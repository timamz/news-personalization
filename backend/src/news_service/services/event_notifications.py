import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.event import judge_event_match, judge_notification_duplicate
from news_service.models.news_item import NewsItem
from news_service.models.sent_item import SentItem
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource

logger = logging.getLogger(__name__)

_RECENT_MATCH_CONCURRENCY = 3
_NOTIFICATION_HISTORY_LOOKBACK_DAYS = 30

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


@dataclass(slots=True)
class RecentNotificationEntry:
    sent_at: datetime
    source: str
    title: str
    summary: str
    starts_at: datetime | None


async def subscription_matches_event(subscription: Subscription, item: NewsItem) -> bool:
    if subscription.event_matching_mode != "strict_with_prefilter":
        return True

    decision = await judge_event_match(
        headline=item.headline,
        body=item.body,
        published_at=item.published_at,
        raw_prompt=subscription.raw_prompt,
        event_title=item.event_title,
        event_summary=item.event_summary,
        event_starts_at=item.event_starts_at,
    )
    if not decision.matches:
        logger.info(
            "Event %s did not match strict prompt for subscription %s: %s",
            item.id,
            subscription.id,
            decision.reason,
            extra={"subscription_id": str(subscription.id), "news_item_id": str(item.id)},
        )
    return decision.matches


async def load_recent_notification_history(
    session: AsyncSession,
    subscription_id: uuid.UUID,
    *,
    lookback_days: int = _NOTIFICATION_HISTORY_LOOKBACK_DAYS,
) -> list[RecentNotificationEntry]:
    cutoff = datetime.now(UTC) - timedelta(days=lookback_days)
    result = await session.execute(
        select(SentItem.sent_at, NewsItem)
        .join(NewsItem, NewsItem.id == SentItem.news_item_id)
        .where(
            SentItem.subscription_id == subscription_id,
            SentItem.sent_at >= cutoff,
            NewsItem.event_title.is_not(None),
        )
        .order_by(SentItem.sent_at.desc())
    )
    return [
        notification_history_entry_from_item(item, sent_at=sent_at)
        for sent_at, item in result.all()
    ]


def notification_history_entry_from_item(
    item: NewsItem,
    *,
    sent_at: datetime,
) -> RecentNotificationEntry:
    return RecentNotificationEntry(
        sent_at=sent_at,
        source=item.source,
        title=item.event_title or item.headline,
        summary=item.event_summary or item.headline,
        starts_at=item.event_starts_at,
    )


async def notification_was_already_shown(
    item: NewsItem,
    history: list[RecentNotificationEntry],
) -> bool:
    if not history:
        return False

    decision = await judge_notification_duplicate(
        headline=item.headline,
        body=item.body,
        published_at=item.published_at,
        recent_notifications=[_format_history_entry(entry) for entry in history],
        event_title=item.event_title,
        event_summary=item.event_summary,
        event_starts_at=item.event_starts_at,
    )
    if decision.already_notified:
        logger.info(
            "Event %s treated as already notified: %s",
            item.id,
            decision.reason,
            extra={"news_item_id": str(item.id)},
        )
    return decision.already_notified


async def list_recent_subscription_events(
    session: AsyncSession,
    subscription: Subscription,
    *,
    lookback_days: int = 7,
    now: datetime | None = None,
) -> list[NewsItem]:
    reference_now = now or datetime.now(UTC)
    cutoff = reference_now - timedelta(days=lookback_days)
    recent_marker = func.coalesce(NewsItem.published_at, NewsItem.fetched_at)

    result = await session.execute(
        select(NewsItem)
        .join(SubscriptionSource, SubscriptionSource.feed_id == NewsItem.feed_id)
        .where(
            SubscriptionSource.subscription_id == subscription.id,
            NewsItem.event_title.is_not(None),
            recent_marker >= cutoff,
        )
        .order_by(recent_marker.desc(), NewsItem.fetched_at.desc())
    )
    items = list(result.scalars().all())

    if not items:
        return []

    if subscription.event_matching_mode == "strict_with_prefilter":
        semaphore = asyncio.Semaphore(_RECENT_MATCH_CONCURRENCY)

        async def _match_candidate(item: NewsItem) -> NewsItem | None:
            async with semaphore:
                try:
                    matches = await subscription_matches_event(subscription, item)
                except Exception:
                    logger.exception(
                        "Failed to evaluate recent event for subscription %s",
                        extra={
                            "subscription_id": str(subscription.id),
                            "news_item_id": str(item.id),
                        },
                    )
                    return None
                return item if matches else None

        matched = await asyncio.gather(*(_match_candidate(item) for item in items))
        items = [item for item in matched if item is not None]
        if not items:
            return []

    history = await load_recent_notification_history(session, subscription.id)
    return await _filter_new_notifications(items, history, sent_at=reference_now)


def build_event_notification(digest_language: str, item: NewsItem) -> tuple[str, str]:
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


async def _filter_new_notifications(
    items: list[NewsItem],
    history: list[RecentNotificationEntry],
    *,
    sent_at: datetime,
) -> list[NewsItem]:
    rolling_history = list(history)
    filtered: list[NewsItem] = []
    for item in items:
        try:
            already_notified = await notification_was_already_shown(item, rolling_history)
        except Exception:
            logger.exception(
                "Failed to evaluate duplicate notification status for event %s",
                item.id,
                extra={"news_item_id": str(item.id)},
            )
            already_notified = False

        if already_notified:
            continue

        filtered.append(item)
        rolling_history.insert(0, notification_history_entry_from_item(item, sent_at=sent_at))

    return filtered


def _format_history_entry(entry: RecentNotificationEntry) -> str:
    lines = [
        f"Shown at: {entry.sent_at.isoformat()}",
        f"Event: {entry.title}",
    ]
    if entry.starts_at is not None:
        lines.append(f"When: {entry.starts_at.isoformat()}")
    lines.append(f"Source: {entry.source}")
    lines.append(f"Summary: {entry.summary}")
    return "\n".join(lines)


def _labels_for(digest_language: str) -> dict[str, str]:
    normalized = digest_language.lower().split("-", maxsplit=1)[0]
    return _LABELS.get(normalized, _LABELS["en"])


def _format_event_time(value: datetime) -> str:
    value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return value.strftime("%Y-%m-%d %H:%M UTC")
