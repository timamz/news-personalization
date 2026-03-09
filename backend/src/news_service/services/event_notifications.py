import asyncio
import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.event import (
    judge_event_match,
    judge_notification_duplicate,
    localize_event_text,
    render_recent_events_preview,
)
from news_service.core.config import get_settings
from news_service.models.news_item import NewsItem
from news_service.models.sent_item import SentItem
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource

logger = logging.getLogger(__name__)
settings = get_settings()

_RECENT_MATCH_CONCURRENCY = settings.recent_event_match_concurrency
_NOTIFICATION_HISTORY_LOOKBACK_DAYS = 30
_DETERMINISTIC_DUPLICATE_STARTS_AT_WINDOW = timedelta(hours=6)
_DETERMINISTIC_DUPLICATE_TOKEN_OVERLAP = 0.4
_DETERMINISTIC_DUPLICATE_TEXT_SIMILARITY = 0.82
_TITLE_EQUIVALENCE_TOKEN_OVERLAP = 0.85
_TITLE_EQUIVALENCE_TEXT_SIMILARITY = 0.96
_NORMALIZED_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

_LABELS = {
    "en": {
        "subject": "Upcoming event",
        "event": "Event",
        "when": "When",
        "source": "Source",
    },
    "ru": {
        "subject": "Предстоящее событие",
        "event": "Событие",
        "when": "Когда",
        "source": "Источник",
    },
}


@dataclass(slots=True)
class RecentNotificationEntry:
    sent_at: datetime
    source: str
    title: str
    summary: str
    starts_at: datetime | None


@dataclass(slots=True)
class RecentEventsPreview:
    news_item_ids: list[uuid.UUID]
    subject: str
    body: str


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

    duplicate_entry = _deterministic_duplicate_entry(item, history)
    if duplicate_entry is not None:
        logger.info(
            "Event %s treated as already notified by deterministic duplicate match",
            item.id,
            extra={"news_item_id": str(item.id)},
        )
        return True

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
    items = await load_recent_event_candidates(
        session,
        subscription.id,
        lookback_days=lookback_days,
        now=now,
    )
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
    reference_now = now or datetime.now(UTC)
    return await _filter_new_notifications(items, history, sent_at=reference_now)


async def load_recent_event_candidates(
    session: AsyncSession,
    subscription_id: uuid.UUID,
    *,
    lookback_days: int,
    now: datetime | None = None,
) -> list[NewsItem]:
    reference_now = now or datetime.now(UTC)
    cutoff = reference_now - timedelta(days=lookback_days)
    recent_marker = func.coalesce(NewsItem.published_at, NewsItem.fetched_at)

    result = await session.execute(
        select(NewsItem)
        .join(SubscriptionSource, SubscriptionSource.feed_id == NewsItem.feed_id)
        .where(
            SubscriptionSource.subscription_id == subscription_id,
            NewsItem.event_title.is_not(None),
            recent_marker >= cutoff,
        )
        .order_by(recent_marker.desc(), NewsItem.fetched_at.desc())
    )
    items = list(result.scalars().all())

    if not items:
        return []
    return items


async def build_event_notification(digest_language: str, item: NewsItem) -> tuple[str, str]:
    labels = _labels_for(digest_language)
    title = item.event_title or item.headline
    summary = item.event_summary or item.headline
    try:
        localized = await localize_event_text(
            headline=item.headline,
            body=item.body,
            event_title=item.event_title,
            event_summary=item.event_summary,
            event_starts_at=item.event_starts_at,
            target_language=digest_language,
        )
    except Exception:
        logger.exception(
            "Failed to localize event notification for news item %s",
            item.id,
            extra={"news_item_id": str(item.id)},
        )
    else:
        title = localized.title
        summary = localized.summary

    subject = f"{labels['subject']}: {title}"
    lines = [f"{labels['event']}: {title}"]
    if item.event_starts_at is not None:
        lines.append(f"{labels['when']}: {_format_event_time(item.event_starts_at)}")

    if summary:
        lines.extend(["", summary])

    lines.extend(["", f"{labels['source']}: {item.source}", item.url])
    return subject, "\n".join(lines)


async def build_recent_events_preview(
    digest_language: str,
    items: list[NewsItem],
    *,
    lookback_days: int,
) -> RecentEventsPreview:
    news_item_ids = [item.id for item in items]
    subject, body = _fallback_recent_events_preview(digest_language, items, lookback_days)
    return RecentEventsPreview(
        news_item_ids=news_item_ids,
        subject=subject,
        body=body,
    )


async def build_recent_events_preview_for_subscription(
    session: AsyncSession,
    subscription: Subscription,
    *,
    lookback_days: int = 7,
    now: datetime | None = None,
) -> RecentEventsPreview | None:
    items = await load_recent_event_candidates(
        session,
        subscription.id,
        lookback_days=lookback_days,
        now=now,
    )
    if not items:
        return None

    deduplicated_items = _deduplicate_preview_candidates(items)
    history = await load_recent_notification_history(session, subscription.id)
    try:
        decision = await render_recent_events_preview(
            raw_prompt=subscription.raw_prompt,
            target_language=subscription.digest_language,
            event_matching_mode=subscription.event_matching_mode,
            lookback_days=lookback_days,
            candidate_events=[_format_preview_candidate(item) for item in deduplicated_items],
            recent_notifications=[_format_history_entry(entry) for entry in history],
        )
    except Exception:
        logger.exception(
            "Failed to select recent events preview for subscription %s",
            subscription.id,
            extra={"subscription_id": str(subscription.id)},
        )
        fallback_items = await list_recent_subscription_events(
            session,
            subscription,
            lookback_days=lookback_days,
            now=now,
        )
        if not fallback_items:
            return None
        return await build_recent_events_preview(
            subscription.digest_language,
            fallback_items,
            lookback_days=lookback_days,
        )

    if not decision.selected_item_ids:
        return None

    items_by_id = {str(item.id): item for item in deduplicated_items}
    selected_items = [
        items_by_id[item_id] for item_id in decision.selected_item_ids if item_id in items_by_id
    ]
    if not selected_items:
        return None

    subject = decision.subject.strip()
    body = decision.body.strip()
    if not subject or not body:
        return await build_recent_events_preview(
            subscription.digest_language,
            selected_items,
            lookback_days=lookback_days,
        )

    return RecentEventsPreview(
        news_item_ids=[item.id for item in selected_items],
        subject=subject,
        body=body,
    )


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


def _format_preview_candidate(item: NewsItem) -> str:
    lines = [f"ID: {item.id}", f"Title: {item.event_title or item.headline}"]
    if item.event_starts_at is not None:
        lines.append(f"When: {_format_event_time(item.event_starts_at)}")
    if item.event_summary:
        lines.append(f"Summary: {item.event_summary}")
    lines.append(f"Source: {item.source}")
    lines.append(f"URL: {item.url}")
    return "\n".join(lines)


def _fallback_recent_events_preview(
    digest_language: str,
    items: list[NewsItem],
    lookback_days: int,
) -> tuple[str, str]:
    if digest_language.strip().lower().split("-", maxsplit=1)[0] == "ru":
        subject = "Что вы могли пропустить"
        intro = f"Вот релевантные события за последние {lookback_days} дней:"
    else:
        subject = "Recent events you may have missed"
        intro = f"Here are the relevant events from the last {lookback_days} days:"

    bullets = []
    for item in items:
        title = item.event_title or item.headline
        when = (
            _format_event_time(item.event_starts_at) if item.event_starts_at is not None else None
        )
        summary = item.event_summary or item.headline
        bullet_parts = [title]
        if when is not None:
            bullet_parts.append(when)
        bullet_parts.append(summary)
        bullet_parts.append(item.source)
        bullets.append(f"- {' | '.join(bullet_parts)}\n{item.url}")
    return subject, f"{intro}\n\n" + "\n\n".join(bullets)


def _deduplicate_preview_candidates(items: list[NewsItem]) -> list[NewsItem]:
    deduplicated: list[NewsItem] = []
    preview_history: list[RecentNotificationEntry] = []
    for item in items:
        if _deterministic_duplicate_entry(item, preview_history) is not None:
            continue
        deduplicated.append(item)
        preview_history.insert(
            0,
            notification_history_entry_from_item(
                item,
                sent_at=item.published_at or item.fetched_at,
            ),
        )
    return deduplicated


def _deterministic_duplicate_entry(
    item: NewsItem,
    history: list[RecentNotificationEntry],
) -> RecentNotificationEntry | None:
    for entry in history:
        if _events_are_equivalent(item, entry):
            return entry
    return None


def _events_are_equivalent(item: NewsItem, entry: RecentNotificationEntry) -> bool:
    if _titles_are_equivalent(item, entry):
        return True
    return _same_occurrence_by_time_and_text(item, entry)


def _titles_are_equivalent(item: NewsItem, entry: RecentNotificationEntry) -> bool:
    candidate_title = _normalize_event_text(item.event_title or item.headline)
    history_title = _normalize_event_text(entry.title)
    if not candidate_title or not history_title:
        return False
    if candidate_title == history_title:
        return True
    return (
        _token_overlap(candidate_title, history_title) >= _TITLE_EQUIVALENCE_TOKEN_OVERLAP
        or _text_similarity(candidate_title, history_title) >= _TITLE_EQUIVALENCE_TEXT_SIMILARITY
    )


def _same_occurrence_by_time_and_text(item: NewsItem, entry: RecentNotificationEntry) -> bool:
    if item.event_starts_at is None or entry.starts_at is None:
        return False
    if not _starts_at_close(item.event_starts_at, entry.starts_at):
        return False

    candidate_text = _normalize_event_text(item.event_title, item.event_summary, item.headline)
    history_text = _normalize_event_text(entry.title, entry.summary)
    if not candidate_text or not history_text:
        return False

    return (
        _token_overlap(candidate_text, history_text) >= _DETERMINISTIC_DUPLICATE_TOKEN_OVERLAP
        or _text_similarity(candidate_text, history_text)
        >= _DETERMINISTIC_DUPLICATE_TEXT_SIMILARITY
    )


def _normalize_event_text(*parts: str | None) -> str:
    values = [part for part in parts if part]
    if not values:
        return ""
    normalized = " ".join(values).casefold().replace("ё", "е")
    tokens = _NORMALIZED_TOKEN_PATTERN.findall(normalized)
    return " ".join(tokens)


def _token_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in left.split() if len(token) >= 4}
    right_tokens = {token for token in right.split() if len(token) >= 4}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(a=left, b=right).ratio()


def _starts_at_close(left: datetime, right: datetime) -> bool:
    left = left.replace(tzinfo=UTC) if left.tzinfo is None else left.astimezone(UTC)
    right = right.replace(tzinfo=UTC) if right.tzinfo is None else right.astimezone(UTC)
    return abs(left - right) <= _DETERMINISTIC_DUPLICATE_STARTS_AT_WINDOW


def _labels_for(digest_language: str) -> dict[str, str]:
    normalized = digest_language.lower().split("-", maxsplit=1)[0]
    return _LABELS.get(normalized, _LABELS["en"])


def _format_event_time(value: datetime) -> str:
    value = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
    return value.strftime("%Y-%m-%d %H:%M UTC")
