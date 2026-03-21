import logging
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from difflib import SequenceMatcher

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.event import render_recent_events_preview
from news_service.core.config import get_settings
from news_service.models.news_item import NewsItem
from news_service.models.sent_item import SentItem
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource

logger = logging.getLogger(__name__)
settings = get_settings()

_NOTIFICATION_HISTORY_LOOKBACK_DAYS = 30
_NORMALIZED_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


@dataclass(slots=True)
class RecentNotificationEntry:
    sent_at: datetime
    source: str
    title: str
    summary: str


@dataclass(slots=True)
class RecentEventsPreview:
    news_item_ids: list[uuid.UUID]
    subject: str
    body: str


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
        title=item.headline,
        summary=item.body,
    )


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
        .join(SubscriptionSource, SubscriptionSource.source_id == NewsItem.source_id)
        .where(
            SubscriptionSource.subscription_id == subscription_id,
            recent_marker >= cutoff,
        )
        .order_by(recent_marker.desc(), NewsItem.fetched_at.desc())
        .limit(200)
    )
    return list(result.scalars().all())


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

    history = await load_recent_notification_history(session, subscription.id)
    try:
        decision = await render_recent_events_preview(
            raw_prompt=subscription.canonical_prompt,
            target_language=subscription.digest_language,
            lookback_days=lookback_days,
            candidate_events=[_format_preview_candidate(item) for item in items],
            recent_notifications=[_format_history_entry(entry) for entry in history],
        )
    except Exception:
        logger.exception(
            "Failed to select recent events preview for subscription %s",
            subscription.id,
            extra={"subscription_id": str(subscription.id)},
        )
        return None

    if not decision.selected_item_ids:
        return None

    items_by_id = {str(item.id): item for item in items}
    selected_items = [
        items_by_id[item_id] for item_id in decision.selected_item_ids if item_id in items_by_id
    ]
    if not selected_items:
        return None

    subject = decision.subject.strip()
    body = decision.body.strip()
    if not subject or not body:
        return None

    return RecentEventsPreview(
        news_item_ids=[item.id for item in selected_items],
        subject=subject,
        body=body,
    )


def _format_history_entry(entry: RecentNotificationEntry) -> str:
    lines = [
        f"Shown at: {entry.sent_at.isoformat()}",
        f"Event: {entry.title}",
        f"Source: {entry.source}",
        f"Summary: {entry.summary}",
    ]
    return "\n".join(lines)


def _format_preview_candidate(item: NewsItem) -> str:
    lines = [
        f"ID: {item.id}",
        f"Title: {item.headline}",
    ]
    body_preview = item.body or ""
    if body_preview:
        lines.append(f"Summary: {body_preview}")
    lines.append(f"URL: {item.url}")
    return "\n".join(lines)


def normalize_event_text(*parts: str | None) -> str:
    values = [part for part in parts if part]
    if not values:
        return ""
    normalized = " ".join(values).casefold().replace("\u0451", "\u0435")
    tokens = _NORMALIZED_TOKEN_PATTERN.findall(normalized)
    return " ".join(tokens)


def token_overlap(left: str, right: str) -> float:
    left_tokens = {token for token in left.split() if len(token) >= 4}
    right_tokens = {token for token in right.split() if len(token) >= 4}
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def text_similarity(left: str, right: str) -> float:
    return SequenceMatcher(a=left, b=right).ratio()
