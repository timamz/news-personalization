import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.models.news_item import NewsItem
from news_service.models.sent_item import SentItem

_NOTIFICATION_HISTORY_LOOKBACK_DAYS = 30


@dataclass(slots=True)
class RecentNotificationEntry:
    sent_at: datetime
    source: str
    title: str
    summary: str


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
