import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from news_service.db.session import async_session_factory
from news_service.models.news_item import NewsItem
from news_service.models.rss_feed import RssFeed
from news_service.models.sent_item import SentItem
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.models.user import User
from news_service.tasks import deliver_events

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_event_notification_delivery_marks_item_as_sent(mocker) -> None:
    news_item_id = uuid.uuid4()
    subscription_id: uuid.UUID

    async with async_session_factory() as session:
        user = User(api_key="event-test-api-key")
        feed = RssFeed(
            url="https://example.com/events.xml",
            title="Events Feed",
            topic_tags=["television"],
            is_active=True,
            subscriber_count=1,
        )
        session.add(user)
        session.add(feed)
        await session.flush()

        subscription = Subscription(
            user_id=user.id,
            raw_prompt="Notify me when a new episode is announced",
            topics=["Severance"],
            delivery_mode="event",
            schedule_cron=None,
            format_instructions="brief summary",
            digest_language="en",
            delivery_webhook_url="http://frontend.example.test/deliver/1",
            is_active=True,
        )
        session.add(subscription)
        await session.flush()

        subscription_id = subscription.id
        session.add(
            SubscriptionSource(
                subscription_id=subscription.id,
                feed_id=feed.id,
            )
        )
        session.add(
            NewsItem(
                id=news_item_id,
                feed_id=feed.id,
                headline="Severance season finale release announced",
                body="Apple confirmed the finale release date for next month.",
                url=f"https://example.com/events/{news_item_id}",
                source="Events Feed",
                published_at=datetime(2026, 3, 3, 12, 0, tzinfo=UTC),
                event_title="Severance season finale",
                event_summary="Apple confirmed the new episode release date.",
                event_starts_at=datetime(2026, 4, 1, 20, 0, tzinfo=UTC),
                fetched_at=datetime(2026, 3, 3, 12, 1, tzinfo=UTC),
            )
        )
        await session.commit()

    channel = AsyncMock()
    mocker.patch.object(deliver_events, "get_delivery_channel", return_value=channel)

    result = await deliver_events._deliver_event_notifications(news_item_id)

    assert result == {
        "status": "delivered",
        "delivered": 1,
        "failed": 0,
        "news_item_id": str(news_item_id),
    }
    channel.send.assert_awaited_once()
    assert channel.send.await_args.args[0] == "Upcoming event: Severance season finale"
    assert "When: 2026-04-01 20:00 UTC" in channel.send.await_args.args[1]
    assert "Apple confirmed the new episode release date." in channel.send.await_args.args[1]

    async with async_session_factory() as session:
        sent_result = await session.execute(
            select(SentItem).where(
                SentItem.subscription_id == subscription_id,
                SentItem.news_item_id == news_item_id,
            )
        )
        sent_record = sent_result.scalar_one_or_none()
        assert sent_record is not None
