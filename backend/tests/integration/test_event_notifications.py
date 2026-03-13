import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from news_service.agents.event import EventMatchDecision, NotificationDuplicateDecision
from news_service.db.session import async_session_factory
from news_service.models.news_item import NewsItem
from news_service.models.rss_feed import RssFeed
from news_service.models.sent_item import SentItem
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.models.user import User
from news_service.services import event_notifications
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
            source_description="Television events feed",
            is_active=True,
            subscriber_count=1,
        )
        session.add(user)
        session.add(feed)
        await session.flush()

        subscription = Subscription(
            user_id=user.id,
            raw_prompt="Notify me when a new episode is announced",
            prompt_summary="Severance episode announcements",
            delivery_mode="event",
            event_matching_mode="basic",
            event_constraints=[],
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


async def test_strict_event_notification_skips_non_matching_event(mocker) -> None:
    news_item_id = uuid.uuid4()
    subscription_id: uuid.UUID

    async with async_session_factory() as session:
        user = User(api_key="strict-event-test-api-key")
        feed = RssFeed(
            url="https://example.com/strict-events.xml",
            title="Strict Events Feed",
            source_description="Science events feed",
            is_active=True,
            subscriber_count=1,
        )
        session.add(user)
        session.add(feed)
        await session.flush()

        subscription = Subscription(
            user_id=user.id,
            raw_prompt="Only Stanislav Drobyshevsky's own lectures",
            prompt_summary="Drobyshevsky lectures only",
            delivery_mode="event",
            event_matching_mode="strict_with_prefilter",
            event_constraints=[],
            schedule_cron=None,
            format_instructions="brief summary",
            digest_language="en",
            delivery_webhook_url="http://frontend.example.test/deliver/1",
            is_active=True,
        )
        session.add(subscription)
        await session.flush()

        subscription_id = subscription.id
        session.add(SubscriptionSource(subscription_id=subscription.id, feed_id=feed.id))
        session.add(
            NewsItem(
                id=news_item_id,
                feed_id=feed.id,
                headline="Лекция Александра Очередного в центре Дробышевского",
                body="Приглашаем на лекцию Александра Очередного в центре популяризации науки.",
                url=f"https://example.com/strict-events/{news_item_id}",
                source="Strict Events Feed",
                published_at=datetime(2026, 3, 3, 12, 0, tzinfo=UTC),
                event_title="Лекция в центре Дробышевского",
                event_summary="Лекция Александра Очередного в центре Дробышевского.",
                event_starts_at=datetime(2026, 4, 1, 20, 0, tzinfo=UTC),
                fetched_at=datetime(2026, 3, 3, 12, 1, tzinfo=UTC),
            )
        )
        await session.commit()

    channel = AsyncMock()
    mocker.patch.object(deliver_events, "get_delivery_channel", return_value=channel)
    mocker.patch.object(
        event_notifications,
        "judge_event_match",
        new=AsyncMock(
            return_value=EventMatchDecision(
                matches=False,
                reason="The post is about another speaker.",
            )
        ),
    )

    result = await deliver_events._deliver_event_notifications(news_item_id)

    assert result == {"status": "skipped", "reason": "already_sent"}
    channel.send.assert_not_awaited()

    async with async_session_factory() as session:
        sent_result = await session.execute(
            select(SentItem).where(
                SentItem.subscription_id == subscription_id,
                SentItem.news_item_id == news_item_id,
            )
        )
        assert sent_result.scalar_one_or_none() is None


async def test_event_notification_does_not_skip_when_only_another_subscription_has_recent_history(
    mocker,
) -> None:
    current_news_item_id = uuid.uuid4()

    async with async_session_factory() as session:
        user = User(api_key="duplicate-history-test-api-key")
        old_feed = RssFeed(
            url="https://example.com/old-events.xml",
            title="Old Events Feed",
            source_description="Science events feed",
            is_active=True,
            subscriber_count=0,
        )
        current_feed = RssFeed(
            url="https://example.com/current-events.xml",
            title="Current Events Feed",
            source_description="Science events feed",
            is_active=True,
            subscriber_count=1,
        )
        session.add_all([user, old_feed, current_feed])
        await session.flush()

        old_subscription = Subscription(
            user_id=user.id,
            raw_prompt="Notify me about Drobyshevsky lectures",
            prompt_summary="Drobyshevsky lectures",
            delivery_mode="event",
            event_matching_mode="basic",
            event_constraints=[],
            schedule_cron=None,
            format_instructions="brief summary",
            digest_language="en",
            delivery_webhook_url="http://frontend.example.test/deliver/1",
            is_active=False,
        )
        current_subscription = Subscription(
            user_id=user.id,
            raw_prompt="Notify me about Drobyshevsky lectures",
            prompt_summary="Drobyshevsky lectures",
            delivery_mode="event",
            event_matching_mode="basic",
            event_constraints=[],
            schedule_cron=None,
            format_instructions="brief summary",
            digest_language="en",
            delivery_webhook_url="http://frontend.example.test/deliver/1",
            is_active=True,
        )
        session.add_all([old_subscription, current_subscription])
        await session.flush()

        old_news_item = NewsItem(
            feed_id=old_feed.id,
            headline="Lecture announced",
            body="Stanislav Drobyshevsky will lecture next week.",
            url="https://example.com/old-events/1",
            source="Old Events Feed",
            published_at=datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
            event_title="Stanislav Drobyshevsky lecture",
            event_summary="Stanislav Drobyshevsky will lecture next week.",
            event_starts_at=datetime(2026, 3, 10, 19, 0, tzinfo=UTC),
            fetched_at=datetime(2026, 3, 1, 12, 1, tzinfo=UTC),
        )
        current_news_item = NewsItem(
            id=current_news_item_id,
            feed_id=current_feed.id,
            headline="Reminder about the lecture",
            body="Reminder: Stanislav Drobyshevsky will lecture next week.",
            url="https://example.com/current-events/1",
            source="Current Events Feed",
            published_at=datetime(2026, 3, 3, 12, 0, tzinfo=UTC),
            event_title="Stanislav Drobyshevsky lecture reminder",
            event_summary="Reminder: Stanislav Drobyshevsky will lecture next week.",
            event_starts_at=datetime(2026, 3, 10, 19, 0, tzinfo=UTC),
            fetched_at=datetime(2026, 3, 3, 12, 1, tzinfo=UTC),
        )
        session.add_all([old_news_item, current_news_item])
        session.add(
            SubscriptionSource(
                subscription_id=current_subscription.id,
                feed_id=current_feed.id,
            )
        )
        await session.flush()
        session.add(SentItem(subscription_id=old_subscription.id, news_item_id=old_news_item.id))
        await session.commit()

    channel = AsyncMock()
    mocker.patch.object(deliver_events, "get_delivery_channel", return_value=channel)
    duplicate_judge = mocker.patch.object(
        event_notifications,
        "judge_notification_duplicate",
        new=AsyncMock(
            return_value=NotificationDuplicateDecision(
                already_notified=True,
                reason="The user already got the same event from another subscription.",
            )
        ),
    )

    result = await deliver_events._deliver_event_notifications(current_news_item_id)

    assert result == {
        "status": "delivered",
        "delivered": 1,
        "failed": 0,
        "news_item_id": str(current_news_item_id),
    }
    duplicate_judge.assert_not_awaited()
    channel.send.assert_awaited_once()
