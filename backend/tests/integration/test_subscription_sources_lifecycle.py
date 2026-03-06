import uuid
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from news_service.db.session import async_session_factory
from news_service.models.rss_feed import RssFeed
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.schemas.subscription import SubscriptionConfig

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_deactivate_subscription_removes_fixed_source_links(
    api_client: AsyncClient,
    mocker,
) -> None:
    parsed_config = SubscriptionConfig(
        topics=["ai"],
        schedule_cron="0 8 * * *",
        schedule_was_explicit=True,
        format_instructions="brief summary",
        digest_language="en",
    )
    mocker.patch(
        "news_service.api.routes_subscriptions.parse_subscription",
        new=AsyncMock(return_value=parsed_config),
    )

    async def fake_ensure_topic_coverage(session, topics, topics_embedding):  # noqa: ANN001
        assert topics_embedding == [2.0] * 1536
        feed = RssFeed(
            url="https://example.com/rss.xml",
            title="Example Feed",
            topic_tags=topics,
            topic_embedding=[0.0] * 1536,
            is_active=True,
            subscriber_count=1,
        )
        session.add(feed)
        await session.flush()
        return [feed]

    mocker.patch(
        "news_service.api.routes_subscriptions.ensure_topic_coverage",
        new=fake_ensure_topic_coverage,
    )

    user_response = await api_client.post("/users")
    assert user_response.status_code == 201
    api_key = user_response.json()["api_key"]

    create_response = await api_client.post(
        "/subscriptions",
        headers={"X-API-Key": api_key},
        json={
            "prompt": "AI updates every morning",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
        },
    )
    assert create_response.status_code == 201
    subscription_id = uuid.UUID(create_response.json()["id"])

    async with async_session_factory() as session:
        link_result = await session.execute(
            select(SubscriptionSource).where(SubscriptionSource.subscription_id == subscription_id)
        )
        links = list(link_result.scalars().all())
        assert len(links) == 1

        feed = await session.get(RssFeed, links[0].feed_id)
        assert feed is not None
        assert feed.subscriber_count == 1
        assert feed.is_active is True

    delete_response = await api_client.delete(
        f"/subscriptions/{subscription_id}",
        headers={"X-API-Key": api_key},
    )
    assert delete_response.status_code == 204

    async with async_session_factory() as session:
        subscription = await session.get(Subscription, subscription_id)
        assert subscription is not None
        assert subscription.is_active is False

        link_result = await session.execute(
            select(SubscriptionSource).where(SubscriptionSource.subscription_id == subscription_id)
        )
        assert link_result.scalars().all() == []

        feed_result = await session.execute(
            select(RssFeed).where(RssFeed.url == "https://example.com/rss.xml")
        )
        feed = feed_result.scalar_one_or_none()
        assert feed is not None
        assert feed.subscriber_count == 0
        assert feed.is_active is False


async def test_create_event_subscription_forces_schedule_off(
    api_client: AsyncClient,
    mocker,
) -> None:
    parsed_config = SubscriptionConfig(
        topics=["tv"],
        delivery_mode="event",
        event_matching_mode="strict_with_prefilter",
        schedule_cron="0 8 * * *",
        schedule_was_explicit=True,
        format_instructions="brief summary",
        digest_language="en",
    )
    mocker.patch(
        "news_service.api.routes_subscriptions.parse_subscription",
        new=AsyncMock(return_value=parsed_config),
    )

    async def fake_ensure_topic_coverage(session, topics, topics_embedding):  # noqa: ANN001
        assert topics_embedding == [2.0] * 1536
        feed = RssFeed(
            url="https://example.com/shows.xml",
            title="Shows Feed",
            topic_tags=topics,
            topic_embedding=[0.0] * 1536,
            is_active=True,
            subscriber_count=1,
        )
        session.add(feed)
        await session.flush()
        return [feed]

    mocker.patch(
        "news_service.api.routes_subscriptions.ensure_topic_coverage",
        new=fake_ensure_topic_coverage,
    )

    user_response = await api_client.post("/users")
    assert user_response.status_code == 201
    api_key = user_response.json()["api_key"]

    create_response = await api_client.post(
        "/subscriptions",
        headers={"X-API-Key": api_key},
        json={
            "prompt": "Notify me when the next episode is announced",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
            "delivery_mode": "event",
        },
    )

    assert create_response.status_code == 201
    assert create_response.json()["delivery_mode"] == "event"
    assert create_response.json()["schedule_cron"] is None

    subscription_id = uuid.UUID(create_response.json()["id"])
    async with async_session_factory() as session:
        subscription = await session.get(Subscription, subscription_id)
        assert subscription is not None
        assert subscription.delivery_mode == "event"
        assert subscription.event_matching_mode == "strict_with_prefilter"
        assert subscription.event_constraints == []
        assert subscription.schedule_cron is None
