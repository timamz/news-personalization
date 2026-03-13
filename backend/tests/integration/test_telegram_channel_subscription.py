import uuid
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from news_service.db.session import async_session_factory
from news_service.models.rss_feed import RssFeed
from news_service.models.subscription_source import SubscriptionSource
from news_service.schemas.subscription import SubscriptionConfig
from tests.integration.helpers import create_user

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_subscription_with_telegram_channel_registers_source(
    api_client: AsyncClient,
    mocker,
) -> None:
    parsed_config = SubscriptionConfig(
        topics=["science"],
        schedule_cron="0 8 * * *",
        schedule_was_explicit=True,
        format_instructions="brief summary",
        digest_language="en",
    )
    mocker.patch(
        "news_service.api.routes_subscriptions.parse_subscription",
        new=AsyncMock(return_value=parsed_config),
    )
    ensure_topic_coverage = AsyncMock()
    mocker.patch(
        "news_service.api.routes_subscriptions.ensure_topic_coverage",
        new=ensure_topic_coverage,
    )

    user = await create_user(api_client, timezone="UTC")
    api_key = user["api_key"]

    create_response = await api_client.post(
        "/subscriptions",
        headers={"X-API-Key": api_key},
        json={
            "prompt": "Track @fondnauk every morning",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
            "fixed_telegram_channels": ["fondnauk"],
            "include_discovered_sources": False,
        },
    )
    assert create_response.status_code == 201
    subscription_id = uuid.UUID(create_response.json()["id"])

    async with async_session_factory() as session:
        result = await session.execute(
            select(RssFeed).where(RssFeed.url == "https://t.me/s/fondnauk")
        )
        feed = result.scalar_one_or_none()
        source_link_result = await session.execute(
            select(SubscriptionSource).where(
                SubscriptionSource.subscription_id == subscription_id,
                SubscriptionSource.feed_id == feed.id if feed is not None else None,
            )
        )
        source_link = source_link_result.scalar_one_or_none()

    assert feed is not None
    assert feed.title == "Telegram @fondnauk"
    assert feed.subscriber_count == 1
    assert list(feed.topic_embedding) == [2.0] * 1536
    assert source_link is not None
    ensure_topic_coverage.assert_not_awaited()


async def test_subscription_prompt_extracts_telegram_channel_source(
    api_client: AsyncClient,
    mocker,
) -> None:
    parsed_config = SubscriptionConfig(
        topics=["science"],
        schedule_cron="0 8 * * *",
        schedule_was_explicit=True,
        format_instructions="brief summary",
        digest_language="en",
    )
    mocker.patch(
        "news_service.api.routes_subscriptions.parse_subscription",
        new=AsyncMock(return_value=parsed_config),
    )
    ensure_topic_coverage = AsyncMock()
    mocker.patch(
        "news_service.api.routes_subscriptions.ensure_topic_coverage",
        new=ensure_topic_coverage,
    )

    user = await create_user(api_client, timezone="UTC")
    api_key = user["api_key"]

    create_response = await api_client.post(
        "/subscriptions",
        headers={"X-API-Key": api_key},
        json={
            "prompt": "Track @fondnauk every morning",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
            "include_discovered_sources": False,
        },
    )
    assert create_response.status_code == 201
    subscription_id = uuid.UUID(create_response.json()["id"])

    async with async_session_factory() as session:
        result = await session.execute(
            select(RssFeed).where(RssFeed.url == "https://t.me/s/fondnauk")
        )
        feed = result.scalar_one_or_none()
        source_link_result = await session.execute(
            select(SubscriptionSource).where(
                SubscriptionSource.subscription_id == subscription_id,
                SubscriptionSource.feed_id == feed.id if feed is not None else None,
            )
        )
        source_link = source_link_result.scalar_one_or_none()

    assert feed is not None
    assert source_link is not None
    ensure_topic_coverage.assert_not_awaited()
