from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from news_service.db.session import async_session_factory
from news_service.models.rss_feed import RssFeed
from news_service.schemas.subscription import SubscriptionConfig

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_subscription_with_telegram_channel_registers_source(
    api_client: AsyncClient,
    mocker,
) -> None:
    parsed_config = SubscriptionConfig(
        topics=["science"],
        schedule_cron="0 8 * * *",
        format_instructions="brief summary",
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
    mocker.patch(
        "news_service.services.coverage.embed_text",
        new=AsyncMock(return_value=[0.0] * 1536),
    )

    user_response = await api_client.post("/users")
    assert user_response.status_code == 201
    api_key = user_response.json()["api_key"]

    create_response = await api_client.post(
        "/subscriptions",
        headers={"X-API-Key": api_key},
        json={
            "prompt": "Track @fondnauk every morning",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
        },
    )
    assert create_response.status_code == 201

    async with async_session_factory() as session:
        result = await session.execute(
            select(RssFeed).where(RssFeed.url == "https://t.me/s/fondnauk")
        )
        feed = result.scalar_one_or_none()

    assert feed is not None
    assert feed.title == "Telegram @fondnauk"
    assert feed.subscriber_count == 1
    ensure_topic_coverage.assert_not_awaited()
