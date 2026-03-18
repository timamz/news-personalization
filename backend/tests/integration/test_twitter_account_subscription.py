import uuid
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from news_service.db.session import async_session_factory
from news_service.models.rss_feed import RssFeed
from news_service.models.subscription_source import SubscriptionSource
from tests.integration.helpers import create_user

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_subscription_with_twitter_account_registers_source(
    api_client: AsyncClient,
    mocker,
) -> None:
    ensure_prompt_coverage = AsyncMock()
    mocker.patch(
        "news_service.api.routes_subscriptions.ensure_prompt_coverage",
        new=ensure_prompt_coverage,
    )

    user = await create_user(api_client, timezone="UTC")
    api_key = user["api_key"]

    create_response = await api_client.post(
        "/subscriptions",
        headers={"X-API-Key": api_key},
        json={
            "prompt": "Track X posts from OpenAI every morning",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
            "fixed_twitter_accounts": ["OpenAI"],
            "include_discovered_sources": False,
            "prompt_summary": "AI updates",
            "schedule_cron_override": "0 8 * * *",
            "format_instructions": "brief summary",
            "digest_language_override": "en",
        },
    )
    assert create_response.status_code == 201
    subscription_id = uuid.UUID(create_response.json()["id"])

    async with async_session_factory() as session:
        result = await session.execute(select(RssFeed).where(RssFeed.url == "https://x.com/openai"))
        feed = result.scalar_one_or_none()
        source_link_result = await session.execute(
            select(SubscriptionSource).where(
                SubscriptionSource.subscription_id == subscription_id,
                SubscriptionSource.feed_id == feed.id if feed is not None else None,
            )
        )
        source_link = source_link_result.scalar_one_or_none()

    assert feed is not None
    assert feed.title == "X @openai"
    assert feed.subscriber_count == 1
    assert list(feed.source_description_embedding) == [2.0] * 1536
    assert source_link is not None
    ensure_prompt_coverage.assert_not_awaited()


async def test_subscription_prompt_extracts_twitter_account_source(
    api_client: AsyncClient,
    mocker,
) -> None:
    ensure_prompt_coverage = AsyncMock()
    mocker.patch(
        "news_service.api.routes_subscriptions.ensure_prompt_coverage",
        new=ensure_prompt_coverage,
    )

    user = await create_user(api_client, timezone="UTC")
    api_key = user["api_key"]

    create_response = await api_client.post(
        "/subscriptions",
        headers={"X-API-Key": api_key},
        json={
            "prompt": "Track https://x.com/OpenAI every morning",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
            "include_discovered_sources": False,
            "prompt_summary": "AI updates",
            "schedule_cron_override": "0 8 * * *",
            "format_instructions": "brief summary",
            "digest_language_override": "en",
        },
    )
    assert create_response.status_code == 201
    subscription_id = uuid.UUID(create_response.json()["id"])

    async with async_session_factory() as session:
        result = await session.execute(select(RssFeed).where(RssFeed.url == "https://x.com/openai"))
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
    ensure_prompt_coverage.assert_not_awaited()
