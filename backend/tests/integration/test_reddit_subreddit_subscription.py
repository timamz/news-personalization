import uuid
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from news_service.db.session import async_session_factory
from news_service.models.source import Source
from news_service.models.subscription_source import SubscriptionSource
from tests.integration.helpers import create_subscription_via_stream, create_user

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_subscription_with_reddit_subreddit_registers_source(
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

    sub = await create_subscription_via_stream(
        api_client,
        api_key,
        {
            "prompt": "Track r/badminton every morning",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
            "fixed_reddit_subreddits": ["badminton"],
            "include_discovered_sources": False,
            "prompt_summary": "Badminton updates",
            "schedule_cron_override": "0 8 * * *",
            "format_instructions": "brief summary",
            "digest_language_override": "en",
        },
    )
    subscription_id = uuid.UUID(sub["id"])

    async with async_session_factory() as session:
        result = await session.execute(
            select(Source).where(Source.url == "https://www.reddit.com/r/badminton/new/")
        )
        source = result.scalar_one_or_none()
        source_link_result = await session.execute(
            select(SubscriptionSource).where(
                SubscriptionSource.subscription_id == subscription_id,
                SubscriptionSource.source_id == source.id if source is not None else None,
            )
        )
        source_link = source_link_result.scalar_one_or_none()

    assert source is not None
    assert source.title == "Reddit r/badminton"
    assert source.subscriber_count == 1
    assert list(source.source_description_embedding) == [2.0] * 1536
    assert source_link is not None
    ensure_prompt_coverage.assert_not_awaited()


async def test_subscription_prompt_extracts_reddit_subreddit_source(
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

    sub = await create_subscription_via_stream(
        api_client,
        api_key,
        {
            "prompt": "Track r/badminton every morning",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
            "include_discovered_sources": False,
            "prompt_summary": "Badminton updates",
            "schedule_cron_override": "0 8 * * *",
            "format_instructions": "brief summary",
            "digest_language_override": "en",
        },
    )
    subscription_id = uuid.UUID(sub["id"])

    async with async_session_factory() as session:
        result = await session.execute(
            select(Source).where(Source.url == "https://www.reddit.com/r/badminton/new/")
        )
        source = result.scalar_one_or_none()
        source_link_result = await session.execute(
            select(SubscriptionSource).where(
                SubscriptionSource.subscription_id == subscription_id,
                SubscriptionSource.source_id == source.id if source is not None else None,
            )
        )
        source_link = source_link_result.scalar_one_or_none()

    assert source is not None
    assert source_link is not None
    ensure_prompt_coverage.assert_not_awaited()
