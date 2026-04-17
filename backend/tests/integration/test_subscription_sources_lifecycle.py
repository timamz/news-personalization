import uuid
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient
from sqlalchemy import select

from news_service.db.session import async_session_factory
from news_service.models.source import Source
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from tests.integration.helpers import create_subscription_via_stream, create_user

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_deactivate_subscription_removes_fixed_source_links(
    api_client: AsyncClient,
    mocker,
) -> None:
    async def fake_ensure_prompt_coverage(session, topic_text, prompt_embedding):  # noqa: ANN001
        assert topic_text == "AI updates every morning"
        assert prompt_embedding == [2.0] * 1536
        src = Source(
            url="https://example.com/rss.xml",
            title="Example Feed",
            source_description=f"Example Feed ({topic_text})",
            source_description_embedding=[0.0] * 1536,
            is_active=True,
            subscriber_count=1,
        )
        session.add(src)
        await session.flush()
        return [src]

    mocker.patch(
        "news_service.api.routes_subscriptions.ensure_prompt_coverage",
        new=fake_ensure_prompt_coverage,
    )

    user = await create_user(api_client, timezone="UTC")
    api_key = user["api_key"]

    sub = await create_subscription_via_stream(
        api_client,
        api_key,
        {
            "prompt": "AI updates every morning",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
            "schedule_cron_override": "0 8 * * *",
            "digest_language_override": "en",
        },
    )
    subscription_id = uuid.UUID(sub["id"])

    async with async_session_factory() as session:
        link_result = await session.execute(
            select(SubscriptionSource).where(SubscriptionSource.subscription_id == subscription_id)
        )
        links = list(link_result.scalars().all())
        assert len(links) == 1

        source = await session.get(Source, links[0].source_id)
        assert source is not None
        assert source.subscriber_count == 1
        assert source.is_active is True

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

        source_result = await session.execute(
            select(Source).where(Source.url == "https://example.com/rss.xml")
        )
        source = source_result.scalar_one_or_none()
        assert source is not None
        assert source.subscriber_count == 0
        assert source.is_active is False


async def test_create_event_subscription_forces_schedule_off(
    api_client: AsyncClient,
    mocker,
) -> None:
    async def fake_ensure_prompt_coverage(session, topic_text, prompt_embedding):  # noqa: ANN001
        assert topic_text == "Notify me when the next episode is announced"
        assert prompt_embedding == [2.0] * 1536
        src = Source(
            url="https://example.com/shows.xml",
            title="Shows Feed",
            source_description=f"Shows Feed ({topic_text})",
            source_description_embedding=[0.0] * 1536,
            is_active=True,
            subscriber_count=1,
        )
        session.add(src)
        await session.flush()
        return [src]

    mocker.patch(
        "news_service.api.routes_subscriptions.ensure_prompt_coverage",
        new=fake_ensure_prompt_coverage,
    )

    user = await create_user(api_client)
    api_key = user["api_key"]

    sub = await create_subscription_via_stream(
        api_client,
        api_key,
        {
            "prompt": "Notify me when the next episode is announced",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
            "delivery_mode": "event",
            "schedule_cron_override": "0 8 * * *",
            "digest_language_override": "en",
        },
    )

    assert sub["delivery_mode"] == "event"
    assert sub["schedule_cron"] is None

    subscription_id = uuid.UUID(sub["id"])
    async with async_session_factory() as session:
        subscription = await session.get(Subscription, subscription_id)
        assert subscription is not None
        assert subscription.delivery_mode == "event"
        assert subscription.schedule_cron is None


async def test_append_subscription_sources_adds_only_new_links(
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
            "prompt": "Track @fondnauk every morning",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
            "fixed_telegram_channels": ["fondnauk"],
            "include_discovered_sources": False,
            "schedule_cron_override": "0 8 * * *",
            "digest_language_override": "en",
        },
    )
    subscription_id = uuid.UUID(sub["id"])

    append_response = await api_client.post(
        f"/subscriptions/{subscription_id}/sources",
        headers={"X-API-Key": api_key},
        json={
            "fixed_telegram_channels": ["fondnauk"],
            "fixed_reddit_subreddits": ["badminton"],
        },
    )

    assert append_response.status_code == 200
    assert append_response.json() == {
        "added_telegram_channels": [],
        "added_reddit_subreddits": ["badminton"],
        "added_twitter_accounts": [],
        "added_sources_count": 1,
    }

    async with async_session_factory() as session:
        links_result = await session.execute(
            select(SubscriptionSource).where(SubscriptionSource.subscription_id == subscription_id)
        )
        links = list(links_result.scalars().all())
        assert len(links) == 2

        telegram_result = await session.execute(
            select(Source).where(Source.url == "https://t.me/s/fondnauk")
        )
        telegram_source = telegram_result.scalar_one_or_none()
        reddit_result = await session.execute(
            select(Source).where(Source.url == "https://www.reddit.com/r/badminton/new/")
        )
        reddit_source = reddit_result.scalar_one_or_none()

        assert telegram_source is not None
        assert reddit_source is not None
        assert telegram_source.subscriber_count == 1
        assert reddit_source.subscriber_count == 1
        assert ensure_prompt_coverage.await_count == 0
