import uuid
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient

from news_service.db.session import async_session_factory
from news_service.models.rss_feed import RssFeed
from news_service.models.subscription import Subscription
from news_service.schemas.subscription import SubscriptionConfig

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _create_user_and_subscription(
    api_client: AsyncClient,
    mocker,
    *,
    delivery_mode: str = "digest",
) -> tuple[str, uuid.UUID]:
    parsed_config = SubscriptionConfig(
        topics=["artificial intelligence"],
        delivery_mode="digest",
        schedule_cron="0 8 * * *",
        schedule_was_explicit=True,
        format_instructions="brief summary",
        digest_language="en",
    )
    mocker.patch(
        "news_service.api.routes_subscriptions.parse_subscription",
        new=AsyncMock(return_value=parsed_config),
    )

    async def fake_ensure_topic_coverage(session, topics):  # noqa: ANN001
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
            "prompt": "AI updates every morning in a brief summary",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
            "delivery_mode": delivery_mode,
        },
    )
    assert create_response.status_code == 201
    return api_key, uuid.UUID(create_response.json()["id"])


async def test_update_subscription_updates_lightweight_fields(
    api_client: AsyncClient,
    mocker,
) -> None:
    api_key, subscription_id = await _create_user_and_subscription(api_client, mocker)

    response = await api_client.patch(
        f"/subscriptions/{subscription_id}",
        headers={"X-API-Key": api_key},
        json={
            "schedule_cron": "0 10 * * 1-5",
            "format_instructions": "detailed analysis",
            "delivery_webhook_url": None,
        },
    )

    assert response.status_code == 200
    assert response.json()["schedule_cron"] == "0 10 * * 1-5"
    assert response.json()["format_instructions"] == "detailed analysis"
    assert response.json()["delivery_webhook_url"] is None

    async with async_session_factory() as session:
        subscription = await session.get(Subscription, subscription_id)
        assert subscription is not None
        assert subscription.schedule_cron == "0 10 * * 1-5"
        assert subscription.format_instructions == "detailed analysis"
        assert subscription.delivery_webhook_url is None


async def test_update_subscription_rejects_schedule_for_event_mode(
    api_client: AsyncClient,
    mocker,
) -> None:
    api_key, subscription_id = await _create_user_and_subscription(
        api_client,
        mocker,
        delivery_mode="event",
    )

    response = await api_client.patch(
        f"/subscriptions/{subscription_id}",
        headers={"X-API-Key": api_key},
        json={"schedule_cron": "0 10 * * 1-5"},
    )

    assert response.status_code == 409
    assert (
        response.json()["detail"]
        == "Automatic schedule is available only for digest subscriptions"
    )


async def test_update_subscription_rejects_empty_patch(
    api_client: AsyncClient,
    mocker,
) -> None:
    api_key, subscription_id = await _create_user_and_subscription(api_client, mocker)

    response = await api_client.patch(
        f"/subscriptions/{subscription_id}",
        headers={"X-API-Key": api_key},
        json={},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "No editable fields were provided"


async def test_update_subscription_rejects_invalid_schedule(
    api_client: AsyncClient,
    mocker,
) -> None:
    api_key, subscription_id = await _create_user_and_subscription(api_client, mocker)

    response = await api_client.patch(
        f"/subscriptions/{subscription_id}",
        headers={"X-API-Key": api_key},
        json={"schedule_cron": "not a cron"},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Invalid cron expression: not a cron"


async def test_create_subscription_rejects_invalid_schedule_override(
    api_client: AsyncClient,
    mocker,
) -> None:
    parsed_config = SubscriptionConfig(
        topics=["artificial intelligence"],
        delivery_mode="digest",
        schedule_cron=None,
        schedule_was_explicit=False,
        format_instructions="brief summary",
        digest_language="en",
    )
    mocker.patch(
        "news_service.api.routes_subscriptions.parse_subscription",
        new=AsyncMock(return_value=parsed_config),
    )

    async def fake_ensure_topic_coverage(session, topics):  # noqa: ANN001
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
            "prompt": "AI updates",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
            "schedule_cron_override": "not a cron",
        },
    )

    assert create_response.status_code == 422
    assert create_response.json()["detail"] == "Invalid cron expression: not a cron"
