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
    delivery_mode: str | None = None,
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
            "prompt": "AI updates every morning in a brief summary",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
            "delivery_mode": delivery_mode,
        },
    )
    assert create_response.status_code == 201
    subscription_id = uuid.UUID(create_response.json()["id"])
    return api_key, subscription_id


async def test_send_now_queues_digest_task(api_client: AsyncClient, mocker) -> None:
    api_key, subscription_id = await _create_user_and_subscription(api_client, mocker)

    delay_mock = mocker.patch("news_service.api.routes_subscriptions.deliver_digest.delay")
    delay_mock.return_value.id = "task-123"

    response = await api_client.post(
        f"/subscriptions/{subscription_id}/send-now",
        headers={"X-API-Key": api_key},
    )

    assert response.status_code == 202
    assert response.json() == {"task_id": "task-123", "status": "queued"}
    delay_mock.assert_called_once_with(str(subscription_id), True)


async def test_send_now_rejects_inactive_subscription(api_client: AsyncClient, mocker) -> None:
    api_key, subscription_id = await _create_user_and_subscription(api_client, mocker)

    async with async_session_factory() as session:
        subscription = await session.get(Subscription, subscription_id)
        assert subscription is not None
        subscription.is_active = False
        await session.commit()

    response = await api_client.post(
        f"/subscriptions/{subscription_id}/send-now",
        headers={"X-API-Key": api_key},
    )
    assert response.status_code == 409
    assert response.json()["detail"] == "Subscription is inactive"


async def test_send_now_rejects_event_subscription(api_client: AsyncClient, mocker) -> None:
    api_key, subscription_id = await _create_user_and_subscription(
        api_client,
        mocker,
        delivery_mode="event",
    )

    response = await api_client.post(
        f"/subscriptions/{subscription_id}/send-now",
        headers={"X-API-Key": api_key},
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Send now is available only for digest subscriptions"
