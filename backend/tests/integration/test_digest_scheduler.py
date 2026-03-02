import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient

from news_service.db.session import async_session_factory
from news_service.models.rss_feed import RssFeed
from news_service.models.subscription import Subscription
from news_service.schemas.subscription import SubscriptionConfig
from news_service.tasks import schedule_digests

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_dispatcher_queues_due_subscription_created_via_api(
    api_client: AsyncClient,
    mocker,
) -> None:
    parsed_config = SubscriptionConfig(
        topics=["artificial intelligence"],
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
        },
    )
    assert create_response.status_code == 201
    subscription_id = uuid.UUID(create_response.json()["id"])

    due_time = datetime(2026, 2, 26, 8, 0, tzinfo=UTC)
    previous_run = datetime(2026, 2, 25, 8, 0, tzinfo=UTC)

    async with async_session_factory() as session:
        subscription = await session.get(Subscription, subscription_id)
        assert subscription is not None
        subscription.created_at = previous_run
        subscription.last_digest_scheduled_at = previous_run
        await session.commit()

    send_task = mocker.patch.object(schedule_digests.celery_app, "send_task")

    first_run = await schedule_digests._schedule_due_digests(now=due_time)
    send_task.assert_called_once_with(
        schedule_digests.DELIVER_DIGEST_TASK,
        args=[str(subscription_id)],
    )
    assert first_run["queued"] == 1

    async with async_session_factory() as session:
        updated = await session.get(Subscription, subscription_id)
        assert updated is not None
        assert updated.last_digest_scheduled_at == due_time

    second_run = await schedule_digests._schedule_due_digests(now=due_time)
    assert second_run["queued"] == 0
    assert send_task.call_count == 1
