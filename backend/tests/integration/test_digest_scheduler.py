import uuid
from datetime import UTC, datetime

import pytest
from httpx import AsyncClient

from news_service.db.session import async_session_factory
from news_service.models.source import Source
from news_service.models.subscription import Subscription
from news_service.tasks import schedule_digests
from tests.integration.helpers import create_subscription_via_stream, create_user

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_dispatcher_queues_due_subscription_created_via_api(
    api_client: AsyncClient,
    mocker,
) -> None:
    async def fake_ensure_prompt_coverage(session, topic_text, prompt_embedding):  # noqa: ANN001
        assert topic_text == "AI updates every morning in a brief summary"
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
            "prompt": "AI updates every morning in a brief summary",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
            "schedule_cron_override": "0 8 * * *",
            "digest_language_override": "en",
        },
    )
    subscription_id = uuid.UUID(sub["id"])

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
