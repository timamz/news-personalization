import uuid

import pytest
from httpx import AsyncClient

from news_service.db.session import async_session_factory
from news_service.models.rss_feed import RssFeed
from news_service.models.subscription import Subscription
from tests.integration.helpers import create_user

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _create_user_and_subscription(
    api_client: AsyncClient,
    mocker,
    *,
    delivery_mode: str = "digest",
) -> tuple[str, uuid.UUID]:
    async def fake_ensure_prompt_coverage(session, raw_prompt, raw_prompt_embedding):  # noqa: ANN001
        assert raw_prompt == "AI updates every morning in a brief summary"
        assert raw_prompt_embedding == [2.0] * 1536
        feed = RssFeed(
            url="https://example.com/rss.xml",
            title="Example Feed",
            source_description=f"Example Feed ({raw_prompt})",
            source_description_embedding=[0.0] * 1536,
            is_active=True,
            subscriber_count=1,
        )
        session.add(feed)
        await session.flush()
        return [feed]

    mocker.patch(
        "news_service.api.routes_subscriptions.ensure_prompt_coverage",
        new=fake_ensure_prompt_coverage,
    )

    user = await create_user(api_client, timezone="UTC")
    api_key = user["api_key"]

    create_response = await api_client.post(
        "/subscriptions",
        headers={"X-API-Key": api_key},
        json={
            "prompt": "AI updates every morning in a brief summary",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
            "delivery_mode": delivery_mode,
            "prompt_summary": "AI updates",
            "schedule_cron_override": "0 8 * * *",
            "format_instructions": "brief summary",
            "digest_language_override": "en",
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
        response.json()["detail"] == "Automatic schedule is available only for digest subscriptions"
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
    async def fake_ensure_prompt_coverage(session, raw_prompt, raw_prompt_embedding):  # noqa: ANN001
        assert raw_prompt == "AI updates"
        assert raw_prompt_embedding == [2.0] * 1536
        feed = RssFeed(
            url="https://example.com/rss.xml",
            title="Example Feed",
            source_description=f"Example Feed ({raw_prompt})",
            source_description_embedding=[0.0] * 1536,
            is_active=True,
            subscriber_count=1,
        )
        session.add(feed)
        await session.flush()
        return [feed]

    mocker.patch(
        "news_service.api.routes_subscriptions.ensure_prompt_coverage",
        new=fake_ensure_prompt_coverage,
    )

    user = await create_user(api_client)
    api_key = user["api_key"]

    create_response = await api_client.post(
        "/subscriptions",
        headers={"X-API-Key": api_key},
        json={
            "prompt": "AI updates",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
            "schedule_cron_override": "not a cron",
            "prompt_summary": "AI updates",
            "format_instructions": "brief summary",
            "digest_language_override": "en",
        },
    )

    assert create_response.status_code == 422
    assert create_response.json()["detail"] == "Invalid cron expression: not a cron"


async def test_create_subscription_applies_digest_language_override(
    api_client: AsyncClient,
    mocker,
) -> None:
    async def fake_ensure_prompt_coverage(session, raw_prompt, raw_prompt_embedding):  # noqa: ANN001
        assert raw_prompt == "AI updates every morning in a brief summary"
        assert raw_prompt_embedding == [2.0] * 1536
        feed = RssFeed(
            url="https://example.com/rss.xml",
            title="Example Feed",
            source_description=f"Example Feed ({raw_prompt})",
            source_description_embedding=[0.0] * 1536,
            is_active=True,
            subscriber_count=1,
        )
        session.add(feed)
        await session.flush()
        return [feed]

    mocker.patch(
        "news_service.api.routes_subscriptions.ensure_prompt_coverage",
        new=fake_ensure_prompt_coverage,
    )

    user = await create_user(api_client, timezone="UTC")
    api_key = user["api_key"]

    create_response = await api_client.post(
        "/subscriptions",
        headers={"X-API-Key": api_key},
        json={
            "prompt": "AI updates every morning in a brief summary",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
            "digest_language_override": "ru",
            "prompt_summary": "AI updates",
            "schedule_cron_override": "0 8 * * *",
            "format_instructions": "brief summary",
        },
    )

    assert create_response.status_code == 201
    assert create_response.json()["digest_language"] == "ru"


async def test_update_subscription_updates_digest_language(
    api_client: AsyncClient,
    mocker,
) -> None:
    api_key, subscription_id = await _create_user_and_subscription(api_client, mocker)

    response = await api_client.patch(
        f"/subscriptions/{subscription_id}",
        headers={"X-API-Key": api_key},
        json={"digest_language": "ru"},
    )

    assert response.status_code == 200
    assert response.json()["digest_language"] == "ru"

    async with async_session_factory() as session:
        subscription = await session.get(Subscription, subscription_id)
        assert subscription is not None
        assert subscription.digest_language == "ru"
