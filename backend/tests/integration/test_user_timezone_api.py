import pytest
from httpx import AsyncClient

from news_service.models.source import Source
from tests.integration.helpers import create_user

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_get_and_update_user_timezone(api_client: AsyncClient) -> None:
    user = await create_user(api_client)
    api_key = user["api_key"]

    get_response = await api_client.get("/users/me", headers={"X-API-Key": api_key})

    assert get_response.status_code == 200
    assert get_response.json()["timezone"] is None

    patch_response = await api_client.patch(
        "/users/me",
        headers={"X-API-Key": api_key},
        json={"timezone": "Europe/Berlin"},
    )

    assert patch_response.status_code == 200
    assert patch_response.json()["timezone"] == "Europe/Berlin"


async def test_resolve_user_timezone_returns_candidates(api_client: AsyncClient) -> None:
    user = await create_user(api_client)
    api_key = user["api_key"]

    response = await api_client.post(
        "/users/resolve-timezone",
        headers={"X-API-Key": api_key},
        json={"query": "tiblisi"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "resolved"
    assert response.json()["candidates"][0]["timezone"] == "Asia/Tbilisi"


async def test_create_subscription_rejects_schedule_without_timezone(
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

    user = await create_user(api_client)
    api_key = user["api_key"]

    response = await api_client.post(
        "/subscriptions/stream",
        headers={"X-API-Key": api_key},
        json={
            "prompt": "AI updates every morning",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
            "schedule_cron_override": "0 8 * * *",
            "digest_language_override": "en",
        },
    )

    assert response.status_code == 409
    assert response.json()["detail"] == "Set your timezone before enabling automatic schedules"
