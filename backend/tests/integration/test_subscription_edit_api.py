import uuid

import pytest
from httpx import AsyncClient

from news_service.db.session import async_session_factory
from news_service.models.source import Source
from news_service.models.subscription import Subscription
from tests.integration.helpers import create_subscription_via_stream, create_user

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def _create_subscription(api_client: AsyncClient, mocker) -> tuple[str, uuid.UUID]:
    async def fake_ensure_prompt_coverage(session, raw_prompt, prompt_embedding):  # noqa: ANN001
        assert raw_prompt == "Notify me when new episodes of Apothecary Diaries air."
        assert prompt_embedding == [2.0] * 1536
        src = Source(
            url="https://example.com/anime.xml",
            title="Anime Feed",
            source_description=f"Anime Feed ({raw_prompt})",
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
            "prompt": "Notify me when new episodes of Apothecary Diaries air.",
            "delivery_webhook_url": "http://frontend.example.test/deliver/1",
            "delivery_mode": "event",
            "format_instructions": "brief summary",
            "digest_language_override": "en",
        },
    )
    return api_key, uuid.UUID(sub["id"])


async def test_apply_config_updates_format_instructions(
    api_client: AsyncClient,
    mocker,
) -> None:
    api_key, subscription_id = await _create_subscription(api_client, mocker)

    apply_response = await api_client.post(
        f"/subscriptions/{subscription_id}/edit/apply-config",
        headers={"X-API-Key": api_key},
        json={
            "delivery_mode": "event",
            "format_instructions": "detailed analysis",
            "digest_language": "en",
        },
    )

    assert apply_response.status_code == 200
    assert apply_response.json()["format_instructions"] == "detailed analysis"

    async with async_session_factory() as session:
        subscription = await session.get(Subscription, subscription_id)
        assert subscription is not None
        assert subscription.raw_prompt == "Notify me when new episodes of Apothecary Diaries air."
        assert subscription.format_instructions == "detailed analysis"
