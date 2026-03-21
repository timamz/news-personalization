import uuid
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient

from news_service.db.session import async_session_factory
from news_service.models.source import Source
from news_service.models.subscription import Subscription
from news_service.schemas.subscription import SubscriptionEditProposalResponse
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
            "prompt_summary": "Anime episode notifications",
            "format_instructions": "brief summary",
            "digest_language_override": "en",
        },
    )
    return api_key, uuid.UUID(sub["id"])


async def test_subscription_edit_proposal_and_apply_updates_canonical_prompt(
    api_client: AsyncClient,
    mocker,
) -> None:
    api_key, subscription_id = await _create_subscription(api_client, mocker)
    proposal = SubscriptionEditProposalResponse(
        canonical_prompt="Notify me when new episodes of Apothecary Diaries and Frieren air.",
        prompt_summary="Anime episode notifications",
        format_instructions="brief summary",
        change_summary="Added Frieren to the tracked show list.",
    )
    mocker.patch(
        "news_service.api.routes_subscriptions.propose_subscription_edit",
        new=AsyncMock(return_value=proposal),
    )

    propose_response = await api_client.post(
        f"/subscriptions/{subscription_id}/edit/propose",
        headers={"X-API-Key": api_key},
        json={"change_request": "Also add Frieren."},
    )

    assert propose_response.status_code == 200
    assert propose_response.json() == {
        "canonical_prompt": proposal.canonical_prompt,
        "prompt_summary": proposal.prompt_summary,
        "format_instructions": proposal.format_instructions,
        "change_summary": proposal.change_summary,
    }

    apply_response = await api_client.post(
        f"/subscriptions/{subscription_id}/edit/apply",
        headers={"X-API-Key": api_key},
        json={
            "canonical_prompt": proposal.canonical_prompt,
            "prompt_summary": proposal.prompt_summary,
            "format_instructions": proposal.format_instructions,
        },
    )

    assert apply_response.status_code == 200
    assert apply_response.json()["prompt_summary"] == "Anime episode notifications"
    assert apply_response.json()["canonical_prompt"] == proposal.canonical_prompt

    async with async_session_factory() as session:
        subscription = await session.get(Subscription, subscription_id)
        assert subscription is not None
        assert subscription.raw_prompt == "Notify me when new episodes of Apothecary Diaries air."
        assert subscription.canonical_prompt == proposal.canonical_prompt
        assert list(subscription.canonical_prompt_embedding) == [2.0] * 1536
        assert subscription.format_instructions == "brief summary"
