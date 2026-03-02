from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient

from news_service.schemas.subscription import SubscriptionConfig

pytestmark = pytest.mark.asyncio(loop_scope="session")


async def test_parse_subscription_endpoint_returns_llm_structure(
    api_client: AsyncClient,
    mocker,
) -> None:
    parsed_config = SubscriptionConfig(
        topics=["machine learning"],
        schedule_cron=None,
        schedule_was_explicit=False,
        format_instructions="brief summary",
        digest_language="ru",
    )
    mocker.patch(
        "news_service.api.routes_subscriptions.parse_subscription",
        new=AsyncMock(return_value=parsed_config),
    )

    user_response = await api_client.post("/users")
    assert user_response.status_code == 201
    api_key = user_response.json()["api_key"]

    response = await api_client.post(
        "/subscriptions/parse",
        headers={"X-API-Key": api_key},
        json={"prompt": "Хочу новости по ML"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "topics": ["machine learning"],
        "schedule_cron": None,
        "schedule_was_explicit": False,
        "format_instructions": "brief summary",
        "digest_language": "ru",
    }


async def test_parse_schedule_endpoint_returns_cron(
    api_client: AsyncClient,
    mocker,
) -> None:
    mocker.patch(
        "news_service.api.routes_subscriptions.parse_schedule_preference",
        new=AsyncMock(return_value="0 9 * * 1-5"),
    )

    user_response = await api_client.post("/users")
    assert user_response.status_code == 201
    api_key = user_response.json()["api_key"]

    response = await api_client.post(
        "/subscriptions/parse-schedule",
        headers={"X-API-Key": api_key},
        json={"schedule_text": "каждый будний день в 9"},
    )

    assert response.status_code == 200
    assert response.json() == {"schedule_cron": "0 9 * * 1-5"}
