"""Backend HTTP client used by the tgbot.

Only two categories remain after agent unification:

- user registration (first /start per telegram_id)
- conversational turns (start / continue against the agent)

Subscription creation / update / delete / source management / timezone / etc.
are handled entirely by the agent via its tools, so the tgbot never hits
those endpoints directly.
"""

import json
from collections.abc import AsyncGenerator

import httpx

from tgbot.core.config import get_settings

settings = get_settings()


class BackendClient:
    """Thin async HTTP wrapper around the backend REST API."""

    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or settings.backend_url).rstrip("/")

    def _request_timeout(self) -> float:
        return settings.backend_request_timeout_seconds

    def _slow_request_timeout(self) -> float:
        return settings.backend_slow_request_timeout_seconds

    async def register_user(self) -> str:
        async with httpx.AsyncClient(timeout=self._request_timeout()) as client:
            response = await client.post(f"{self.base_url}/users")
            response.raise_for_status()
            data = response.json()
            return data["api_key"]

    async def start_subscription_conversation_stream(
        self,
        api_key: str,
        message: str,
    ) -> AsyncGenerator[dict, None]:
        payload: dict[str, object] = {"message": message}
        async with (
            httpx.AsyncClient(
                timeout=httpx.Timeout(self._slow_request_timeout(), connect=10.0),
            ) as client,
            client.stream(
                "POST",
                f"{self.base_url}/subscriptions/conversations/stream",
                headers={"X-API-Key": api_key},
                json=payload,
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip():
                    yield json.loads(line)

    async def continue_subscription_conversation_stream(
        self,
        api_key: str,
        conversation_id: str,
        message: str,
    ) -> AsyncGenerator[dict, None]:
        async with (
            httpx.AsyncClient(
                timeout=httpx.Timeout(self._slow_request_timeout(), connect=10.0),
            ) as client,
            client.stream(
                "POST",
                f"{self.base_url}/subscriptions/conversations/{conversation_id}/messages/stream",
                headers={"X-API-Key": api_key},
                json={"message": message},
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip():
                    yield json.loads(line)
