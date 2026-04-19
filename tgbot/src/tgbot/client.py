"""Backend HTTP client used by the tgbot.

The backend exposes a single persistent conversation per user (keyed by
the authenticated user, not by a conversation id). The tgbot therefore
needs only three calls:

- user registration (first /start per telegram_id)
- a single streaming message endpoint
- a thread-reset endpoint (invoked on /start)

Subscription management, source CRUD, timezone, etc. are driven entirely
by the backend agent's tools, so the tgbot never talks to those
endpoints directly.
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

    async def acknowledge_onboarding(self, api_key: str) -> None:
        """Tell the backend the user has seen the frontend's onboarding screen.

        The backend flips ``has_onboarded`` so the conversational agent
        will not re-greet the user on their first real message.
        """
        async with httpx.AsyncClient(timeout=self._request_timeout()) as client:
            response = await client.post(
                f"{self.base_url}/users/me/acknowledge-onboarding",
                headers={"X-API-Key": api_key},
            )
            response.raise_for_status()

    async def api_key_is_valid(self, api_key: str) -> bool:
        """Return True if the backend recognizes this key, False on 401.

        Used to detect dangling local keys after the backend DB is wiped
        so the tgbot can re-register instead of looping on 401s.
        """
        async with httpx.AsyncClient(timeout=self._request_timeout()) as client:
            response = await client.get(
                f"{self.base_url}/users/me",
                headers={"X-API-Key": api_key},
            )
            if response.status_code == 401:
                return False
            response.raise_for_status()
            return True

    async def send_conversation_message_stream(
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

    async def reset_conversation(self, api_key: str) -> None:
        async with httpx.AsyncClient(timeout=self._request_timeout()) as client:
            response = await client.delete(
                f"{self.base_url}/subscriptions/conversations",
                headers={"X-API-Key": api_key},
            )
            response.raise_for_status()
