import logging
from dataclasses import dataclass

import httpx

from tgbot.core.config import get_settings

logger = logging.getLogger(__name__)

settings = get_settings()


@dataclass
class SubscriptionInfo:
    id: str
    raw_prompt: str
    topics: list[str]
    schedule_cron: str
    format_instructions: str
    is_active: bool


class BackendClient:
    def __init__(self, base_url: str | None = None) -> None:
        self.base_url = (base_url or settings.backend_url).rstrip("/")

    async def register_user(self) -> str:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(f"{self.base_url}/users")
            response.raise_for_status()
            data = response.json()
            return data["api_key"]

    async def create_subscription(
        self,
        api_key: str,
        prompt: str,
        delivery_webhook_url: str,
    ) -> SubscriptionInfo:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/subscriptions",
                headers={"X-API-Key": api_key},
                json={"prompt": prompt, "delivery_webhook_url": delivery_webhook_url},
            )
            response.raise_for_status()
            data = response.json()
            return SubscriptionInfo(
                id=data["id"],
                raw_prompt=data["raw_prompt"],
                topics=data["topics"],
                schedule_cron=data["schedule_cron"],
                format_instructions=data["format_instructions"],
                is_active=data["is_active"],
            )

    async def list_subscriptions(self, api_key: str) -> list[SubscriptionInfo]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{self.base_url}/subscriptions",
                headers={"X-API-Key": api_key},
            )
            response.raise_for_status()
            return [
                SubscriptionInfo(
                    id=s["id"],
                    raw_prompt=s["raw_prompt"],
                    topics=s["topics"],
                    schedule_cron=s["schedule_cron"],
                    format_instructions=s["format_instructions"],
                    is_active=s["is_active"],
                )
                for s in response.json()
            ]

    async def delete_subscription(self, api_key: str, subscription_id: str) -> None:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.delete(
                f"{self.base_url}/subscriptions/{subscription_id}",
                headers={"X-API-Key": api_key},
            )
            response.raise_for_status()
