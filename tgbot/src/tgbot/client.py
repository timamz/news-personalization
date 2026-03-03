from dataclasses import dataclass

import httpx

from tgbot.core.config import get_settings

settings = get_settings()


@dataclass
class SubscriptionInfo:
    id: str
    topics: list[str]
    delivery_mode: str
    schedule_cron: str | None
    format_instructions: str


@dataclass
class SubscriptionParseInfo:
    topics: list[str]
    delivery_mode: str
    schedule_cron: str | None
    schedule_was_explicit: bool
    format_instructions: str
    digest_language: str


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
        fixed_telegram_channels: list[str] | None = None,
        include_discovered_sources: bool | None = None,
        schedule_cron_override: str | None = None,
        manual_only: bool | None = None,
        delivery_mode: str | None = None,
    ) -> SubscriptionInfo:
        payload: dict[str, object] = {
            "prompt": prompt,
            "delivery_webhook_url": delivery_webhook_url,
        }
        if fixed_telegram_channels is not None:
            payload["fixed_telegram_channels"] = fixed_telegram_channels
        if include_discovered_sources is not None:
            payload["include_discovered_sources"] = include_discovered_sources
        if schedule_cron_override is not None:
            payload["schedule_cron_override"] = schedule_cron_override
        if manual_only is not None:
            payload["manual_only"] = manual_only
        if delivery_mode is not None:
            payload["delivery_mode"] = delivery_mode

        async with httpx.AsyncClient(
            timeout=settings.backend_create_subscription_timeout_seconds
        ) as client:
            response = await client.post(
                f"{self.base_url}/subscriptions",
                headers={"X-API-Key": api_key},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return SubscriptionInfo(
                id=data["id"],
                topics=data["topics"],
                delivery_mode=data["delivery_mode"],
                schedule_cron=data["schedule_cron"],
                format_instructions=data["format_instructions"],
            )

    async def parse_subscription_prompt(self, api_key: str, prompt: str) -> SubscriptionParseInfo:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/subscriptions/parse",
                headers={"X-API-Key": api_key},
                json={"prompt": prompt},
            )
            response.raise_for_status()
            data = response.json()
            return SubscriptionParseInfo(
                topics=data["topics"],
                delivery_mode=data["delivery_mode"],
                schedule_cron=data["schedule_cron"],
                schedule_was_explicit=data["schedule_was_explicit"],
                format_instructions=data["format_instructions"],
                digest_language=data["digest_language"],
            )

    async def parse_schedule(self, api_key: str, schedule_text: str) -> str:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.base_url}/subscriptions/parse-schedule",
                headers={"X-API-Key": api_key},
                json={"schedule_text": schedule_text},
            )
            response.raise_for_status()
            data = response.json()
            return data["schedule_cron"]

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
                    topics=s["topics"],
                    delivery_mode=s.get("delivery_mode", "digest"),
                    schedule_cron=s["schedule_cron"],
                    format_instructions=s["format_instructions"],
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

    async def send_now(self, api_key: str, subscription_id: str) -> dict[str, str]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                f"{self.base_url}/subscriptions/{subscription_id}/send-now",
                headers={"X-API-Key": api_key},
            )
            response.raise_for_status()
            data = response.json()
            return {"task_id": data["task_id"], "status": data["status"]}
