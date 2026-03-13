from dataclasses import dataclass
from datetime import datetime

import httpx

from tgbot.core.config import get_settings

settings = get_settings()
_UNSET = object()


@dataclass
class SubscriptionInfo:
    id: str
    topics: list[str]
    delivery_mode: str
    schedule_cron: str | None
    format_instructions: str
    digest_language: str


@dataclass
class SubscriptionParseInfo:
    topics: list[str]
    delivery_mode: str
    schedule_cron: str | None
    schedule_was_explicit: bool
    format_instructions: str
    digest_language: str


@dataclass
class RecentEventsPreviewInfo:
    news_item_ids: list[str]
    subject: str
    body: str


@dataclass
class UserInfo:
    id: str
    api_key: str
    timezone: str | None
    created_at: datetime


@dataclass
class TimezoneCandidateInfo:
    label: str
    timezone: str
    local_time: str


@dataclass
class TimezoneResolutionInfo:
    status: str
    candidates: list[TimezoneCandidateInfo]


class BackendClient:
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

    async def get_current_user(self, api_key: str) -> UserInfo:
        async with httpx.AsyncClient(timeout=self._request_timeout()) as client:
            response = await client.get(
                f"{self.base_url}/users/me",
                headers={"X-API-Key": api_key},
            )
            response.raise_for_status()
            data = response.json()
            return UserInfo(
                id=data["id"],
                api_key=data["api_key"],
                timezone=data.get("timezone"),
                created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
            )

    async def resolve_timezone(self, api_key: str, query: str) -> TimezoneResolutionInfo:
        async with httpx.AsyncClient(timeout=self._request_timeout()) as client:
            response = await client.post(
                f"{self.base_url}/users/resolve-timezone",
                headers={"X-API-Key": api_key},
                json={"query": query},
            )
            response.raise_for_status()
            data = response.json()
            return TimezoneResolutionInfo(
                status=data["status"],
                candidates=[
                    TimezoneCandidateInfo(
                        label=item["label"],
                        timezone=item["timezone"],
                        local_time=item["local_time"],
                    )
                    for item in data["candidates"]
                ],
            )

    async def update_user_timezone(self, api_key: str, timezone: str) -> UserInfo:
        async with httpx.AsyncClient(timeout=self._request_timeout()) as client:
            response = await client.patch(
                f"{self.base_url}/users/me",
                headers={"X-API-Key": api_key},
                json={"timezone": timezone},
            )
            response.raise_for_status()
            data = response.json()
            return UserInfo(
                id=data["id"],
                api_key=data["api_key"],
                timezone=data.get("timezone"),
                created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00")),
            )

    async def create_subscription(
        self,
        api_key: str,
        prompt: str,
        delivery_webhook_url: str,
        fixed_telegram_channels: list[str] | None = None,
        fixed_reddit_subreddits: list[str] | None = None,
        fixed_twitter_accounts: list[str] | None = None,
        include_discovered_sources: bool | None = None,
        schedule_cron_override: str | None = None,
        manual_only: bool | None = None,
        delivery_mode: str | None = None,
        digest_language: str | None = None,
    ) -> SubscriptionInfo:
        payload: dict[str, object] = {
            "prompt": prompt,
            "delivery_webhook_url": delivery_webhook_url,
        }
        if fixed_telegram_channels is not None:
            payload["fixed_telegram_channels"] = fixed_telegram_channels
        if fixed_reddit_subreddits is not None:
            payload["fixed_reddit_subreddits"] = fixed_reddit_subreddits
        if fixed_twitter_accounts is not None:
            payload["fixed_twitter_accounts"] = fixed_twitter_accounts
        if include_discovered_sources is not None:
            payload["include_discovered_sources"] = include_discovered_sources
        if schedule_cron_override is not None:
            payload["schedule_cron_override"] = schedule_cron_override
        if manual_only is not None:
            payload["manual_only"] = manual_only
        if delivery_mode is not None:
            payload["delivery_mode"] = delivery_mode
        if digest_language is not None:
            payload["digest_language_override"] = digest_language

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
                digest_language=data["digest_language"],
            )

    async def parse_subscription_prompt(self, api_key: str, prompt: str) -> SubscriptionParseInfo:
        async with httpx.AsyncClient(timeout=self._slow_request_timeout()) as client:
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
        async with httpx.AsyncClient(timeout=self._slow_request_timeout()) as client:
            response = await client.post(
                f"{self.base_url}/subscriptions/parse-schedule",
                headers={"X-API-Key": api_key},
                json={"schedule_text": schedule_text},
            )
            response.raise_for_status()
            data = response.json()
            return data["schedule_cron"]

    async def list_subscriptions(self, api_key: str) -> list[SubscriptionInfo]:
        async with httpx.AsyncClient(timeout=self._request_timeout()) as client:
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
                    digest_language=s["digest_language"],
                )
                for s in response.json()
            ]

    async def delete_subscription(self, api_key: str, subscription_id: str) -> None:
        async with httpx.AsyncClient(timeout=self._request_timeout()) as client:
            response = await client.delete(
                f"{self.base_url}/subscriptions/{subscription_id}",
                headers={"X-API-Key": api_key},
            )
            response.raise_for_status()

    async def update_subscription(
        self,
        api_key: str,
        subscription_id: str,
        *,
        schedule_cron: str | None | object = _UNSET,
        format_instructions: str | object = _UNSET,
        delivery_webhook_url: str | None | object = _UNSET,
        digest_language: str | object = _UNSET,
    ) -> SubscriptionInfo:
        payload: dict[str, object | None] = {}
        if schedule_cron is not _UNSET:
            payload["schedule_cron"] = schedule_cron
        if format_instructions is not _UNSET:
            payload["format_instructions"] = format_instructions
        if delivery_webhook_url is not _UNSET:
            payload["delivery_webhook_url"] = delivery_webhook_url
        if digest_language is not _UNSET:
            payload["digest_language"] = digest_language

        async with httpx.AsyncClient(timeout=self._request_timeout()) as client:
            response = await client.patch(
                f"{self.base_url}/subscriptions/{subscription_id}",
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
                digest_language=data["digest_language"],
            )

    async def list_recent_events(
        self,
        api_key: str,
        subscription_id: str,
    ) -> RecentEventsPreviewInfo | None:
        async with httpx.AsyncClient(timeout=self._slow_request_timeout()) as client:
            response = await client.get(
                f"{self.base_url}/subscriptions/{subscription_id}/recent-events",
                headers={"X-API-Key": api_key},
            )
            response.raise_for_status()
            payload = response.json()
            if payload is None:
                return None
            return RecentEventsPreviewInfo(
                news_item_ids=payload["news_item_ids"],
                subject=payload["subject"],
                body=payload["body"],
            )

    async def acknowledge_recent_events(
        self,
        api_key: str,
        subscription_id: str,
        news_item_ids: list[str],
    ) -> None:
        async with httpx.AsyncClient(timeout=self._request_timeout()) as client:
            response = await client.post(
                f"{self.base_url}/subscriptions/{subscription_id}/recent-events/acknowledge",
                headers={"X-API-Key": api_key},
                json={"news_item_ids": news_item_ids},
            )
            response.raise_for_status()

    async def send_now(self, api_key: str, subscription_id: str) -> dict[str, str]:
        async with httpx.AsyncClient(timeout=self._request_timeout()) as client:
            response = await client.post(
                f"{self.base_url}/subscriptions/{subscription_id}/send-now",
                headers={"X-API-Key": api_key},
            )
            response.raise_for_status()
            data = response.json()
            return {"task_id": data["task_id"], "status": data["status"]}
