import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from datetime import datetime

import httpx

from tgbot.core.config import get_settings

settings = get_settings()
_UNSET = object()


@dataclass
class SubscriptionInfo:
    id: str
    prompt_summary: str
    delivery_mode: str
    schedule_cron: str | None
    format_instructions: str
    digest_language: str
    short_label: str = ""
    raw_prompt: str | None = None
    canonical_prompt: str | None = None


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


@dataclass
class SubscriptionSourcesAppendInfo:
    added_telegram_channels: list[str]
    added_reddit_subreddits: list[str]
    added_twitter_accounts: list[str]
    added_sources_count: int


@dataclass
class SubscriptionEditProposalInfo:
    canonical_prompt: str
    prompt_summary: str
    format_instructions: str
    change_summary: str


@dataclass
class ConversationTurnInfo:
    conversation_id: str
    agent_message: str
    status: str  # "in_progress" or "ready"
    finalized_config: dict | None


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
        prompt_summary: str | None = None,
        short_label: str | None = None,
        format_instructions: str | None = None,
    ) -> SubscriptionInfo:
        payload = self._build_create_payload(
            prompt,
            delivery_webhook_url,
            fixed_telegram_channels=fixed_telegram_channels,
            fixed_reddit_subreddits=fixed_reddit_subreddits,
            fixed_twitter_accounts=fixed_twitter_accounts,
            include_discovered_sources=include_discovered_sources,
            schedule_cron_override=schedule_cron_override,
            manual_only=manual_only,
            delivery_mode=delivery_mode,
            digest_language=digest_language,
            prompt_summary=prompt_summary,
            short_label=short_label,
            format_instructions=format_instructions,
        )

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
            return self._parse_subscription(data)

    async def create_subscription_stream(
        self,
        api_key: str,
        prompt: str,
        delivery_webhook_url: str,
        **kwargs: object,
    ) -> AsyncGenerator[dict, None]:
        payload = self._build_create_payload(prompt, delivery_webhook_url, **kwargs)
        async with (
            httpx.AsyncClient(
                timeout=httpx.Timeout(
                    settings.backend_create_subscription_timeout_seconds, connect=10.0
                ),
            ) as client,
            client.stream(
                "POST",
                f"{self.base_url}/subscriptions/stream",
                headers={"X-API-Key": api_key},
                json=payload,
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip():
                    yield json.loads(line)

    @staticmethod
    def _build_create_payload(
        prompt: str,
        delivery_webhook_url: str,
        **kwargs: object,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "prompt": prompt,
            "delivery_webhook_url": delivery_webhook_url,
        }
        key_map = {"digest_language": "digest_language_override"}
        for key, value in kwargs.items():
            if value is not None:
                payload[key_map.get(key, key)] = value
        return payload

    @staticmethod
    def _parse_subscription(data: dict) -> SubscriptionInfo:
        return SubscriptionInfo(
            id=data["id"],
            prompt_summary=data["prompt_summary"],
            delivery_mode=data["delivery_mode"],
            schedule_cron=data["schedule_cron"],
            format_instructions=data["format_instructions"],
            digest_language=data["digest_language"],
            short_label=data.get("short_label", ""),
            raw_prompt=data.get("raw_prompt"),
            canonical_prompt=data.get("canonical_prompt"),
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
                    prompt_summary=s["prompt_summary"],
                    delivery_mode=s.get("delivery_mode", "digest"),
                    schedule_cron=s["schedule_cron"],
                    format_instructions=s["format_instructions"],
                    digest_language=s["digest_language"],
                    short_label=s.get("short_label", ""),
                    raw_prompt=s.get("raw_prompt"),
                    canonical_prompt=s.get("canonical_prompt"),
                )
                for s in response.json()
            ]

    async def backfill_short_labels(self, api_key: str) -> int:
        async with httpx.AsyncClient(timeout=self._slow_request_timeout()) as client:
            response = await client.post(
                f"{self.base_url}/subscriptions/backfill-labels",
                headers={"X-API-Key": api_key},
            )
            response.raise_for_status()
            return response.json().get("updated", 0)

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
                prompt_summary=data["prompt_summary"],
                delivery_mode=data["delivery_mode"],
                schedule_cron=data["schedule_cron"],
                format_instructions=data["format_instructions"],
                digest_language=data["digest_language"],
                short_label=data.get("short_label", ""),
                raw_prompt=data.get("raw_prompt"),
                canonical_prompt=data.get("canonical_prompt"),
            )

    async def append_subscription_sources(
        self,
        api_key: str,
        subscription_id: str,
        *,
        fixed_telegram_channels: list[str] | None = None,
        fixed_reddit_subreddits: list[str] | None = None,
        fixed_twitter_accounts: list[str] | None = None,
    ) -> SubscriptionSourcesAppendInfo:
        payload: dict[str, object] = {}
        if fixed_telegram_channels is not None:
            payload["fixed_telegram_channels"] = fixed_telegram_channels
        if fixed_reddit_subreddits is not None:
            payload["fixed_reddit_subreddits"] = fixed_reddit_subreddits
        if fixed_twitter_accounts is not None:
            payload["fixed_twitter_accounts"] = fixed_twitter_accounts

        async with httpx.AsyncClient(timeout=self._request_timeout()) as client:
            response = await client.post(
                f"{self.base_url}/subscriptions/{subscription_id}/sources",
                headers={"X-API-Key": api_key},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return SubscriptionSourcesAppendInfo(
                added_telegram_channels=data["added_telegram_channels"],
                added_reddit_subreddits=data["added_reddit_subreddits"],
                added_twitter_accounts=data["added_twitter_accounts"],
                added_sources_count=data["added_sources_count"],
            )

    async def propose_subscription_edit(
        self,
        api_key: str,
        subscription_id: str,
        *,
        change_request: str,
        draft_canonical_prompt: str | None = None,
        draft_format_instructions: str | None = None,
    ) -> SubscriptionEditProposalInfo:
        payload: dict[str, object] = {"change_request": change_request}
        if draft_canonical_prompt is not None:
            payload["draft_canonical_prompt"] = draft_canonical_prompt
        if draft_format_instructions is not None:
            payload["draft_format_instructions"] = draft_format_instructions

        async with httpx.AsyncClient(timeout=self._slow_request_timeout()) as client:
            response = await client.post(
                f"{self.base_url}/subscriptions/{subscription_id}/edit/propose",
                headers={"X-API-Key": api_key},
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
            return SubscriptionEditProposalInfo(
                canonical_prompt=data["canonical_prompt"],
                prompt_summary=data["prompt_summary"],
                format_instructions=data["format_instructions"],
                change_summary=data["change_summary"],
            )

    async def apply_subscription_edit(
        self,
        api_key: str,
        subscription_id: str,
        *,
        canonical_prompt: str,
        prompt_summary: str,
        format_instructions: str,
    ) -> SubscriptionInfo:
        async with httpx.AsyncClient(timeout=self._request_timeout()) as client:
            response = await client.post(
                f"{self.base_url}/subscriptions/{subscription_id}/edit/apply",
                headers={"X-API-Key": api_key},
                json={
                    "canonical_prompt": canonical_prompt,
                    "prompt_summary": prompt_summary,
                    "format_instructions": format_instructions,
                },
            )
            response.raise_for_status()
            data = response.json()
            return SubscriptionInfo(
                id=data["id"],
                prompt_summary=data["prompt_summary"],
                delivery_mode=data["delivery_mode"],
                schedule_cron=data["schedule_cron"],
                format_instructions=data["format_instructions"],
                digest_language=data["digest_language"],
                short_label=data.get("short_label", ""),
                raw_prompt=data.get("raw_prompt"),
                canonical_prompt=data.get("canonical_prompt"),
            )

    async def list_recent_events_stream(
        self,
        api_key: str,
        subscription_id: str,
    ) -> AsyncGenerator[dict, None]:
        """Stream recent events preview (polls sources, then composes). Yields NDJSON dicts."""
        async with (
            httpx.AsyncClient(
                timeout=httpx.Timeout(
                    settings.backend_create_subscription_timeout_seconds, connect=10.0
                ),
            ) as client,
            client.stream(
                "POST",
                f"{self.base_url}/subscriptions/{subscription_id}/recent-events/stream",
                headers={"X-API-Key": api_key},
            ) as response,
        ):
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.strip():
                    yield json.loads(line)

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

    async def start_subscription_conversation(
        self,
        api_key: str,
        message: str,
        user_language: str | None = None,
        user_timezone: str | None = None,
    ) -> ConversationTurnInfo:
        payload: dict[str, object] = {"message": message}
        if user_language is not None:
            payload["user_language"] = user_language
        if user_timezone is not None:
            payload["user_timezone"] = user_timezone

        async with httpx.AsyncClient(timeout=self._slow_request_timeout()) as client:
            response = await client.post(
                f"{self.base_url}/subscriptions/conversations",
                headers={"X-API-Key": api_key},
                json=payload,
            )
            response.raise_for_status()
            return self._parse_conversation_turn(response.json())

    async def continue_subscription_conversation(
        self,
        api_key: str,
        conversation_id: str,
        message: str,
    ) -> ConversationTurnInfo:
        async with httpx.AsyncClient(timeout=self._slow_request_timeout()) as client:
            response = await client.post(
                f"{self.base_url}/subscriptions/conversations/{conversation_id}/messages",
                headers={"X-API-Key": api_key},
                json={"message": message},
            )
            response.raise_for_status()
            return self._parse_conversation_turn(response.json())

    async def cancel_subscription_conversation(
        self,
        api_key: str,
        conversation_id: str,
    ) -> None:
        async with httpx.AsyncClient(timeout=self._request_timeout()) as client:
            response = await client.delete(
                f"{self.base_url}/subscriptions/conversations/{conversation_id}",
                headers={"X-API-Key": api_key},
            )
            response.raise_for_status()

    async def start_subscription_conversation_stream(
        self,
        api_key: str,
        message: str,
        user_language: str | None = None,
        user_timezone: str | None = None,
    ) -> AsyncGenerator[dict, None]:
        payload: dict[str, object] = {"message": message}
        if user_language is not None:
            payload["user_language"] = user_language
        if user_timezone is not None:
            payload["user_timezone"] = user_timezone

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

    @staticmethod
    def _parse_conversation_turn(data: dict) -> ConversationTurnInfo:
        return ConversationTurnInfo(
            conversation_id=data["conversation_id"],
            agent_message=data["agent_message"],
            status=data["status"],
            finalized_config=data.get("finalized_config"),
        )

    async def send_now(self, api_key: str, subscription_id: str) -> dict[str, str]:
        async with httpx.AsyncClient(timeout=self._request_timeout()) as client:
            response = await client.post(
                f"{self.base_url}/subscriptions/{subscription_id}/send-now",
                headers={"X-API-Key": api_key},
            )
            response.raise_for_status()
            data = response.json()
            return {"task_id": data["task_id"], "status": data["status"]}
