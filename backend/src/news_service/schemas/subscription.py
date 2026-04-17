import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

type DeliveryMode = Literal["digest", "event"]


class SubscriptionCreate(BaseModel):
    prompt: str = Field(..., min_length=5, description="Natural language topic / request")
    preferences: str | None = Field(
        default=None,
        description=(
            "Freeform presentation guidance -- length, format, exclusions, tone. "
            "Rendered into user_spec's Preferences section."
        ),
    )
    delivery_webhook_url: str | None = Field(
        default=None, description="URL where digest will be POSTed"
    )
    fixed_telegram_channels: list[str] = Field(
        default_factory=list,
        description="Telegram channels explicitly chosen by the user",
    )
    fixed_reddit_subreddits: list[str] = Field(
        default_factory=list,
        description="Reddit subreddits explicitly chosen by the user",
    )
    fixed_twitter_accounts: list[str] = Field(
        default_factory=list,
        description="Twitter/X accounts explicitly chosen by the user",
    )
    include_discovered_sources: bool | None = Field(
        default=None,
        description=(
            "Whether to add discovered RSS, Telegram, Reddit, and Twitter/X sources "
            "to the fixed list"
        ),
    )
    schedule_cron_override: str | None = Field(
        default=None,
        description="Override cron schedule selected in the conversational flow",
    )
    manual_only: bool | None = Field(
        default=None,
        description="When true, digest is available only via explicit send-now action",
    )
    delivery_mode: DeliveryMode | None = Field(
        default=None,
        description="Override parsed delivery mode: digest or event notification",
    )
    digest_language_override: str | None = Field(
        default=None,
        min_length=2,
        max_length=16,
        description="Override output language for digests and event notifications",
    )


class SubscriptionResponse(BaseModel):
    id: uuid.UUID
    user_spec: str
    delivery_mode: DeliveryMode
    schedule_cron: str | None
    digest_language: str
    delivery_webhook_url: str | None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class RecentEventsPreviewResponse(BaseModel):
    news_item_ids: list[uuid.UUID]
    subject: str
    body: str


class RecentEventAcknowledgeRequest(BaseModel):
    news_item_ids: list[uuid.UUID] = Field(
        ...,
        min_length=1,
        description="Recent event preview items that were actually shown to the user",
    )


class SubscriptionUpdate(BaseModel):
    schedule_cron: str | None = Field(
        default=None,
        min_length=1,
        description="New cron schedule for digest subscriptions; null disables scheduling",
    )
    user_spec: str | None = Field(
        default=None,
        min_length=1,
        description=(
            "Updated markdown user_spec. Overwrites the existing document. "
            "Use when the user wants to change topic, preferences, or format."
        ),
    )
    delivery_webhook_url: str | None = Field(
        default=None,
        description="Updated delivery webhook URL; null disables webhook delivery",
    )
    digest_language: str | None = Field(
        default=None,
        min_length=2,
        max_length=16,
        description="Updated output language for digests and event notifications",
    )


class SubscriptionSourcesAppendRequest(BaseModel):
    fixed_telegram_channels: list[str] = Field(
        default_factory=list,
        description="Telegram channels to append to this subscription",
    )
    fixed_reddit_subreddits: list[str] = Field(
        default_factory=list,
        description="Reddit subreddits to append to this subscription",
    )
    fixed_twitter_accounts: list[str] = Field(
        default_factory=list,
        description="Twitter/X accounts to append to this subscription",
    )

    @model_validator(mode="after")
    def validate_has_sources(self) -> "SubscriptionSourcesAppendRequest":
        if (
            not self.fixed_telegram_channels
            and not self.fixed_reddit_subreddits
            and not self.fixed_twitter_accounts
        ):
            raise ValueError("At least one source must be provided")
        return self


class SubscriptionSourcesAppendResponse(BaseModel):
    added_telegram_channels: list[str]
    added_reddit_subreddits: list[str]
    added_twitter_accounts: list[str]
    added_sources_count: int


class ScheduleParseRequest(BaseModel):
    schedule_text: str = Field(
        ...,
        min_length=3,
        description="Natural language schedule preference",
    )


class ScheduleParseResponse(BaseModel):
    schedule_cron: str = Field(..., description="Parsed 5-field cron expression")
