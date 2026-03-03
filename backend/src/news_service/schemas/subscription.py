import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

type DeliveryMode = Literal["digest", "event"]


class SubscriptionCreate(BaseModel):
    prompt: str = Field(..., min_length=5, description="Natural language news preference")
    delivery_webhook_url: str | None = Field(
        default=None, description="URL where digest will be POSTed"
    )
    fixed_telegram_channels: list[str] = Field(
        default_factory=list,
        description="Telegram channels explicitly chosen by the user",
    )
    include_discovered_sources: bool | None = Field(
        default=None,
        description="Whether to add discovered RSS/Telegram sources to the fixed list",
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


class SubscriptionConfig(BaseModel):
    """Structured output from the Parser Agent."""

    topics: list[str] = Field(..., min_length=1, description="List of news topics")
    delivery_mode: DeliveryMode = Field(
        default="digest",
        description="Whether the user wants a periodic digest or event notifications",
    )
    schedule_cron: str | None = Field(
        default=None,
        description="Cron expression for delivery schedule, if explicitly requested",
    )
    schedule_was_explicit: bool = Field(
        ...,
        description="Whether the user explicitly requested automatic schedule in the prompt",
    )
    format_instructions: str = Field(
        default="brief summary", description="How the user wants to consume news"
    )
    digest_language: str = Field(
        ...,
        min_length=2,
        max_length=16,
        description="Language code for digest output (for example: en, ru, es)",
    )


class SubscriptionResponse(BaseModel):
    id: uuid.UUID
    raw_prompt: str
    topics: list[str]
    delivery_mode: DeliveryMode
    schedule_cron: str | None
    format_instructions: str
    digest_language: str
    delivery_webhook_url: str | None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class SubscriptionParseRequest(BaseModel):
    prompt: str = Field(..., min_length=5, description="Natural language subscription request")


class SubscriptionParseResponse(BaseModel):
    topics: list[str]
    delivery_mode: DeliveryMode
    schedule_cron: str | None
    schedule_was_explicit: bool
    format_instructions: str
    digest_language: str


class ScheduleParseRequest(BaseModel):
    schedule_text: str = Field(
        ...,
        min_length=3,
        description="Natural language schedule preference",
    )


class ScheduleParseResponse(BaseModel):
    schedule_cron: str = Field(..., description="Parsed 5-field cron expression")
