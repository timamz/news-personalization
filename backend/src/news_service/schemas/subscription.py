import uuid
from datetime import datetime

from pydantic import BaseModel, Field


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


class SubscriptionConfig(BaseModel):
    """Structured output from the Parser Agent."""

    topics: list[str] = Field(..., min_length=1, description="List of news topics")
    schedule_cron: str = Field(..., description="Cron expression for delivery schedule")
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
    schedule_cron: str
    format_instructions: str
    digest_language: str
    delivery_webhook_url: str | None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}
