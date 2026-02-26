import uuid
from datetime import datetime

from pydantic import BaseModel, Field


class SubscriptionCreate(BaseModel):
    prompt: str = Field(..., min_length=5, description="Natural language news preference")
    delivery_webhook_url: str | None = Field(
        default=None, description="URL where digest will be POSTed"
    )


class SubscriptionConfig(BaseModel):
    """Structured output from the Parser Agent."""

    topics: list[str] = Field(..., min_length=1, description="List of news topics")
    schedule_cron: str = Field(..., description="Cron expression for delivery schedule")
    format_instructions: str = Field(
        default="brief summary", description="How the user wants to consume news"
    )


class SubscriptionResponse(BaseModel):
    id: uuid.UUID
    raw_prompt: str
    topics: list[str]
    schedule_cron: str
    format_instructions: str
    delivery_webhook_url: str | None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}
