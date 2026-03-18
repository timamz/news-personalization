from typing import Literal

from pydantic import BaseModel, Field

from news_service.schemas.subscription import DeliveryMode, EventMatchingMode


class ConversationChoice(BaseModel):
    label: str = Field(..., description="Button text to display")
    value: str = Field(..., description="Value to send back as user message if chosen")


class ConversationStartRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Initial user message")
    user_language: str | None = Field(
        default=None, description="Frontend-stored language preference (e.g. 'en', 'ru')"
    )
    user_timezone: str | None = Field(default=None, description="User IANA timezone from profile")


class ConversationMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message to continue conversation")


class FinalizedSubscriptionConfig(BaseModel):
    prompt_summary: str = Field(..., description="Short human-readable summary")
    short_label: str = Field(..., description="Ultra-short 2-3 word label")
    delivery_mode: DeliveryMode = Field(default="digest")
    event_matching_mode: EventMatchingMode = Field(default="basic")
    schedule_cron: str | None = Field(default=None, description="5-field cron expression")
    manual_only: bool = Field(default=False)
    format_instructions: str = Field(default="brief summary")
    digest_language: str = Field(default="en", min_length=2, max_length=16)
    fixed_telegram_channels: list[str] = Field(default_factory=list)
    fixed_reddit_subreddits: list[str] = Field(default_factory=list)
    fixed_twitter_accounts: list[str] = Field(default_factory=list)
    include_discovered_sources: bool = Field(default=True)


class ConversationTurnResponse(BaseModel):
    conversation_id: str = Field(..., description="Unique conversation identifier")
    agent_message: str = Field(..., description="Agent response text to display")
    status: Literal["in_progress", "ready"] = Field(...)
    choices: list[ConversationChoice] | None = Field(
        default=None, description="Optional buttons to render"
    )
    finalized_config: FinalizedSubscriptionConfig | None = Field(
        default=None, description="Populated only when status is ready"
    )


class AgentTurnOutput(BaseModel):
    """Structured output from the subscription parser agent."""

    message: str = Field(..., description="What to show the user")
    status: Literal["in_progress", "ready"] = Field(...)
    choices: list[ConversationChoice] | None = Field(
        default=None, description="Optional choices for button rendering"
    )
    finalized_config: FinalizedSubscriptionConfig | None = Field(
        default=None, description="Populated only when status is ready"
    )


class ConversationState(BaseModel):
    user_id: str = Field(..., description="User UUID")
    messages: list[dict] = Field(default_factory=list)
    status: Literal["in_progress", "ready"] = Field(default="in_progress")
    finalized_config: FinalizedSubscriptionConfig | None = Field(default=None)
    user_language: str | None = Field(default=None)
    user_timezone: str | None = Field(default=None)
