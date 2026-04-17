from typing import Any, Literal

from pydantic import BaseModel, Field


class StreamEvent(BaseModel):
    """A single event in the NDJSON conversation stream."""

    event: Literal["status", "done", "error"] = Field(...)
    status_key: str | None = Field(default=None, description="Translation key for status text")
    data: dict[str, Any] | None = Field(
        default=None, description="Final result payload (for done events)"
    )


class ConversationStartRequest(BaseModel):
    message: str = Field(..., min_length=1, description="Initial user message")
    user_language: str | None = Field(
        default=None, description="Frontend-stored language hint (e.g. 'en', 'ru')"
    )


class ConversationMessageRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message to continue conversation")


class ConversationTurnResponse(BaseModel):
    conversation_id: str = Field(..., description="Unique conversation identifier")
    agent_message: str = Field(..., description="Agent response text to display")


class AgentTurnOutput(BaseModel):
    """Structured output from one agent turn."""

    message: str = Field(..., description="What to show the user")
    status: Literal["in_progress"] = Field(default="in_progress")


class ConversationState(BaseModel):
    user_id: str = Field(..., description="User UUID")
    messages: list[dict] = Field(default_factory=list)
    user_language: str | None = Field(default=None)
