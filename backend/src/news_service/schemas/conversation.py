"""Wire schemas for the single-thread-per-user conversation API.

There is one persistent chat per user. The API has no conversation ids:
every turn is just a ``ConversationTurnRequest`` against the user-scoped
endpoint, and the backend keeps the transcript in Redis keyed by user id.

Scenario-based compaction keeps the hot transcript bounded: the agent
calls ``close_scenario`` when a logical task finishes, which moves the
messages so far into ``compacted_log`` and keeps only the latest exchange
live. A byte-size guardrail drops the oldest entries if the agent forgets.
"""

from typing import Literal

from pydantic import BaseModel, Field


class ConversationTurnRequest(BaseModel):
    """One message from the user against the persistent thread.

    ``message`` is bounded to ``settings.max_user_message_chars``: a
    chat-shaped UI never needs more, and capping at the API boundary
    prevents a single oversized request from burning the LLM budget
    or overflowing the model's context window.
    """

    message: str = Field(..., min_length=1, max_length=10_000, description="User message")
    user_language: str | None = Field(
        default=None, description="Frontend-stored language hint (e.g. 'en', 'ru')"
    )


class AgentTurnOutput(BaseModel):
    """Structured output from one agent turn."""

    message: str = Field(..., description="What to show the user")
    status: Literal["in_progress"] = Field(default="in_progress")


class ConfirmationDecisionRequest(BaseModel):
    """Frontend's response to a ``requires_confirmation`` event.

    ``nonce`` is the server-generated identifier delivered alongside
    the event. ``decision`` is ``"confirm"`` (run the action) or
    ``"cancel"`` (drop the pending action).
    """

    nonce: str = Field(..., min_length=1, max_length=128)
    decision: Literal["confirm", "cancel"]


class ConfirmationDecisionResponse(BaseModel):
    """Result of a confirmation decision."""

    status: Literal["executed", "cancelled"]
    action: str
    result: str | None = None


class ConversationState(BaseModel):
    """Persistent per-user conversation state held in Redis.

    ``messages`` is the hot transcript the agent sees verbatim next turn.
    ``compacted_log`` is the append-only list of one-line scenario
    summaries written by ``close_scenario`` (or by the size guardrail);
    it is rendered into the system prompt so the agent remembers that a
    task was completed even after the hot transcript was trimmed.
    """

    user_id: str = Field(..., description="User UUID")
    messages: list[dict] = Field(default_factory=list)
    compacted_log: list[str] = Field(default_factory=list)
    user_language: str | None = Field(default=None)
