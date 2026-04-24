"""Conversation endpoint: one persistent thread per user.

Every user message flows through the conversational agent against a single
Redis-backed transcript keyed by ``user_id``. There is no ``conversation_id``
in the API: the user has one ongoing chat and the backend keeps it alive.

After each turn:

- If the agent called ``close_scenario``, the messages up to (but not
  including) the current exchange are collapsed into a one-line entry in
  ``compacted_log`` and the hot transcript is reset to just the latest
  user/assistant pair. That line is rendered back into the next turn's
  system prompt so continuity is preserved.
- If the hot transcript crosses ``conversation_hot_max_bytes``, a
  deterministic guardrail drops the oldest entries as a safety floor.

The long Redis TTL (``conversation_ttl_seconds`` -- 30 days by default) is
just a dormancy floor; real bounded-size is provided by the agent itself.
"""

import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.conversational import run_conversation_turn_streaming
from news_service.api.dependencies import get_current_user
from news_service.core.config import get_settings
from news_service.core.llm_usage import user_tag
from news_service.core.redis import get_redis_client
from news_service.db.session import get_session
from news_service.models.user import User
from news_service.schemas.conversation import (
    AgentTurnOutput,
    ConversationState,
    ConversationTurnRequest,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/subscriptions/conversations", tags=["conversations"])

REDIS_KEY_PREFIX = "conv:user:"


def _redis_key(user_id: str) -> str:
    return f"{REDIS_KEY_PREFIX}{user_id}"


async def _load_state(user_id: str) -> ConversationState:
    """Return the stored thread or a fresh one if nothing is cached yet."""
    raw = await get_redis_client().get(_redis_key(user_id))
    if raw is None:
        return ConversationState(user_id=user_id)
    return ConversationState.model_validate_json(raw)


async def _save_state(state: ConversationState) -> None:
    await get_redis_client().set(
        _redis_key(state.user_id),
        state.model_dump_json(),
        ex=settings.conversation_ttl_seconds,
    )


async def _delete_state(user_id: str) -> None:
    await get_redis_client().delete(_redis_key(user_id))


def _messages_byte_size(messages: list[dict]) -> int:
    return len(json.dumps(messages, ensure_ascii=False).encode("utf-8"))


def _apply_scenario_close(state: ConversationState, summary: str) -> None:
    """Move messages up to the latest exchange into compacted_log.

    The tail kept live is the last user turn plus the assistant reply to
    it, so an immediate follow-up still has natural context. Anything
    older is collapsed into ``summary`` (one line).
    """
    cleaned = summary.strip()
    if not cleaned:
        return
    state.compacted_log.append(cleaned)
    tail_len = 2 if len(state.messages) >= 2 else len(state.messages)
    state.messages = state.messages[-tail_len:]


def _enforce_size_guardrail(state: ConversationState, max_bytes: int) -> None:
    """Drop the oldest messages until the hot transcript fits in the cap.

    Fires only when the agent forgets to close scenarios. Deterministic,
    no LLM call. Records one compacted_log entry so the trim is visible.
    """
    if _messages_byte_size(state.messages) <= max_bytes:
        return
    dropped = 0
    while len(state.messages) > 2 and _messages_byte_size(state.messages) > max_bytes:
        state.messages.pop(0)
        dropped += 1
    if dropped:
        state.compacted_log.append(
            f"[auto-trimmed {dropped} older messages to stay under size cap]"
        )


async def _run_turn_streaming(
    state: ConversationState,
    *,
    db_session: AsyncSession,
    user: User,
) -> AsyncGenerator[str, None]:
    conversation_summary = user.conversation_summary or ""

    with user_tag(user.id):
        async for event in run_conversation_turn_streaming(
            state.messages,
            db_session=db_session,
            user=user,
            conversation_summary=conversation_summary,
            user_language=state.user_language,
            compacted_log=list(state.compacted_log),
        ):
            if event["event"] == "done":
                output = AgentTurnOutput.model_validate(event["output"])
                state.messages.extend(event["new_messages"])

                shared = event.get("shared_state") or {}
                close_summary = shared.get("scenario_close_summary")
                if close_summary:
                    _apply_scenario_close(state, close_summary)

                _enforce_size_guardrail(state, settings.conversation_hot_max_bytes)

                await _save_state(state)
                yield (
                    json.dumps(
                        {
                            "event": "done",
                            "agent_message": output.message,
                        }
                    )
                    + "\n"
                )
            else:
                yield json.dumps(event) + "\n"


@router.post("/stream")
async def send_conversation_message_stream(
    payload: ConversationTurnRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Send one message against the user's persistent conversation thread."""
    state = await _load_state(str(user.id))
    if payload.user_language:
        state.user_language = payload.user_language
    elif state.user_language is None:
        state.user_language = user.language
    state.messages.append({"role": "user", "content": payload.message})
    return StreamingResponse(
        _run_turn_streaming(state, db_session=session, user=user),
        media_type="application/x-ndjson",
    )


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def reset_conversation(user: User = Depends(get_current_user)) -> None:
    """Drop the persistent thread for this user (e.g. on /start)."""
    await _delete_state(str(user.id))
