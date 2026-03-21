"""Conversation-based subscription setup endpoints."""

import json
import logging
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from news_service.agents.subscription_parser import run_conversation_turn_streaming
from news_service.api.dependencies import get_current_user
from news_service.core.config import get_settings
from news_service.core.redis import get_redis_client
from news_service.models.user import User
from news_service.schemas.conversation import (
    AgentTurnOutput,
    ConversationMessageRequest,
    ConversationStartRequest,
    ConversationState,
)

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/subscriptions/conversations", tags=["conversations"])

REDIS_KEY_PREFIX = "conv:"


def _redis_key(conversation_id: str) -> str:
    return f"{REDIS_KEY_PREFIX}{conversation_id}"


async def _load_state(conversation_id: str, user_id: str) -> ConversationState:
    """Load conversation state from Redis, raising 404/403 as appropriate."""
    redis = get_redis_client()
    try:
        raw = await redis.get(_redis_key(conversation_id))
    finally:
        await redis.aclose()

    if raw is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or expired",
        )
    state = ConversationState.model_validate_json(raw)
    if state.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied",
        )
    return state


async def _save_state(conversation_id: str, state: ConversationState) -> None:
    """Save conversation state to Redis with TTL."""
    redis = get_redis_client()
    try:
        await redis.set(
            _redis_key(conversation_id),
            state.model_dump_json(),
            ex=settings.conversation_ttl_seconds,
        )
    finally:
        await redis.aclose()


async def _delete_state(conversation_id: str) -> None:
    """Delete conversation state from Redis."""
    redis = get_redis_client()
    try:
        await redis.delete(_redis_key(conversation_id))
    finally:
        await redis.aclose()


async def _run_turn_streaming(
    conversation_id: str,
    messages: list[dict],
    conv_state: ConversationState,
) -> AsyncGenerator[str, None]:
    """Shared streaming generator for both start and continue endpoints."""
    async for event in run_conversation_turn_streaming(
        messages,
        user_language=conv_state.user_language,
        user_timezone=conv_state.user_timezone,
    ):
        if event["event"] == "done":
            output = AgentTurnOutput.model_validate(event["output"])
            messages.extend(event["new_messages"])
            conv_state.messages = messages
            conv_state.status = output.status
            conv_state.finalized_config = output.finalized_config
            await _save_state(conversation_id, conv_state)
            yield (
                json.dumps(
                    {
                        "event": "done",
                        "conversation_id": conversation_id,
                        "agent_message": output.message,
                        "status": output.status,
                        "finalized_config": (
                            output.finalized_config.model_dump()
                            if output.finalized_config
                            else None
                        ),
                    }
                )
                + "\n"
            )
        else:
            yield json.dumps(event) + "\n"


@router.post("/stream")
async def start_conversation_stream(
    payload: ConversationStartRequest,
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Start a new conversation with streaming status updates (NDJSON)."""
    conversation_id = uuid.uuid4().hex
    messages: list[dict] = [{"role": "user", "content": payload.message}]
    conv_state = ConversationState(
        user_id=str(user.id),
        messages=messages,
        user_language=payload.user_language,
        user_timezone=payload.user_timezone,
    )
    return StreamingResponse(
        _run_turn_streaming(conversation_id, messages, conv_state),
        media_type="application/x-ndjson",
    )


@router.post("/{conversation_id}/messages/stream")
async def continue_conversation_stream(
    conversation_id: str,
    payload: ConversationMessageRequest,
    user: User = Depends(get_current_user),
) -> StreamingResponse:
    """Continue a conversation with streaming status updates (NDJSON)."""
    conv_state = await _load_state(conversation_id, str(user.id))
    if conv_state.status == "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Conversation already finalized",
        )
    conv_state.messages.append({"role": "user", "content": payload.message})
    return StreamingResponse(
        _run_turn_streaming(conversation_id, conv_state.messages, conv_state),
        media_type="application/x-ndjson",
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
) -> None:
    """Cancel and delete a subscription setup conversation."""
    await _load_state(conversation_id, str(user.id))
    await _delete_state(conversation_id)
