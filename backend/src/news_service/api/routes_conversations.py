"""Conversation-based subscription setup endpoints."""

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, status

from news_service.agents.subscription_parser import run_conversation_turn
from news_service.api.dependencies import get_current_user
from news_service.core.config import get_settings
from news_service.core.redis import get_redis_client
from news_service.models.user import User
from news_service.schemas.conversation import (
    ConversationMessageRequest,
    ConversationStartRequest,
    ConversationState,
    ConversationTurnResponse,
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


@router.post("", response_model=ConversationTurnResponse)
async def start_conversation(
    payload: ConversationStartRequest,
    user: User = Depends(get_current_user),
) -> ConversationTurnResponse:
    """Start a new subscription setup conversation."""
    conversation_id = uuid.uuid4().hex

    messages: list[dict[str, str]] = [{"role": "user", "content": payload.message}]

    agent_output = await run_conversation_turn(
        messages,
        user_language=payload.user_language,
        user_timezone=payload.user_timezone,
    )

    messages.append({"role": "assistant", "content": agent_output.message})

    state = ConversationState(
        user_id=str(user.id),
        messages=messages,
        status=agent_output.status,
        finalized_config=agent_output.finalized_config,
        user_language=payload.user_language,
        user_timezone=payload.user_timezone,
    )
    await _save_state(conversation_id, state)

    return ConversationTurnResponse(
        conversation_id=conversation_id,
        agent_message=agent_output.message,
        status=agent_output.status,
        choices=agent_output.choices,
        finalized_config=agent_output.finalized_config,
    )


@router.post("/{conversation_id}/messages", response_model=ConversationTurnResponse)
async def continue_conversation(
    conversation_id: str,
    payload: ConversationMessageRequest,
    user: User = Depends(get_current_user),
) -> ConversationTurnResponse:
    """Continue an existing subscription setup conversation."""
    state = await _load_state(conversation_id, str(user.id))

    if state.status == "ready":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Conversation already finalized",
        )

    state.messages.append({"role": "user", "content": payload.message})

    agent_output = await run_conversation_turn(
        state.messages,
        user_language=state.user_language,
        user_timezone=state.user_timezone,
    )

    state.messages.append({"role": "assistant", "content": agent_output.message})
    state.status = agent_output.status
    state.finalized_config = agent_output.finalized_config

    await _save_state(conversation_id, state)

    return ConversationTurnResponse(
        conversation_id=conversation_id,
        agent_message=agent_output.message,
        status=agent_output.status,
        choices=agent_output.choices,
        finalized_config=agent_output.finalized_config,
    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def cancel_conversation(
    conversation_id: str,
    user: User = Depends(get_current_user),
) -> None:
    """Cancel and delete a subscription setup conversation."""
    await _load_state(conversation_id, str(user.id))
    await _delete_state(conversation_id)
