"""Conversation-based subscription setup endpoints."""

import json
import logging
import uuid
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.conversational import run_conversation_turn_streaming
from news_service.api.dependencies import get_current_user
from news_service.core.config import get_settings
from news_service.core.redis import get_redis_client
from news_service.db.session import get_session
from news_service.models.source import Source
from news_service.models.subscription import Subscription
from news_service.models.subscription_source import SubscriptionSource
from news_service.models.user import User
from news_service.schemas.conversation import (
    AgentTurnOutput,
    ConversationMessageRequest,
    ConversationStartRequest,
    ConversationState,
    ExistingSubscriptionContext,
)
from news_service.services.reddit import extract_reddit_subreddit_from_url
from news_service.services.telegram import extract_telegram_channel_from_url
from news_service.services.twitter import extract_twitter_account_from_url

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
    *,
    db_session: AsyncSession,
    user: User,
) -> AsyncGenerator[str, None]:
    """Shared streaming generator for both start and continue endpoints."""
    user_spec = ""
    conversation_summary = user.conversation_summary or ""
    if conv_state.existing_config is not None:
        user_spec = conv_state.existing_config.user_spec

    async for event in run_conversation_turn_streaming(
        messages,
        db_session=db_session,
        user=user,
        user_spec=user_spec,
        conversation_summary=conversation_summary,
        user_language=conv_state.user_language,
        existing_config=conv_state.existing_config,
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


def _classify_source_url(url: str) -> tuple[str, str] | None:
    """Extract (kind, identifier) from a source URL, or None if unrecognized."""
    channel = extract_telegram_channel_from_url(url)
    if channel is not None:
        return ("telegram", channel)
    subreddit = extract_reddit_subreddit_from_url(url)
    if subreddit is not None:
        return ("reddit", subreddit)
    account = extract_twitter_account_from_url(url)
    if account is not None:
        return ("twitter", account)
    return None


async def _build_existing_config(
    session: AsyncSession,
    subscription: Subscription,
) -> ExistingSubscriptionContext:
    """Build an ExistingSubscriptionContext from a subscription and its linked sources."""
    source_result = await session.execute(
        select(Source.url)
        .join(SubscriptionSource, SubscriptionSource.source_id == Source.id)
        .where(SubscriptionSource.subscription_id == subscription.id)
    )
    source_urls = [row[0] for row in source_result.all()]

    telegram_channels: list[str] = []
    reddit_subreddits: list[str] = []
    twitter_accounts: list[str] = []
    for url in source_urls:
        classified = _classify_source_url(url)
        if classified is None:
            continue
        kind, identifier = classified
        if kind == "telegram":
            telegram_channels.append(identifier)
        elif kind == "reddit":
            reddit_subreddits.append(identifier)
        elif kind == "twitter":
            twitter_accounts.append(identifier)

    return ExistingSubscriptionContext(
        subscription_id=str(subscription.id),
        user_spec=subscription.user_spec,
        delivery_mode=subscription.delivery_mode,
        schedule_cron=subscription.schedule_cron,
        format_instructions=subscription.format_instructions,
        digest_language=subscription.digest_language,
        fixed_telegram_channels=telegram_channels,
        fixed_reddit_subreddits=reddit_subreddits,
        fixed_twitter_accounts=twitter_accounts,
    )


@router.post("/stream")
async def start_conversation_stream(
    payload: ConversationStartRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> StreamingResponse:
    """Start a new conversation with streaming status updates (NDJSON)."""
    existing_config: ExistingSubscriptionContext | None = None

    if payload.mode == "edit" and payload.subscription_id is not None:
        result = await session.execute(
            select(Subscription).where(
                Subscription.id == payload.subscription_id,
                Subscription.user_id == user.id,
            )
        )
        subscription = result.scalar_one_or_none()
        if subscription is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Subscription not found",
            )
        if not subscription.is_active:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Subscription is inactive",
            )
        existing_config = await _build_existing_config(session, subscription)

    conversation_id = uuid.uuid4().hex
    messages: list[dict] = [{"role": "user", "content": payload.message}]
    conv_state = ConversationState(
        user_id=str(user.id),
        messages=messages,
        user_language=payload.user_language,
        user_timezone=payload.user_timezone,
        mode=payload.mode,
        existing_config=existing_config,
    )
    return StreamingResponse(
        _run_turn_streaming(conversation_id, messages, conv_state, db_session=session, user=user),
        media_type="application/x-ndjson",
    )


@router.post("/{conversation_id}/messages/stream")
async def continue_conversation_stream(
    conversation_id: str,
    payload: ConversationMessageRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
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
        _run_turn_streaming(
            conversation_id, conv_state.messages, conv_state, db_session=session, user=user
        ),
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
