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
- If the hot transcript crosses ``conversation_hot_max_tokens``, a
  deterministic guardrail drops the oldest entries as a safety floor.

The long Redis TTL (``conversation_ttl_seconds`` -- 30 days by default) is
just a dormancy floor; real bounded-size is provided by the agent itself.
"""

import json
import logging
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.agents.conversational import run_conversation_turn_streaming
from news_service.agents.conversational.tools import build_tools_by_name
from news_service.api.dependencies import get_current_user
from news_service.core import confirmations
from news_service.core.config import get_settings
from news_service.core.llm_usage import user_tag
from news_service.core.rate_limit import RateLimitExceeded, check_rate_limit
from news_service.core.redis import get_redis_client
from news_service.db.session import async_session_factory, get_session
from news_service.models.user import User
from news_service.schemas.conversation import (
    AgentTurnOutput,
    ConfirmationDecisionRequest,
    ConfirmationDecisionResponse,
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


def _messages_token_size(messages: list[dict], model: str) -> int:
    """Count tokens across the hot transcript using LiteLLM's tokenizer.

    LiteLLM picks the right tokenizer for the configured provider and
    falls back to a generic estimator when the exact tokenizer is not
    bundled, so this stays correct across model swaps without us
    pinning a tokenizer per provider.
    """
    import litellm

    joined = json.dumps(messages, ensure_ascii=False)
    try:
        return int(litellm.token_counter(model=model, text=joined))
    except Exception:
        return len(joined) // 4


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


def _enforce_size_guardrail(state: ConversationState, max_tokens: int) -> None:
    """Drop the oldest messages until the hot transcript fits in the cap.

    Fires only when the agent forgets to close scenarios. Deterministic,
    no LLM call. Records one compacted_log entry so the trim is visible.
    Tokens beat bytes here because providers bill and clamp on tokens.
    """
    model = settings.litellm_model
    if _messages_token_size(state.messages, model) <= max_tokens:
        return
    dropped = 0
    while len(state.messages) > 2 and _messages_token_size(state.messages, model) > max_tokens:
        state.messages.pop(0)
        dropped += 1
    if dropped:
        state.compacted_log.append(
            f"[auto-trimmed {dropped} older messages to stay under token cap]"
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

                _enforce_size_guardrail(state, settings.conversation_hot_max_tokens)

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
    try:
        await check_rate_limit(
            scope="conversation",
            subject_id=str(user.id),
            limit=settings.rate_limit_conversation_per_hour,
            window_seconds=3600,
        )
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Conversation rate limit hit ({exc.limit}/hour). "
                f"Retry in {exc.retry_after_seconds}s."
            ),
            headers={"Retry-After": str(exc.retry_after_seconds)},
        ) from exc
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


async def _record_button_decision(user_id: str, summary: str, max_tokens: int) -> None:
    """Append a synthetic assistant note so the LLM sees the button outcome.

    Without this, the next user turn would not know that a destructive
    or expensive action just ran via an inline button, and might offer
    to do it again. The note is short and clearly marks the source as
    a button decision, not a chat message.
    """
    state = await _load_state(user_id)
    state.messages.append({"role": "assistant", "content": f"[inline-button] {summary}"})
    _enforce_size_guardrail(state, max_tokens)
    await _save_state(state)


_DISCOVERY_QUEUED_MSG: dict[str, str] = {
    "ru": "Поиск источников запущен — вы получите отдельное сообщение, когда он завершится.",
    "en": "Source discovery is running — you'll get a follow-up message when it's done.",
}


def _user_facing_result(tool_name: str, raw_result: str, language: str) -> str:
    if tool_name == "trigger_source_discovery" and raw_result == "discovery_queued":
        return _DISCOVERY_QUEUED_MSG.get(language, _DISCOVERY_QUEUED_MSG["en"])
    return raw_result


@router.post("/confirm", response_model=ConfirmationDecisionResponse)
async def confirm_action(
    payload: ConfirmationDecisionRequest,
    user: User = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
) -> ConfirmationDecisionResponse:
    """Redeem (or cancel) a pending tool confirmation by nonce.

    Looks up the pending record under ``payload.nonce`` and verifies
    it belongs to the authenticated user. On ``cancel``, drops the
    record and returns. On ``confirm``, dispatches straight to the
    target tool with the nonce as ``confirmation_token`` -- the tool's
    own gate consumes the nonce atomically and proceeds.

    Records the decision in the conversation transcript so the LLM's
    next turn knows the action ran (and does not offer to redo it).
    """
    pending = await confirmations.peek(payload.nonce, str(user.id))
    if pending is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Pending confirmation not found or expired.",
        )

    if payload.decision == "cancel":
        await confirmations.cancel(payload.nonce, str(user.id))
        await _record_button_decision(
            str(user.id),
            f"User cancelled pending action: {pending.description}",
            settings.conversation_hot_max_tokens,
        )
        return ConfirmationDecisionResponse(status="cancelled", action=pending.tool_name)

    shared_state: dict = {
        "status": "in_progress",
        "status_queue": None,
        "display_language": user.language or "en",
    }
    tools_by_name = build_tools_by_name(
        user=user,
        db_session=session,
        scoped_factory=async_session_factory,
        shared_state=shared_state,
    )
    tool = tools_by_name.get(pending.tool_name)
    if tool is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unknown tool {pending.tool_name!r}",
        )

    with user_tag(user.id):
        try:
            result = await tool(**pending.args, confirmation_token=payload.nonce)
        except Exception:
            logger.exception(
                "Confirmation executor crashed for tool=%s user=%s",
                pending.tool_name,
                user.id,
            )
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Confirmation executor failed; the action may not have run.",
            ) from None

    await _record_button_decision(
        str(user.id),
        f"User confirmed via button; {pending.tool_name} -> {str(result)[:300]}",
        settings.conversation_hot_max_tokens,
    )
    return ConfirmationDecisionResponse(
        status="executed",
        action=pending.tool_name,
        result=_user_facing_result(pending.tool_name, str(result), user.language or "en"),
    )
