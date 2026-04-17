"""Flush idle conversations into User.conversation_summary.

Periodic Celery task. Scans Redis for conversations whose TTL is nearing
expiry, runs one LLM call per conversation to produce an updated persistent
summary, writes to User.conversation_summary, and deletes the Redis key.

This implements the OpenClaw-style memory-flush / dreaming pattern:
transcripts are transient; durable insights about the user survive via the
summary so the agent can pick up where it left off in the next session.
"""

import asyncio
import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from news_service.core.config import get_settings
from news_service.core.llm import chat_completion
from news_service.core.redis import get_redis_client
from news_service.db.session import get_task_session
from news_service.models.user import User
from news_service.schemas.conversation import ConversationState
from news_service.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)
settings = get_settings()

_FLUSH_WHEN_TTL_BELOW_SECONDS = 600
_SUMMARIZE_LAST_N_MESSAGES = 40
_SUMMARY_BYTE_LIMIT = 2048

_SYSTEM_PROMPT = (
    "You maintain a compact running summary of what we already know about this user. "
    "Keep it under four sentences total. Focus on durable facts: topics they follow, "
    "strong preferences, schedule constraints, language quirks, personal context "
    "(travel, location, how they talk about content). Skip transient moods and "
    "one-off tasks. Preserve existing summary content unless it is clearly "
    "outdated by new information. Return only the summary text."
)


@celery_app.task(name="news_service.tasks.boundary_flush.flush_idle_conversations")
def flush_idle_conversations() -> dict[str, int]:
    """Scan Redis, flush near-expiry conversations into user summaries."""
    return asyncio.run(_flush_async())


async def _flush_async() -> dict[str, int]:
    redis = get_redis_client()
    flushed = 0
    errors = 0
    try:
        async for key in redis.scan_iter(match="conv:*", count=200):
            key_str = key if isinstance(key, str) else key.decode("utf-8")
            try:
                ttl = await redis.ttl(key_str)
            except Exception:
                logger.exception("boundary flush: TTL lookup failed for %s", key_str)
                errors += 1
                continue
            if ttl < 0 or ttl >= _FLUSH_WHEN_TTL_BELOW_SECONDS:
                continue
            try:
                await _flush_one(redis, key_str)
                flushed += 1
            except Exception:
                logger.exception("boundary flush failed for %s", key_str)
                errors += 1
    finally:
        await redis.aclose()
    logger.info("boundary flush complete: flushed=%d errors=%d", flushed, errors)
    return {"flushed": flushed, "errors": errors}


async def _flush_one(redis_client: object, key: str) -> None:
    raw = await redis_client.get(key)  # type: ignore[attr-defined]
    if raw is None:
        return
    try:
        state = ConversationState.model_validate_json(raw)
    except Exception:
        logger.exception("boundary flush: could not parse state for %s", key)
        await redis_client.delete(key)  # type: ignore[attr-defined]
        return

    async with get_task_session() as session:
        user = await _load_user(session, state.user_id)
        if user is None:
            await redis_client.delete(key)  # type: ignore[attr-defined]
            return
        new_summary = await _summarize(user.conversation_summary or "", state.messages)
        if new_summary and new_summary.strip() != (user.conversation_summary or "").strip():
            user.conversation_summary = new_summary[:_SUMMARY_BYTE_LIMIT]
            await session.commit()

    await redis_client.delete(key)  # type: ignore[attr-defined]


async def _load_user(session: AsyncSession, user_id: str) -> User | None:
    try:
        uid = uuid.UUID(user_id)
    except ValueError:
        return None
    result = await session.execute(select(User).where(User.id == uid))
    return result.scalar_one_or_none()


async def _summarize(existing: str, messages: list[dict]) -> str:
    trimmed = messages[-_SUMMARIZE_LAST_N_MESSAGES:]
    transcript_lines: list[str] = []
    for msg in trimmed:
        role = msg.get("role", "?")
        content = (msg.get("content") or "").strip()
        if content:
            transcript_lines.append(f"{role}: {content}")
    transcript = "\n".join(transcript_lines)

    if not transcript:
        return existing

    user_content = (
        f"Existing summary:\n{existing or '(none)'}\n\n"
        f"Recent transcript:\n{transcript}\n\n"
        "Write the updated summary."
    )
    try:
        response = await chat_completion(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.2,
        )
    except Exception:
        logger.exception("boundary flush: summarization call failed")
        return existing

    try:
        return (response.choices[0].message.content or "").strip()
    except Exception:
        logger.exception("boundary flush: malformed summarization response")
        return existing
