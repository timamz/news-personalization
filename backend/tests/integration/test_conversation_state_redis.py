"""Integration coverage for the real Redis round-trip of ConversationState.

Every other test for the conversation persistence layer mocks the Redis
client, which means a bug in the Pydantic model -> JSON -> Pydantic path
(encoding of Cyrillic content, compacted_log entries written by
``close_scenario``, or any field added later) would pass unit tests yet
corrupt production. This module writes real bytes to Redis via the
production ``_save_state`` helper, disposes of the client, opens a fresh
client instance, and reads the state back via the production
``_load_state`` helper so the full serialization contract is exercised.
"""

import uuid

import pytest
import redis.asyncio as aioredis

import news_service.core.redis as redis_module
from news_service.api.routes_conversations import (
    REDIS_KEY_PREFIX,
    _load_state,
    _save_state,
)
from news_service.core.config import get_settings
from news_service.core.redis import close_redis_client, get_redis_client
from news_service.schemas.conversation import ConversationState


@pytest.mark.asyncio(loop_scope="session")
async def test_conversation_state_round_trips_through_real_redis() -> None:
    """A saved ConversationState reloads byte-identical through a fresh client."""
    settings = get_settings()
    probe = aioredis.from_url(settings.redis_url, decode_responses=True)
    try:
        try:
            await probe.ping()
        except Exception:
            pytest.skip("REDIS_URL not reachable")
        user_id = str(uuid.uuid4())
        key = f"{REDIS_KEY_PREFIX}{user_id}"
        await probe.delete(key)
    finally:
        await probe.aclose()

    original = ConversationState(
        user_id=user_id,
        messages=[
            {"role": "user", "content": "Подпиши меня на новости про ИИ 🤖"},
            {
                "role": "assistant",
                "content": "Готово — подписка создана, дайджест каждое утро.",
            },
            {"role": "user", "content": "Спасибо! Добавь ещё источник про квантовые вычисления."},
        ],
        compacted_log=[
            "scenario=create_subscription summary=подписка ИИ создана в 09:15",
            "[auto-trimmed 4 older messages to stay under size cap]",
        ],
        user_language="ru",
    )

    redis_module._client = None
    try:
        client_a = get_redis_client()
        await _save_state(original)
        assert client_a is get_redis_client(), (
            "save client was silently swapped mid-operation; cannot trust the bytes on Redis"
        )
        await close_redis_client()
        assert redis_module._client is None, (
            "close_redis_client did not null the module-global; the next get would reuse client A"
        )

        client_b = get_redis_client()
        assert client_b is not client_a, (
            "the load path is reusing the save-side client instance; "
            "real-Redis round-trip is not being verified"
        )
        reloaded = await _load_state(user_id)
    finally:
        try:
            cleanup_client = aioredis.from_url(settings.redis_url, decode_responses=True)
            try:
                await cleanup_client.delete(f"{REDIS_KEY_PREFIX}{user_id}")
            finally:
                await cleanup_client.aclose()
        finally:
            await close_redis_client()

    assert reloaded.model_dump() == original.model_dump(), (
        "ConversationState did not survive real Redis serialization intact: "
        f"messages/compacted_log/scenario fields diverged between save and load "
        f"(before={original.model_dump()}, after={reloaded.model_dump()})"
    )
