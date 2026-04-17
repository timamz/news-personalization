"""Tests for the single-endpoint, user-keyed conversation API."""

import json
import logging
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.schemas.conversation import (
    AgentTurnOutput,
    ConversationState,
    ConversationTurnRequest,
)

logging.disable(logging.CRITICAL)

MODULE = "news_service.api.routes_conversations"


def _mock_redis(stored: dict[str, str] | None = None):
    """In-memory fake of the async Redis client."""
    storage = dict(stored or {})

    async def mock_get(key):
        return storage.get(key)

    async def mock_set(key, value, ex=None):
        del ex
        storage[key] = value

    async def mock_delete(key):
        storage.pop(key, None)

    async def mock_aclose():
        pass

    redis_fake = MagicMock()
    redis_fake.get = AsyncMock(side_effect=mock_get)
    redis_fake.set = AsyncMock(side_effect=mock_set)
    redis_fake.delete = AsyncMock(side_effect=mock_delete)
    redis_fake.aclose = AsyncMock(side_effect=mock_aclose)
    redis_fake._storage = storage
    return redis_fake


def _mock_user(user_id: str | None = None) -> MagicMock:
    mock = MagicMock()
    mock.id = user_id or str(uuid.uuid4())
    mock.conversation_summary = ""
    mock.timezone = "Europe/Moscow"
    mock.language = "en"
    return mock


async def _mock_streaming_turn(
    output: AgentTurnOutput,
    *,
    scenario_close_summary: str | None = None,
):
    yield {
        "event": "done",
        "output": output.model_dump(),
        "new_messages": [{"role": "assistant", "content": output.message}],
        "shared_state": {"scenario_close_summary": scenario_close_summary},
    }


async def _collect_streaming_response(response) -> list[dict]:
    chunks: list[bytes] = []
    async for chunk in response.body_iterator:
        if isinstance(chunk, str):
            chunks.append(chunk.encode())
        else:
            chunks.append(chunk)
    raw = b"".join(chunks).decode()
    return [json.loads(line) for line in raw.strip().splitlines() if line.strip()]


@pytest.mark.asyncio
async def test_send_message_stream_produces_done_event(mocker) -> None:
    agent_output = AgentTurnOutput(message=f"ответ {uuid.uuid4().hex[:6]}")
    mocker.patch(
        f"{MODULE}.run_conversation_turn_streaming",
        return_value=_mock_streaming_turn(agent_output),
    )
    mocker.patch(f"{MODULE}.get_redis_client", return_value=_mock_redis())

    from news_service.api.routes_conversations import send_conversation_message_stream

    user = _mock_user()
    request = ConversationTurnRequest(message="Новости ИИ", user_language="ru")
    response = await send_conversation_message_stream(request, user=user, session=AsyncMock())
    events = await _collect_streaming_response(response)
    done = [e for e in events if e.get("event") == "done"]
    assert len(done) == 1, "stream did not produce exactly one done event"


@pytest.mark.asyncio
async def test_send_message_stream_done_event_has_agent_message(mocker) -> None:
    text = f"response {uuid.uuid4().hex[:6]}"
    agent_output = AgentTurnOutput(message=text)
    mocker.patch(
        f"{MODULE}.run_conversation_turn_streaming",
        return_value=_mock_streaming_turn(agent_output),
    )
    mocker.patch(f"{MODULE}.get_redis_client", return_value=_mock_redis())

    from news_service.api.routes_conversations import send_conversation_message_stream

    request = ConversationTurnRequest(message="topic", user_language="en")
    response = await send_conversation_message_stream(
        request, user=_mock_user(), session=AsyncMock()
    )
    events = await _collect_streaming_response(response)
    done = next(e for e in events if e.get("event") == "done")
    assert done["agent_message"] == text, "done event did not contain the expected agent message"


@pytest.mark.asyncio
async def test_send_message_stream_persists_state_keyed_by_user(mocker) -> None:
    agent_output = AgentTurnOutput(message="ok")
    mocker.patch(
        f"{MODULE}.run_conversation_turn_streaming",
        return_value=_mock_streaming_turn(agent_output),
    )
    redis_fake = _mock_redis()
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    from news_service.api.routes_conversations import send_conversation_message_stream

    user = _mock_user()
    request = ConversationTurnRequest(message="hi", user_language="en")
    response = await send_conversation_message_stream(request, user=user, session=AsyncMock())

    await _collect_streaming_response(response)
    assert f"conv:user:{user.id}" in redis_fake._storage, (
        "conversation state was not saved under the user-scoped redis key"
    )


@pytest.mark.asyncio
async def test_send_message_stream_reuses_existing_thread_for_same_user(mocker) -> None:
    user = _mock_user()
    existing = ConversationState(
        user_id=str(user.id),
        messages=[
            {"role": "user", "content": "earlier question"},
            {"role": "assistant", "content": "earlier reply"},
        ],
        user_language="en",
    )
    redis_fake = _mock_redis({f"conv:user:{user.id}": existing.model_dump_json()})
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    captured: dict = {}

    async def _fake_turn(messages, **_kwargs):
        captured["messages"] = list(messages)
        yield {
            "event": "done",
            "output": AgentTurnOutput(message="reply").model_dump(),
            "new_messages": [{"role": "assistant", "content": "reply"}],
            "shared_state": {"scenario_close_summary": None},
        }

    mocker.patch(f"{MODULE}.run_conversation_turn_streaming", new=_fake_turn)

    from news_service.api.routes_conversations import send_conversation_message_stream

    request = ConversationTurnRequest(message="follow-up")
    response = await send_conversation_message_stream(request, user=user, session=AsyncMock())
    await _collect_streaming_response(response)

    roles = [m["role"] for m in captured["messages"]]
    assert roles == ["user", "assistant", "user"], (
        f"new turn did not include prior messages plus the new user turn; roles={roles}"
    )


@pytest.mark.asyncio
async def test_scenario_close_compacts_prior_messages_into_log(mocker) -> None:
    user = _mock_user()
    existing = ConversationState(
        user_id=str(user.id),
        messages=[
            {"role": "user", "content": "set up AI digest"},
            {"role": "assistant", "content": "what schedule?"},
            {"role": "user", "content": "every morning 8am"},
        ],
    )
    redis_fake = _mock_redis({f"conv:user:{user.id}": existing.model_dump_json()})
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    summary = f"created AI digest daily 8am {uuid.uuid4().hex[:6]}"
    mocker.patch(
        f"{MODULE}.run_conversation_turn_streaming",
        return_value=_mock_streaming_turn(
            AgentTurnOutput(message="Done!"), scenario_close_summary=summary
        ),
    )

    from news_service.api.routes_conversations import send_conversation_message_stream

    request = ConversationTurnRequest(message="go ahead")
    response = await send_conversation_message_stream(request, user=user, session=AsyncMock())
    await _collect_streaming_response(response)

    saved = ConversationState.model_validate_json(redis_fake._storage[f"conv:user:{user.id}"])
    assert summary in saved.compacted_log, "scenario summary was not appended to compacted_log"
    assert len(saved.messages) <= 2, (
        f"hot transcript was not trimmed after close_scenario; len={len(saved.messages)}"
    )


@pytest.mark.asyncio
async def test_reset_conversation_deletes_user_thread(mocker) -> None:
    user = _mock_user()
    redis_fake = _mock_redis(
        {f"conv:user:{user.id}": ConversationState(user_id=str(user.id)).model_dump_json()}
    )
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    from news_service.api.routes_conversations import reset_conversation

    await reset_conversation(user=user)
    assert f"conv:user:{user.id}" not in redis_fake._storage, (
        "reset_conversation did not remove the user's stored thread"
    )


@pytest.mark.asyncio
async def test_size_guardrail_trims_hot_transcript_when_over_cap(mocker) -> None:
    user = _mock_user()
    big_messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 2000} for i in range(20)
    ]
    existing = ConversationState(user_id=str(user.id), messages=big_messages)
    redis_fake = _mock_redis({f"conv:user:{user.id}": existing.model_dump_json()})
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    mocker.patch(
        f"{MODULE}.run_conversation_turn_streaming",
        return_value=_mock_streaming_turn(AgentTurnOutput(message="ok")),
    )

    from news_service.api.routes_conversations import send_conversation_message_stream

    request = ConversationTurnRequest(message="short follow-up")
    response = await send_conversation_message_stream(request, user=user, session=AsyncMock())
    await _collect_streaming_response(response)

    saved = ConversationState.model_validate_json(redis_fake._storage[f"conv:user:{user.id}"])
    assert len(saved.messages) < 21, (
        "size guardrail did not drop any messages despite exceeding the cap"
    )
    assert any("auto-trimmed" in line for line in saved.compacted_log), (
        "size guardrail did not record an auto-trim entry in compacted_log"
    )
