"""Tests for conversation REST endpoints (start / continue / cancel)."""

import json
import logging
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from news_service.schemas.conversation import AgentTurnOutput, ConversationState

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


def _state(user_id: str, **kwargs) -> ConversationState:
    return ConversationState(
        user_id=user_id,
        messages=kwargs.get("messages", [{"role": "user", "content": "Новости ИИ"}]),
        user_language=kwargs.get("user_language", "ru"),
    )


async def _mock_streaming_turn(output: AgentTurnOutput):
    yield {
        "event": "done",
        "output": output.model_dump(),
        "new_messages": [{"role": "assistant", "content": output.message}],
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
async def test_start_conversation_stream_produces_done_event(mocker) -> None:
    agent_output = AgentTurnOutput(message=f"ответ {uuid.uuid4().hex[:6]}")
    mocker.patch(
        f"{MODULE}.run_conversation_turn_streaming",
        return_value=_mock_streaming_turn(agent_output),
    )
    mocker.patch(f"{MODULE}.get_redis_client", return_value=_mock_redis())

    mock_session = AsyncMock()
    user = _mock_user()

    from news_service.api.routes_conversations import start_conversation_stream
    from news_service.schemas.conversation import ConversationStartRequest

    request = ConversationStartRequest(message="Новости ИИ", user_language="ru")
    with patch(f"{MODULE}.get_current_user", return_value=user):
        response = await start_conversation_stream(request, user=user, session=mock_session)

    events = await _collect_streaming_response(response)
    done = [e for e in events if e.get("event") == "done"]
    assert len(done) == 1, "stream did not produce exactly one done event"


@pytest.mark.asyncio
async def test_start_conversation_stream_done_event_has_agent_message(mocker) -> None:
    text = f"response {uuid.uuid4().hex[:6]}"
    agent_output = AgentTurnOutput(message=text)
    mocker.patch(
        f"{MODULE}.run_conversation_turn_streaming",
        return_value=_mock_streaming_turn(agent_output),
    )
    mocker.patch(f"{MODULE}.get_redis_client", return_value=_mock_redis())

    mock_session = AsyncMock()
    user = _mock_user()

    from news_service.api.routes_conversations import start_conversation_stream
    from news_service.schemas.conversation import ConversationStartRequest

    request = ConversationStartRequest(message="topic", user_language="en")
    with patch(f"{MODULE}.get_current_user", return_value=user):
        response = await start_conversation_stream(request, user=user, session=mock_session)

    events = await _collect_streaming_response(response)
    done = next(e for e in events if e.get("event") == "done")
    assert done["agent_message"] == text, (
        "done event did not contain the expected agent message"
    )


@pytest.mark.asyncio
async def test_start_conversation_stream_persists_state_to_redis(mocker) -> None:
    agent_output = AgentTurnOutput(message="ok")
    mocker.patch(
        f"{MODULE}.run_conversation_turn_streaming",
        return_value=_mock_streaming_turn(agent_output),
    )
    redis_fake = _mock_redis()
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    mock_session = AsyncMock()
    user = _mock_user()

    from news_service.api.routes_conversations import start_conversation_stream
    from news_service.schemas.conversation import ConversationStartRequest

    request = ConversationStartRequest(message="hi", user_language="en")
    with patch(f"{MODULE}.get_current_user", return_value=user):
        response = await start_conversation_stream(request, user=user, session=mock_session)

    await _collect_streaming_response(response)
    assert len(redis_fake._storage) == 1, (
        "start_conversation_stream did not persist state to redis"
    )


@pytest.mark.asyncio
async def test_continue_conversation_stream_yields_done_event(mocker) -> None:
    user_id = str(uuid.uuid4())
    state = _state(user_id)
    conv_id = uuid.uuid4().hex[:12]
    redis_fake = _mock_redis({f"conv:{conv_id}": state.model_dump_json()})
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    agent_output = AgentTurnOutput(message=f"reply {uuid.uuid4().hex[:6]}")
    mocker.patch(
        f"{MODULE}.run_conversation_turn_streaming",
        return_value=_mock_streaming_turn(agent_output),
    )

    mock_session = AsyncMock()
    user = _mock_user(user_id=user_id)

    from news_service.api.routes_conversations import continue_conversation_stream
    from news_service.schemas.conversation import ConversationMessageRequest

    request = ConversationMessageRequest(message="another line")
    response = await continue_conversation_stream(
        conv_id, request, user=user, session=mock_session
    )
    events = await _collect_streaming_response(response)
    done = [e for e in events if e.get("event") == "done"]
    assert len(done) == 1, "continue stream did not yield a done event"


@pytest.mark.asyncio
async def test_continue_conversation_stream_raises_404_when_not_found(mocker) -> None:
    from fastapi import HTTPException

    mocker.patch(f"{MODULE}.get_redis_client", return_value=_mock_redis())

    from news_service.api.routes_conversations import continue_conversation_stream
    from news_service.schemas.conversation import ConversationMessageRequest

    request = ConversationMessageRequest(message="hi")
    user = _mock_user()
    with pytest.raises(HTTPException) as excinfo:
        await continue_conversation_stream(
            "missing-conv",
            request,
            user=user,
            session=AsyncMock(),
        )
    assert excinfo.value.status_code == 404, (
        f"expected 404 for missing conversation, got {excinfo.value.status_code}"
    )


@pytest.mark.asyncio
async def test_continue_conversation_stream_raises_403_for_wrong_user(mocker) -> None:
    from fastapi import HTTPException

    owner_id = str(uuid.uuid4())
    state = _state(owner_id)
    conv_id = uuid.uuid4().hex[:12]
    redis_fake = _mock_redis({f"conv:{conv_id}": state.model_dump_json()})
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    from news_service.api.routes_conversations import continue_conversation_stream
    from news_service.schemas.conversation import ConversationMessageRequest

    request = ConversationMessageRequest(message="hello")
    intruder = _mock_user(user_id=str(uuid.uuid4()))
    with pytest.raises(HTTPException) as excinfo:
        await continue_conversation_stream(
            conv_id, request, user=intruder, session=AsyncMock()
        )
    assert excinfo.value.status_code == 403, (
        f"expected 403 for cross-user access, got {excinfo.value.status_code}"
    )


@pytest.mark.asyncio
async def test_cancel_conversation_deletes_redis_key(mocker) -> None:
    user_id = str(uuid.uuid4())
    state = _state(user_id)
    conv_id = uuid.uuid4().hex[:12]
    redis_fake = _mock_redis({f"conv:{conv_id}": state.model_dump_json()})
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    from news_service.api.routes_conversations import cancel_conversation

    user = _mock_user(user_id=user_id)
    await cancel_conversation(conv_id, user=user)
    assert f"conv:{conv_id}" not in redis_fake._storage, (
        "cancel_conversation did not remove the redis entry"
    )


@pytest.mark.asyncio
async def test_cancel_conversation_raises_404_when_not_found(mocker) -> None:
    from fastapi import HTTPException

    mocker.patch(f"{MODULE}.get_redis_client", return_value=_mock_redis())

    from news_service.api.routes_conversations import cancel_conversation

    user = _mock_user()
    with pytest.raises(HTTPException) as excinfo:
        await cancel_conversation("no-such-conv", user=user)
    assert excinfo.value.status_code == 404, (
        f"expected 404 for missing conversation, got {excinfo.value.status_code}"
    )
