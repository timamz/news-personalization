import json
import logging
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from news_service.schemas.conversation import (
    AgentTurnOutput,
    ConversationState,
    FinalizedSubscriptionConfig,
)

logging.disable(logging.CRITICAL)

MODULE = "news_service.api.routes_conversations"


def _mock_redis(stored: dict[str, str] | None = None):
    """Fake Redis client with in-memory storage."""
    storage = dict(stored or {})

    async def mock_get(key):
        return storage.get(key)

    async def mock_set(key, value, ex=None):
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


def _make_state(user_id: str | None = None, **kwargs) -> ConversationState:
    return ConversationState(
        user_id=user_id or str(uuid.uuid4()),
        messages=kwargs.get("messages", [{"role": "user", "content": "Новости ИИ"}]),
        status=kwargs.get("status", "in_progress"),
        finalized_config=kwargs.get("finalized_config"),
        user_language=kwargs.get("user_language", "ru"),
        user_timezone=kwargs.get("user_timezone"),
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
async def test_start_conversation_stream_returns_done_event(mocker) -> None:
    agent_output = AgentTurnOutput(
        message="Какое расписание вы предпочитаете?",
        status="in_progress",
    )
    mocker.patch(
        f"{MODULE}.run_conversation_turn_streaming",
        return_value=_mock_streaming_turn(agent_output),
    )

    redis_fake = _mock_redis()
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    mock_user = MagicMock()
    mock_user.id = str(uuid.uuid4())

    from news_service.api.routes_conversations import start_conversation_stream
    from news_service.schemas.conversation import ConversationStartRequest

    request = ConversationStartRequest(message="Новости ИИ", user_language="ru")

    with patch(f"{MODULE}.get_current_user", return_value=mock_user):
        response = await start_conversation_stream(request, user=mock_user)

    events = await _collect_streaming_response(response)
    done_events = [e for e in events if e.get("event") == "done"]
    assert len(done_events) == 1, "stream did not produce exactly one done event"


@pytest.mark.asyncio
async def test_start_conversation_stream_done_event_has_in_progress_status(mocker) -> None:
    agent_output = AgentTurnOutput(
        message="Какое расписание вы предпочитаете?",
        status="in_progress",
    )
    mocker.patch(
        f"{MODULE}.run_conversation_turn_streaming",
        return_value=_mock_streaming_turn(agent_output),
    )

    redis_fake = _mock_redis()
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    mock_user = MagicMock()
    mock_user.id = str(uuid.uuid4())

    from news_service.api.routes_conversations import start_conversation_stream
    from news_service.schemas.conversation import ConversationStartRequest

    request = ConversationStartRequest(message="Новости ИИ", user_language="ru")

    with patch(f"{MODULE}.get_current_user", return_value=mock_user):
        response = await start_conversation_stream(request, user=mock_user)

    events = await _collect_streaming_response(response)
    done = [e for e in events if e.get("event") == "done"][0]
    assert done["status"] == "in_progress", "done event did not have in_progress status"


@pytest.mark.asyncio
async def test_start_conversation_stream_done_event_has_agent_message(mocker) -> None:
    message_text = f"Какое расписание? {uuid.uuid4().hex[:6]}"
    agent_output = AgentTurnOutput(
        message=message_text,
        status="in_progress",
    )
    mocker.patch(
        f"{MODULE}.run_conversation_turn_streaming",
        return_value=_mock_streaming_turn(agent_output),
    )

    redis_fake = _mock_redis()
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    mock_user = MagicMock()
    mock_user.id = str(uuid.uuid4())

    from news_service.api.routes_conversations import start_conversation_stream
    from news_service.schemas.conversation import ConversationStartRequest

    request = ConversationStartRequest(message="Новости ИИ", user_language="ru")

    with patch(f"{MODULE}.get_current_user", return_value=mock_user):
        response = await start_conversation_stream(request, user=mock_user)

    events = await _collect_streaming_response(response)
    done = [e for e in events if e.get("event") == "done"][0]
    assert done["agent_message"] == message_text, (
        "done event did not contain expected agent message"
    )


@pytest.mark.asyncio
async def test_start_conversation_stream_saves_state_to_redis(mocker) -> None:
    agent_output = AgentTurnOutput(
        message="Какое расписание?",
        status="in_progress",
    )
    mocker.patch(
        f"{MODULE}.run_conversation_turn_streaming",
        return_value=_mock_streaming_turn(agent_output),
    )

    redis_fake = _mock_redis()
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    mock_user = MagicMock()
    mock_user.id = str(uuid.uuid4())

    from news_service.api.routes_conversations import start_conversation_stream
    from news_service.schemas.conversation import ConversationStartRequest

    request = ConversationStartRequest(message="Новости ИИ", user_language="ru")

    with patch(f"{MODULE}.get_current_user", return_value=mock_user):
        response = await start_conversation_stream(request, user=mock_user)

    await _collect_streaming_response(response)
    assert len(redis_fake._storage) == 1, "stream did not save conversation state to redis"


@pytest.mark.asyncio
async def test_continue_conversation_stream_returns_done_with_ready_status(mocker) -> None:
    user_id = str(uuid.uuid4())
    state = _make_state(
        user_id=user_id,
        messages=[
            {"role": "user", "content": "Новости ИИ"},
            {"role": "assistant", "content": "Какое расписание?"},
        ],
    )
    conv_id = uuid.uuid4().hex[:12]
    redis_fake = _mock_redis({f"conv:{conv_id}": state.model_dump_json()})
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    agent_output = AgentTurnOutput(
        message="Ваша подписка готова!",
        status="ready",
        finalized_config=FinalizedSubscriptionConfig(
            digest_language="ru",
        ),
    )
    mocker.patch(
        f"{MODULE}.run_conversation_turn_streaming",
        return_value=_mock_streaming_turn(agent_output),
    )

    mock_user = MagicMock()
    mock_user.id = user_id

    from news_service.api.routes_conversations import continue_conversation_stream
    from news_service.schemas.conversation import ConversationMessageRequest

    request = ConversationMessageRequest(message="каждое утро")
    response = await continue_conversation_stream(conv_id, request, user=mock_user)

    events = await _collect_streaming_response(response)
    done = [e for e in events if e.get("event") == "done"][0]
    assert done["status"] == "ready", "continue stream did not return ready status"


@pytest.mark.asyncio
async def test_continue_conversation_stream_returns_finalized_config(mocker) -> None:
    user_id = str(uuid.uuid4())
    state = _make_state(
        user_id=user_id,
        messages=[
            {"role": "user", "content": "Новости ИИ"},
            {"role": "assistant", "content": "Какое расписание?"},
        ],
    )
    conv_id = uuid.uuid4().hex[:12]
    redis_fake = _mock_redis({f"conv:{conv_id}": state.model_dump_json()})
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    language = "ru"
    agent_output = AgentTurnOutput(
        message="Ваша подписка готова!",
        status="ready",
        finalized_config=FinalizedSubscriptionConfig(
            digest_language=language,
        ),
    )
    mocker.patch(
        f"{MODULE}.run_conversation_turn_streaming",
        return_value=_mock_streaming_turn(agent_output),
    )

    mock_user = MagicMock()
    mock_user.id = user_id

    from news_service.api.routes_conversations import continue_conversation_stream
    from news_service.schemas.conversation import ConversationMessageRequest

    request = ConversationMessageRequest(message="каждое утро")
    response = await continue_conversation_stream(conv_id, request, user=mock_user)

    events = await _collect_streaming_response(response)
    done = [e for e in events if e.get("event") == "done"][0]
    assert done["finalized_config"]["digest_language"] == language, (
        "continue stream did not return correct finalized config"
    )


@pytest.mark.asyncio
async def test_continue_conversation_stream_raises_404_when_not_found(mocker) -> None:
    redis_fake = _mock_redis()
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    mock_user = MagicMock()
    mock_user.id = str(uuid.uuid4())

    from fastapi import HTTPException

    from news_service.api.routes_conversations import continue_conversation_stream
    from news_service.schemas.conversation import ConversationMessageRequest

    request = ConversationMessageRequest(message="привет")

    with pytest.raises(HTTPException) as exc_info:
        await continue_conversation_stream(uuid.uuid4().hex, request, user=mock_user)

    assert exc_info.value.status_code == 404, (
        "continue stream did not raise 404 for missing conversation"
    )


@pytest.mark.asyncio
async def test_continue_conversation_stream_raises_403_for_wrong_user(mocker) -> None:
    other_user_id = str(uuid.uuid4())
    state = _make_state(user_id=other_user_id)
    conv_id = uuid.uuid4().hex[:12]
    redis_fake = _mock_redis({f"conv:{conv_id}": state.model_dump_json()})
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    mock_user = MagicMock()
    mock_user.id = str(uuid.uuid4())

    from fastapi import HTTPException

    from news_service.api.routes_conversations import continue_conversation_stream
    from news_service.schemas.conversation import ConversationMessageRequest

    request = ConversationMessageRequest(message="привет")

    with pytest.raises(HTTPException) as exc_info:
        await continue_conversation_stream(conv_id, request, user=mock_user)

    assert exc_info.value.status_code == 403, "continue stream did not raise 403 for wrong user"


@pytest.mark.asyncio
async def test_continue_conversation_stream_raises_409_when_already_finalized(mocker) -> None:
    user_id = str(uuid.uuid4())
    state = _make_state(user_id=user_id, status="ready")
    conv_id = uuid.uuid4().hex[:12]
    redis_fake = _mock_redis({f"conv:{conv_id}": state.model_dump_json()})
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    mock_user = MagicMock()
    mock_user.id = user_id

    from fastapi import HTTPException

    from news_service.api.routes_conversations import continue_conversation_stream
    from news_service.schemas.conversation import ConversationMessageRequest

    request = ConversationMessageRequest(message="привет")

    with pytest.raises(HTTPException) as exc_info:
        await continue_conversation_stream(conv_id, request, user=mock_user)

    assert exc_info.value.status_code == 409, (
        "continue stream did not raise 409 for already finalized conversation"
    )


@pytest.mark.asyncio
async def test_cancel_conversation_deletes_state_from_redis(mocker) -> None:
    user_id = str(uuid.uuid4())
    state = _make_state(user_id=user_id)
    conv_id = uuid.uuid4().hex[:12]
    redis_fake = _mock_redis({f"conv:{conv_id}": state.model_dump_json()})
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    mock_user = MagicMock()
    mock_user.id = user_id

    from news_service.api.routes_conversations import cancel_conversation

    await cancel_conversation(conv_id, user=mock_user)

    assert f"conv:{conv_id}" not in redis_fake._storage, (
        "cancel did not delete conversation state from redis"
    )


@pytest.mark.asyncio
async def test_cancel_conversation_raises_404_when_not_found(mocker) -> None:
    redis_fake = _mock_redis()
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_fake)

    mock_user = MagicMock()
    mock_user.id = str(uuid.uuid4())

    from fastapi import HTTPException

    from news_service.api.routes_conversations import cancel_conversation

    with pytest.raises(HTTPException) as exc_info:
        await cancel_conversation(uuid.uuid4().hex, user=mock_user)

    assert exc_info.value.status_code == 404, "cancel did not raise 404 for missing conversation"
