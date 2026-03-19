"""Tests for conversation-based subscription setup endpoints."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from news_service.schemas.conversation import (
    AgentTurnOutput,
    ConversationState,
    FinalizedSubscriptionConfig,
)

MODULE = "news_service.api.routes_conversations"


def _mock_redis(stored: dict[str, str] | None = None):
    """Create a mock Redis client with optional pre-stored data."""
    storage = dict(stored or {})

    async def mock_get(key):
        return storage.get(key)

    async def mock_set(key, value, ex=None):
        storage[key] = value

    async def mock_delete(key):
        storage.pop(key, None)

    async def mock_aclose():
        pass

    redis_mock = MagicMock()
    redis_mock.get = AsyncMock(side_effect=mock_get)
    redis_mock.set = AsyncMock(side_effect=mock_set)
    redis_mock.delete = AsyncMock(side_effect=mock_delete)
    redis_mock.aclose = AsyncMock(side_effect=mock_aclose)
    redis_mock._storage = storage
    return redis_mock


def _make_state(user_id: str = "user-123", **kwargs) -> ConversationState:
    return ConversationState(
        user_id=user_id,
        messages=kwargs.get("messages", [{"role": "user", "content": "AI news"}]),
        status=kwargs.get("status", "in_progress"),
        finalized_config=kwargs.get("finalized_config"),
        user_language=kwargs.get("user_language", "en"),
        user_timezone=kwargs.get("user_timezone"),
    )


@pytest.mark.asyncio
async def test_start_conversation(mocker):
    agent_output = AgentTurnOutput(
        message="What schedule do you prefer?",
        status="in_progress",
    )
    mocker.patch(
        f"{MODULE}.run_conversation_turn",
        new=AsyncMock(
            return_value=(
                agent_output,
                [{"role": "assistant", "content": agent_output.message}],
            )
        ),
    )

    redis_mock = _mock_redis()
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_mock)

    mock_user = MagicMock()
    mock_user.id = "user-123"

    from news_service.api.routes_conversations import start_conversation
    from news_service.schemas.conversation import ConversationStartRequest

    request = ConversationStartRequest(message="AI news", user_language="en")

    with patch(f"{MODULE}.get_current_user", return_value=mock_user):
        response = await start_conversation(request, user=mock_user)

    assert response.status == "in_progress"
    assert response.agent_message == "What schedule do you prefer?"
    assert response.conversation_id
    assert response.finalized_config is None

    # Verify state was saved
    assert len(redis_mock._storage) == 1


@pytest.mark.asyncio
async def test_continue_conversation(mocker):
    state = _make_state(
        messages=[
            {"role": "user", "content": "AI news"},
            {"role": "assistant", "content": "What schedule?"},
        ],
    )
    conv_id = "abc123"
    redis_mock = _mock_redis({f"conv:{conv_id}": state.model_dump_json()})
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_mock)

    agent_output = AgentTurnOutput(
        message="Your subscription is ready!",
        status="ready",
        finalized_config=FinalizedSubscriptionConfig(
            prompt_summary="AI news digest",
            short_label="AI News",
            digest_language="en",
        ),
    )
    mocker.patch(
        f"{MODULE}.run_conversation_turn",
        new=AsyncMock(
            return_value=(
                agent_output,
                [{"role": "assistant", "content": agent_output.message}],
            )
        ),
    )

    mock_user = MagicMock()
    mock_user.id = "user-123"

    from news_service.api.routes_conversations import continue_conversation
    from news_service.schemas.conversation import ConversationMessageRequest

    request = ConversationMessageRequest(message="every morning")

    response = await continue_conversation(conv_id, request, user=mock_user)

    assert response.status == "ready"
    assert response.finalized_config is not None
    assert response.finalized_config.prompt_summary == "AI news digest"


@pytest.mark.asyncio
async def test_continue_conversation_not_found(mocker):
    redis_mock = _mock_redis()
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_mock)

    mock_user = MagicMock()
    mock_user.id = "user-123"

    from fastapi import HTTPException

    from news_service.api.routes_conversations import continue_conversation
    from news_service.schemas.conversation import ConversationMessageRequest

    request = ConversationMessageRequest(message="hello")

    with pytest.raises(HTTPException) as exc_info:
        await continue_conversation("nonexistent", request, user=mock_user)
    assert exc_info.value.status_code == 404


@pytest.mark.asyncio
async def test_continue_conversation_wrong_user(mocker):
    state = _make_state(user_id="other-user")
    conv_id = "abc123"
    redis_mock = _mock_redis({f"conv:{conv_id}": state.model_dump_json()})
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_mock)

    mock_user = MagicMock()
    mock_user.id = "user-123"

    from fastapi import HTTPException

    from news_service.api.routes_conversations import continue_conversation
    from news_service.schemas.conversation import ConversationMessageRequest

    request = ConversationMessageRequest(message="hello")

    with pytest.raises(HTTPException) as exc_info:
        await continue_conversation(conv_id, request, user=mock_user)
    assert exc_info.value.status_code == 403


@pytest.mark.asyncio
async def test_continue_conversation_already_finalized(mocker):
    state = _make_state(status="ready")
    conv_id = "abc123"
    redis_mock = _mock_redis({f"conv:{conv_id}": state.model_dump_json()})
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_mock)

    mock_user = MagicMock()
    mock_user.id = "user-123"

    from fastapi import HTTPException

    from news_service.api.routes_conversations import continue_conversation
    from news_service.schemas.conversation import ConversationMessageRequest

    request = ConversationMessageRequest(message="hello")

    with pytest.raises(HTTPException) as exc_info:
        await continue_conversation(conv_id, request, user=mock_user)
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_cancel_conversation(mocker):
    state = _make_state()
    conv_id = "abc123"
    redis_mock = _mock_redis({f"conv:{conv_id}": state.model_dump_json()})
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_mock)

    mock_user = MagicMock()
    mock_user.id = "user-123"

    from news_service.api.routes_conversations import cancel_conversation

    await cancel_conversation(conv_id, user=mock_user)

    # Verify state was deleted
    assert f"conv:{conv_id}" not in redis_mock._storage


@pytest.mark.asyncio
async def test_cancel_conversation_not_found(mocker):
    redis_mock = _mock_redis()
    mocker.patch(f"{MODULE}.get_redis_client", return_value=redis_mock)

    mock_user = MagicMock()
    mock_user.id = "user-123"

    from fastapi import HTTPException

    from news_service.api.routes_conversations import cancel_conversation

    with pytest.raises(HTTPException) as exc_info:
        await cancel_conversation("nonexistent", user=mock_user)
    assert exc_info.value.status_code == 404
