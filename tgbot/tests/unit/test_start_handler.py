"""Tests for the /start command and the generic text-message relay handler."""

import logging
import random
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tgbot.handlers import start

logging.disable(logging.CRITICAL)


def _make_message(telegram_id: int, text: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        from_user=SimpleNamespace(id=telegram_id),
        chat=SimpleNamespace(id=telegram_id),
        text=text,
        answer=AsyncMock(),
        bot=SimpleNamespace(send_chat_action=AsyncMock()),
    )


async def _stream_events(events: list[dict]):
    for event in events:
        yield event


@pytest.mark.asyncio
async def test_cmd_start_sends_welcome_message(mocker) -> None:
    telegram_id = random.randint(100000, 999999)
    message = _make_message(telegram_id)
    mocker.patch.object(start, "ensure_api_key", new=AsyncMock(return_value="key"))
    clear_mock = mocker.patch.object(start, "clear_conversation_id", new=AsyncMock())
    await start.cmd_start(message)
    message.answer.assert_awaited_once()
    clear_mock.assert_awaited_once_with(telegram_id)


@pytest.mark.asyncio
async def test_cmd_start_reports_error_when_registration_fails(mocker) -> None:
    telegram_id = random.randint(100000, 999999)
    message = _make_message(telegram_id)
    mocker.patch.object(start, "ensure_api_key", new=AsyncMock(side_effect=RuntimeError("boom")))
    await start.cmd_start(message)
    assert message.answer.await_args is not None, "cmd_start did not answer the user"
    spoken_text = message.answer.await_args.args[0]
    assert "wrong" in spoken_text.lower(), (
        f"cmd_start did not surface an error to the user: {spoken_text!r}"
    )


@pytest.mark.asyncio
async def test_handle_user_message_ignores_empty_text(mocker) -> None:
    telegram_id = random.randint(100000, 999999)
    message = _make_message(telegram_id, text="   ")
    ensure_mock = mocker.patch.object(start, "ensure_api_key", new=AsyncMock())
    await start.handle_user_message(message)
    ensure_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_user_message_starts_fresh_conversation_when_none_stored(
    mocker,
) -> None:
    telegram_id = random.randint(100000, 999999)
    message = _make_message(telegram_id, text="hello")
    api_key = f"key-{uuid.uuid4().hex}"
    conversation_id = f"conv-{uuid.uuid4().hex}"
    agent_text = f"pong-{uuid.uuid4().hex[:6]}"

    mocker.patch.object(start, "ensure_api_key", new=AsyncMock(return_value=api_key))
    mocker.patch.object(start, "get_conversation_id", new=AsyncMock(return_value=None))
    save_mock = mocker.patch.object(start, "save_conversation_id", new=AsyncMock())

    start_stream = mocker.patch.object(
        start.backend,
        "start_subscription_conversation_stream",
        return_value=_stream_events(
            [
                {"event": "status", "status_key": "status_thinking"},
                {
                    "event": "done",
                    "conversation_id": conversation_id,
                    "agent_message": agent_text,
                    "status": "in_progress",
                    "finalized_config": None,
                },
            ]
        ),
    )
    mocker.patch.object(
        start.backend,
        "continue_subscription_conversation_stream",
        new=AsyncMock(),
    )

    await start.handle_user_message(message)

    start_stream.assert_called_once_with(api_key, "hello")
    save_mock.assert_awaited_once_with(telegram_id, conversation_id)
    assert message.answer.await_count == 1, (
        f"handle_user_message sent {message.answer.await_count} messages, expected 1"
    )


@pytest.mark.asyncio
async def test_handle_user_message_continues_existing_conversation(mocker) -> None:
    telegram_id = random.randint(100000, 999999)
    message = _make_message(telegram_id, text="what did i say yesterday?")
    api_key = f"key-{uuid.uuid4().hex}"
    conversation_id = f"conv-{uuid.uuid4().hex}"

    mocker.patch.object(start, "ensure_api_key", new=AsyncMock(return_value=api_key))
    mocker.patch.object(start, "get_conversation_id", new=AsyncMock(return_value=conversation_id))
    mocker.patch.object(start, "save_conversation_id", new=AsyncMock())

    continue_stream = mocker.patch.object(
        start.backend,
        "continue_subscription_conversation_stream",
        return_value=_stream_events(
            [
                {
                    "event": "done",
                    "conversation_id": conversation_id,
                    "agent_message": "ok",
                    "status": "in_progress",
                    "finalized_config": None,
                }
            ]
        ),
    )
    start_stream = mocker.patch.object(
        start.backend, "start_subscription_conversation_stream", new=AsyncMock()
    )

    await start.handle_user_message(message)

    continue_stream.assert_called_once_with(api_key, conversation_id, "what did i say yesterday?")
    start_stream.assert_not_called()


@pytest.mark.asyncio
async def test_handle_user_message_clears_conversation_id_and_retries_on_conflict(
    mocker,
) -> None:
    telegram_id = random.randint(100000, 999999)
    message = _make_message(telegram_id, text="hi again")
    api_key = f"key-{uuid.uuid4().hex}"
    stale_id = f"conv-{uuid.uuid4().hex}"
    fresh_id = f"conv-{uuid.uuid4().hex}"

    mocker.patch.object(start, "ensure_api_key", new=AsyncMock(return_value=api_key))
    mocker.patch.object(start, "get_conversation_id", new=AsyncMock(return_value=stale_id))
    save_mock = mocker.patch.object(start, "save_conversation_id", new=AsyncMock())

    import httpx

    conflict = httpx.HTTPStatusError(
        "Conflict", request=AsyncMock(), response=SimpleNamespace(status_code=409)
    )

    async def _fail_continue(*_args, **_kwargs):
        raise conflict
        yield  # pragma: no cover

    mocker.patch.object(
        start.backend,
        "continue_subscription_conversation_stream",
        side_effect=_fail_continue,
    )
    start_stream = mocker.patch.object(
        start.backend,
        "start_subscription_conversation_stream",
        return_value=_stream_events(
            [
                {
                    "event": "done",
                    "conversation_id": fresh_id,
                    "agent_message": "hi",
                    "status": "in_progress",
                    "finalized_config": None,
                }
            ]
        ),
    )

    await start.handle_user_message(message)

    start_stream.assert_called_once_with(api_key, "hi again")
    save_mock.assert_awaited_once_with(telegram_id, fresh_id)


def test_split_returns_single_chunk_for_short_text() -> None:
    assert start._split("short", 4000) == ["short"], (
        "_split should return the input unchanged when it already fits"
    )


def test_split_breaks_on_paragraph_boundary_when_available() -> None:
    text = "a" * 100 + "\n\n" + "b" * 100
    chunks = start._split(text, 150)
    assert len(chunks) == 2, f"_split produced {len(chunks)} chunks, expected 2"
    assert chunks[0].endswith("a"), "first chunk should end on the paragraph boundary"
    assert chunks[1].startswith("b"), "second chunk should start with the next paragraph"
