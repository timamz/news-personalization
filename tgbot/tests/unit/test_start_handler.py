"""Tests for /start, /help, and the generic text-message relay handler."""

import logging
import random
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from tgbot.handlers import start
from tgbot.text_split import split_for_telegram

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
async def test_cmd_start_sends_fixed_welcome_text(mocker) -> None:
    telegram_id = random.randint(100000, 999999)
    message = _make_message(telegram_id)
    mocker.patch.object(start, "ensure_api_key", new=AsyncMock(return_value="key"))
    mocker.patch.object(start.backend, "acknowledge_onboarding", new=AsyncMock())

    await start.cmd_start(message)

    assert message.answer.await_count >= 1, "cmd_start did not answer the user"
    spoken = "".join(call.args[0] for call in message.answer.await_args_list)
    assert "personal news assistant" in spoken.lower(), (
        f"cmd_start did not surface the onboarding text: {spoken!r}"
    )


@pytest.mark.asyncio
async def test_cmd_start_acknowledges_onboarding_with_the_users_api_key(mocker) -> None:
    telegram_id = random.randint(100000, 999999)
    message = _make_message(telegram_id)
    api_key = f"key-{uuid.uuid4().hex}"
    mocker.patch.object(start, "ensure_api_key", new=AsyncMock(return_value=api_key))
    ack = mocker.patch.object(start.backend, "acknowledge_onboarding", new=AsyncMock())

    await start.cmd_start(message)

    ack.assert_awaited_once_with(api_key)


@pytest.mark.asyncio
async def test_cmd_start_does_not_call_the_conversation_endpoint(mocker) -> None:
    telegram_id = random.randint(100000, 999999)
    message = _make_message(telegram_id)
    mocker.patch.object(start, "ensure_api_key", new=AsyncMock(return_value="key"))
    mocker.patch.object(start.backend, "acknowledge_onboarding", new=AsyncMock())
    stream_mock = mocker.patch.object(
        start.backend, "send_conversation_message_stream", new=AsyncMock()
    )

    await start.cmd_start(message)

    stream_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_cmd_start_still_sends_welcome_if_acknowledge_onboarding_fails(mocker) -> None:
    telegram_id = random.randint(100000, 999999)
    message = _make_message(telegram_id)
    mocker.patch.object(start, "ensure_api_key", new=AsyncMock(return_value="key"))
    mocker.patch.object(
        start.backend,
        "acknowledge_onboarding",
        new=AsyncMock(side_effect=RuntimeError("backend down")),
    )

    await start.cmd_start(message)

    spoken = "".join(call.args[0] for call in message.answer.await_args_list)
    assert "personal news assistant" in spoken.lower(), (
        f"cmd_start dropped the welcome text when acknowledge failed: {spoken!r}"
    )


@pytest.mark.asyncio
async def test_cmd_start_reports_error_when_registration_fails(mocker) -> None:
    telegram_id = random.randint(100000, 999999)
    message = _make_message(telegram_id)
    mocker.patch.object(start, "ensure_api_key", new=AsyncMock(side_effect=RuntimeError("boom")))

    await start.cmd_start(message)

    spoken_text = message.answer.await_args.args[0]
    assert "wrong" in spoken_text.lower(), (
        f"cmd_start did not surface an error to the user: {spoken_text!r}"
    )


@pytest.mark.asyncio
async def test_cmd_help_sends_fixed_help_text(mocker) -> None:
    telegram_id = random.randint(100000, 999999)
    message = _make_message(telegram_id)
    ensure_mock = mocker.patch.object(start, "ensure_api_key", new=AsyncMock())

    await start.cmd_help(message)

    ensure_mock.assert_not_awaited()
    spoken = "".join(call.args[0] for call in message.answer.await_args_list)
    assert "help" in spoken.lower() and "subscriptions" in spoken.lower(), (
        f"cmd_help did not surface the help text: {spoken!r}"
    )


@pytest.mark.asyncio
async def test_handle_user_message_ignores_empty_text(mocker) -> None:
    telegram_id = random.randint(100000, 999999)
    message = _make_message(telegram_id, text="   ")
    ensure_mock = mocker.patch.object(start, "ensure_api_key", new=AsyncMock())

    await start.handle_user_message(message)

    ensure_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_handle_user_message_streams_turn_to_backend(mocker) -> None:
    telegram_id = random.randint(100000, 999999)
    text = f"hello-{uuid.uuid4().hex[:6]}"
    message = _make_message(telegram_id, text=text)
    api_key = f"key-{uuid.uuid4().hex}"
    agent_text = f"pong-{uuid.uuid4().hex[:6]}"

    mocker.patch.object(start, "ensure_api_key", new=AsyncMock(return_value=api_key))
    stream_mock = mocker.patch.object(
        start.backend,
        "send_conversation_message_stream",
        return_value=_stream_events(
            [
                {"event": "status", "status_key": "status_thinking"},
                {"event": "done", "agent_message": agent_text},
            ]
        ),
    )

    await start.handle_user_message(message)

    stream_mock.assert_called_once_with(api_key, text)
    assert message.answer.await_count == 1, (
        f"handle_user_message sent {message.answer.await_count} messages, expected 1"
    )


@pytest.mark.asyncio
async def test_handle_user_message_reports_error_when_stream_raises(mocker) -> None:
    telegram_id = random.randint(100000, 999999)
    message = _make_message(telegram_id, text="hi")
    mocker.patch.object(start, "ensure_api_key", new=AsyncMock(return_value="key"))

    async def _raises(*_args, **_kwargs):
        raise RuntimeError("network dead")
        yield  # pragma: no cover

    mocker.patch.object(
        start.backend,
        "send_conversation_message_stream",
        side_effect=_raises,
    )

    await start.handle_user_message(message)

    spoken = message.answer.await_args.args[0]
    assert "wrong" in spoken.lower(), (
        f"handle_user_message did not surface the stream failure: {spoken!r}"
    )


def test_split_returns_single_chunk_for_short_text() -> None:
    assert split_for_telegram("short", 4000) == ["short"], (
        "split_for_telegram should return the input unchanged when it already fits"
    )


def test_split_breaks_on_paragraph_boundary_when_available() -> None:
    text = "a" * 100 + "\n\n" + "b" * 100
    chunks = split_for_telegram(text, 150)
    assert len(chunks) == 2, f"split_for_telegram produced {len(chunks)} chunks, expected 2"
    assert chunks[0].endswith("a"), "first chunk should end on the paragraph boundary"
    assert chunks[1].startswith("b"), "second chunk should start with the next paragraph"
