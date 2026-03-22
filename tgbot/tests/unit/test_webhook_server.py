import logging
import random
import uuid
from unittest.mock import AsyncMock

import pytest
from aiogram.enums import ParseMode
from aiohttp.test_utils import TestClient, TestServer

from tgbot.webhook_server import (
    TELEGRAM_MAX_MESSAGE_LENGTH,
    _split_text,
    create_webhook_app,
    delivery_webhook_path,
    set_bot,
)

logging.disable(logging.CRITICAL)


async def _make_webhook_client() -> TestClient:
    app = create_webhook_app()
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


@pytest.mark.asyncio
async def test_deliver_returns_200_on_success() -> None:
    client = await _make_webhook_client()
    try:
        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        set_bot(mock_bot)

        chat_id = random.randint(10000, 99999)
        response = await client.post(
            delivery_webhook_path(chat_id),
            json={
                "subject": f"\u0422\u0435\u0441\u0442-{uuid.uuid4().hex[:6]}",
                "body": "\u041d\u043e\u0432\u043e\u0441\u0442\u0438 \u0434\u043d\u044f.",
            },
        )

        assert response.status == 200, "deliver endpoint did not return 200 on success"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_deliver_returns_delivered_status() -> None:
    client = await _make_webhook_client()
    try:
        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        set_bot(mock_bot)

        chat_id = random.randint(10000, 99999)
        response = await client.post(
            delivery_webhook_path(chat_id),
            json={
                "subject": "\u0414\u0430\u0439\u0434\u0436\u0435\u0441\u0442",
                "body": "\u0421\u043e\u0434\u0435\u0440\u0436\u0438\u043c\u043e\u0435.",
            },
        )
        data = await response.json()

        assert data["status"] == "delivered", "deliver endpoint did not return delivered status"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_deliver_sends_message_to_correct_chat_id() -> None:
    client = await _make_webhook_client()
    try:
        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        set_bot(mock_bot)

        chat_id = random.randint(10000, 99999)
        await client.post(
            delivery_webhook_path(chat_id),
            json={"subject": "Test", "body": "\u0422\u0435\u043b\u043e."},
        )
        call_kwargs = mock_bot.send_message.call_args.kwargs

        assert call_kwargs["chat_id"] == chat_id, "deliver did not send to the correct chat_id"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_deliver_uses_html_parse_mode() -> None:
    client = await _make_webhook_client()
    try:
        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        set_bot(mock_bot)

        chat_id = random.randint(10000, 99999)
        await client.post(
            delivery_webhook_path(chat_id),
            json={"subject": "S", "body": "B"},
        )
        call_kwargs = mock_bot.send_message.call_args.kwargs

        assert call_kwargs["parse_mode"] == ParseMode.HTML, "deliver did not use HTML parse mode"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_deliver_disables_web_page_preview() -> None:
    client = await _make_webhook_client()
    try:
        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        set_bot(mock_bot)

        chat_id = random.randint(10000, 99999)
        await client.post(
            delivery_webhook_path(chat_id),
            json={"subject": "S", "body": "B"},
        )
        call_kwargs = mock_bot.send_message.call_args.kwargs

        assert call_kwargs["disable_web_page_preview"] is True, (
            "deliver did not disable web page preview"
        )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_deliver_invalid_json_returns_400() -> None:
    client = await _make_webhook_client()
    try:
        set_bot(AsyncMock())

        chat_id = random.randint(10000, 99999)
        response = await client.post(
            delivery_webhook_path(chat_id),
            data=b"not json",
            headers={"Content-Type": "application/json"},
        )

        assert response.status == 400, "deliver did not return 400 for invalid JSON"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_deliver_no_bot_returns_503() -> None:
    client = await _make_webhook_client()
    try:
        set_bot(None)

        chat_id = random.randint(10000, 99999)
        response = await client.post(
            delivery_webhook_path(chat_id),
            json={"subject": "\u0422\u0435\u0441\u0442", "body": "\u0422\u0435\u043b\u043e"},
        )

        assert response.status == 503, "deliver did not return 503 when bot is not set"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_deliver_splits_long_message_into_multiple_sends() -> None:
    client = await _make_webhook_client()
    try:
        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        set_bot(mock_bot)

        chat_id = random.randint(10000, 99999)
        long_body = "\u0430" * (TELEGRAM_MAX_MESSAGE_LENGTH * 2)
        await client.post(
            delivery_webhook_path(chat_id),
            json={
                "subject": "\u0414\u0430\u0439\u0434\u0436\u0435\u0441\u0442",
                "body": long_body,
            },
        )

        assert mock_bot.send_message.await_count >= 2, (
            "deliver did not split a long message into multiple sends"
        )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_deliver_rejects_legacy_unauthenticated_path() -> None:
    client = await _make_webhook_client()
    try:
        set_bot(AsyncMock())

        chat_id = random.randint(10000, 99999)
        response = await client.post(
            f"/deliver/{chat_id}",
            json={"subject": "Test", "body": "Body"},
        )

        assert response.status == 403, "deliver did not reject legacy unauthenticated path"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_deliver_rejects_invalid_token() -> None:
    client = await _make_webhook_client()
    try:
        set_bot(AsyncMock())

        chat_id = random.randint(10000, 99999)
        response = await client.post(
            f"/deliver/not-the-right-token/{chat_id}",
            json={"subject": "Test", "body": "Body"},
        )

        assert response.status == 403, "deliver did not reject an invalid token"
    finally:
        await client.close()


def test_split_text_returns_single_chunk_for_short_message() -> None:
    short = f"\u041a\u043e\u0440\u043e\u0442\u043a\u0438\u0439-{uuid.uuid4().hex[:8]}"
    result = _split_text(short, 100)

    assert result == [short], "split_text did not return a single chunk for short message"


def test_split_text_splits_paragraphs_into_two_halves() -> None:
    paras = [
        f"\u041f\u0430\u0440\u0430\u0433\u0440\u0430\u0444 {i}: {uuid.uuid4().hex[:8]}"
        for i in range(4)
    ]
    text = "\n\n".join(paras)
    chunks = _split_text(text, len(text) - 1)

    assert len(chunks) == 2, "split_text did not split into two balanced halves"


def test_split_text_preserves_all_content() -> None:
    paras = [
        f"\u0421\u043e\u0434\u0435\u0440\u0436\u0438\u043c\u043e\u0435 {i}: {uuid.uuid4().hex[:10]}"
        for i in range(6)
    ]
    text = "\n\n".join(paras)
    chunks = _split_text(text, len(text) // 2 + 10)
    reassembled = "\n\n".join(chunks)

    assert reassembled == text, "split_text did not preserve all content after reassembly"


def test_split_text_respects_max_length() -> None:
    max_len = random.randint(300, 500)
    paras = [
        f"\u041f\u0430\u0440\u0430\u0433\u0440\u0430\u0444 {i}: " + "\u0445" * 100
        for i in range(10)
    ]
    text = "\n\n".join(paras)
    chunks = _split_text(text, max_len)

    assert all(len(chunk) <= max_len for chunk in chunks), (
        "split_text produced a chunk exceeding max_length"
    )


def test_split_text_no_newlines_splits_at_sentence() -> None:
    sentences = [
        "\u041f\u0435\u0440\u0432\u043e\u0435. ",
        "\u0412\u0442\u043e\u0440\u043e\u0435. ",
        "\u0422\u0440\u0435\u0442\u044c\u0435. ",
        "\u0427\u0435\u0442\u0432\u0451\u0440\u0442\u043e\u0435.",
    ]
    text = "".join(sentences)
    chunks = _split_text(text, len(text) - 1)

    assert len(chunks) == 2, (
        "split_text did not split continuous text into two sentence-based chunks"
    )
