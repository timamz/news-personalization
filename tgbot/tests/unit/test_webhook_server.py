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


@pytest.fixture
async def webhook_client():
    app = create_webhook_app()
    async with TestClient(TestServer(app)) as client:
        yield client


@pytest.mark.asyncio
async def test_deliver_success(webhook_client: TestClient):
    mock_bot = AsyncMock()
    mock_bot.send_message = AsyncMock()
    set_bot(mock_bot)

    response = await webhook_client.post(
        delivery_webhook_path(12345),
        json={"subject": "Test Digest", "body": "Here is your news."},
    )
    assert response.status == 200
    data = await response.json()
    assert data["status"] == "delivered"

    mock_bot.send_message.assert_called_once()
    call_kwargs = mock_bot.send_message.call_args
    assert call_kwargs.kwargs["chat_id"] == 12345
    assert call_kwargs.kwargs["parse_mode"] == ParseMode.HTML
    assert call_kwargs.kwargs["disable_web_page_preview"] is True


@pytest.mark.asyncio
async def test_deliver_invalid_json(webhook_client: TestClient):
    set_bot(AsyncMock())

    response = await webhook_client.post(
        delivery_webhook_path(12345),
        data=b"not json",
        headers={"Content-Type": "application/json"},
    )
    assert response.status == 400


@pytest.mark.asyncio
async def test_deliver_no_bot(webhook_client: TestClient):
    set_bot(None)

    response = await webhook_client.post(
        delivery_webhook_path(12345),
        json={"subject": "Test", "body": "Body"},
    )
    assert response.status == 503


@pytest.mark.asyncio
async def test_deliver_splits_long_message(webhook_client: TestClient):
    mock_bot = AsyncMock()
    mock_bot.send_message = AsyncMock()
    set_bot(mock_bot)

    long_body = "a" * (TELEGRAM_MAX_MESSAGE_LENGTH * 2)
    response = await webhook_client.post(
        delivery_webhook_path(12345),
        json={"subject": "Long Digest", "body": long_body},
    )

    assert response.status == 200
    assert mock_bot.send_message.await_count >= 2


@pytest.mark.asyncio
async def test_deliver_rejects_legacy_unauthenticated_path(webhook_client: TestClient):
    set_bot(AsyncMock())

    response = await webhook_client.post(
        "/deliver/12345",
        json={"subject": "Test Digest", "body": "Here is your news."},
    )

    assert response.status == 403


@pytest.mark.asyncio
async def test_deliver_rejects_invalid_token(webhook_client: TestClient):
    set_bot(AsyncMock())

    response = await webhook_client.post(
        "/deliver/not-the-right-token/12345",
        json={"subject": "Test Digest", "body": "Here is your news."},
    )

    assert response.status == 403


def test_split_text_short_message():
    assert _split_text("short", 100) == ["short"]


def test_split_text_splits_paragraphs_into_balanced_halves():
    paras = ["Para one.", "Para two.", "Para three.", "Para four."]
    text = "\n\n".join(paras)
    # Use a limit that fits 2 paragraphs but not all 4
    chunks = _split_text(text, len(text) - 1)
    assert len(chunks) == 2
    assert "Para one." in chunks[0]
    assert "Para two." in chunks[0]
    assert "Para three." in chunks[1]
    assert "Para four." in chunks[1]


def test_split_text_preserves_all_content():
    paras = [f"Paragraph {i} content here." for i in range(6)]
    text = "\n\n".join(paras)
    chunks = _split_text(text, len(text) // 2 + 10)
    reassembled = "\n\n".join(chunks)
    assert reassembled == text


def test_split_text_respects_max_length():
    paras = [f"Paragraph {i}: " + "x" * 100 for i in range(10)]
    text = "\n\n".join(paras)
    chunks = _split_text(text, 400)
    for chunk in chunks:
        assert len(chunk) <= 400


def test_split_text_no_newlines_splits_at_sentence():
    sentences = ["First sentence. ", "Second sentence. ", "Third sentence. ", "Fourth sentence."]
    text = "".join(sentences)
    # Limit that forces a split but fits ~2 sentences
    chunks = _split_text(text, len(text) - 1)
    assert len(chunks) == 2
    assert chunks[0].endswith(".")
    assert "First sentence" in chunks[0]
    assert "Fourth sentence" in chunks[1]
