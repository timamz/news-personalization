"""Cross-service contract test for backend -> tgbot digest webhook delivery.

The backend's ``deliver`` helper in ``news_service.services.delivery`` POSTs a
JSON body to the frontend's webhook URL. The tgbot frontend exposes an
aiohttp endpoint (``tgbot.webhook_server.handle_deliver``) that parses that
JSON and forwards the message to Telegram via an aiogram ``Bot``.

This test pins the JSON payload contract that both services actually agreed
on: field names, chat_id handling, HTML parse mode, and preview disabling.
If the backend changes the webhook JSON structure, or the tgbot webhook
route changes its kwargs, this test must fail. It replaces mock-heavy unit
tests that independently asserted each side of the contract.

The test injects BOT_TOKEN into os.environ before importing any tgbot
module, because tgbot's ``Settings`` resolves at import time. It adds the
sibling ``tgbot/src`` directory to ``sys.path`` so the tgbot package is
importable from the backend pytest environment (the two services are
sibling packages in the monorepo).
"""

import os
import sys
import uuid
from pathlib import Path

import pytest

os.environ.setdefault("BOT_TOKEN", f"integration-test-token-{uuid.uuid4().hex}")
_TGBOT_SRC = Path(__file__).resolve().parents[3] / "tgbot" / "src"
if str(_TGBOT_SRC) not in sys.path:
    sys.path.insert(0, str(_TGBOT_SRC))

from aiohttp import web  # noqa: E402
from tgbot.webhook_server import (  # noqa: E402
    create_webhook_app,
    delivery_webhook_path,
    set_bot,
)

from news_service.services.delivery import deliver  # noqa: E402


class FakeBot:
    """aiogram Bot stub that records every send_message invocation.

    The real aiogram Bot opens a TCP connection to Telegram on construction,
    which is prohibited in integration tests. This fake exposes only the
    async surface that ``tgbot.webhook_server.handle_deliver`` actually
    calls, and stores each kwargs dict for assertion. The class is kept
    inline in the test module because it exists solely to verify one
    behavioral contract.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def send_message(
        self,
        chat_id: int,
        text: str,
        parse_mode: str,
        disable_web_page_preview: bool,
    ) -> None:
        self.calls.append(
            {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": parse_mode,
                "disable_web_page_preview": disable_web_page_preview,
            }
        )


@pytest.mark.asyncio(loop_scope="session")
async def test_backend_deliver_reaches_tgbot_webhook_with_agreed_json_contract() -> None:
    chat_id = int(uuid.uuid4().int % 1_000_000_000) + 1
    body_marker = uuid.uuid4().hex[:10]
    url_marker = uuid.uuid4().hex[:6]
    cyrillic_subject = f"Дайджест-{uuid.uuid4().hex[:6]}"
    cyrillic_body = (
        f"Новости дня {body_marker}.\n\nИсточник: https://example.test/article-{url_marker}"
    )

    fake_bot = FakeBot()
    set_bot(fake_bot)

    app = create_webhook_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="127.0.0.1", port=0)
    await site.start()

    try:
        bound_port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        webhook_url = f"http://127.0.0.1:{bound_port}{delivery_webhook_path(chat_id)}"

        await deliver(webhook_url, cyrillic_subject, cyrillic_body)

        assert len(fake_bot.calls) == 1, (
            "backend deliver did not result in exactly one send_message call on the tgbot side; "
            f"captured={len(fake_bot.calls)}"
        )
        call = fake_bot.calls[0]
        assert call["chat_id"] == chat_id, (
            "tgbot webhook did not forward the chat_id carried by the URL path to send_message"
        )
        assert isinstance(call["text"], str) and call["text"], (
            "tgbot webhook forwarded an empty or non-string text to send_message"
        )
        assert cyrillic_subject in call["text"], (
            "tgbot webhook dropped the subject field from the rendered message"
        )
        assert body_marker in call["text"], (
            "tgbot webhook dropped the body field from the rendered message"
        )
        assert url_marker in call["text"], (
            "tgbot webhook dropped the source URL from the rendered message"
        )
        assert call["parse_mode"] == "HTML", (
            "tgbot webhook did not request HTML parse_mode for the rendered digest"
        )
        assert call["disable_web_page_preview"] is True, (
            "tgbot webhook did not disable link previews for the rendered digest"
        )
        assert "<a href=" in call["text"], (
            "tgbot HTML renderer did not convert the source URL into an <a> tag; "
            "backend and tgbot may disagree on the body format"
        )
    finally:
        set_bot(None)
        await runner.cleanup()
