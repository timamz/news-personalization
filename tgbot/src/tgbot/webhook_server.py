import logging
from hashlib import sha256
from secrets import compare_digest

from aiogram.enums import ParseMode
from aiohttp import web

from tgbot.core.config import get_settings
from tgbot.telegram_format import render_html_message

logger = logging.getLogger(__name__)

_bot_instance = None
TELEGRAM_MAX_MESSAGE_LENGTH = 4000
settings = get_settings()


def set_bot(bot) -> None:  # noqa: ANN001
    global _bot_instance  # noqa: PLW0603
    _bot_instance = bot


async def handle_deliver(request: web.Request) -> web.Response:
    token = request.match_info["token"]
    chat_id = request.match_info["chat_id"]
    if not compare_digest(token, _delivery_token()):
        logger.warning("Rejected unauthorized webhook delivery for chat_id=%s", chat_id)
        return web.json_response({"error": "forbidden"}, status=403)

    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)

    subject = data.get("subject", "News Digest")
    body = data.get("body", "")
    text = f"{subject}\n\n{body}"

    if _bot_instance is None:
        logger.error("Bot instance not set, cannot deliver to %s", chat_id)
        return web.json_response({"error": "bot not ready"}, status=503)

    try:
        for chunk in _split_text(text, TELEGRAM_MAX_MESSAGE_LENGTH):
            await _bot_instance.send_message(
                chat_id=int(chat_id),
                text=render_html_message(chunk),
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True,
            )
        logger.info("Digest delivered to chat_id=%s", chat_id)
        return web.json_response({"status": "delivered"})
    except Exception:
        logger.exception("Failed to deliver to chat_id=%s", chat_id)
        return web.json_response({"error": "delivery failed"}, status=500)


async def handle_legacy_deliver(request: web.Request) -> web.Response:
    chat_id = request.match_info["chat_id"]
    logger.warning("Rejected legacy unauthenticated delivery for chat_id=%s", chat_id)
    return web.json_response({"error": "forbidden"}, status=403)


def create_webhook_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/deliver/{chat_id}", handle_legacy_deliver)
    app.router.add_post("/deliver/{token}/{chat_id}", handle_deliver)
    return app


def delivery_webhook_path(chat_id: int) -> str:
    return f"/deliver/{_delivery_token()}/{chat_id}"


def delivery_webhook_url(chat_id: int) -> str:
    return f"http://{settings.webhook_public_host}:{settings.webhook_port}{delivery_webhook_path(chat_id)}"


def _split_text(text: str, max_length: int) -> list[str]:
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_length, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks


def _delivery_token() -> str:
    payload = f"deliver:{settings.bot_token}".encode()
    return sha256(payload).hexdigest()[:32]
