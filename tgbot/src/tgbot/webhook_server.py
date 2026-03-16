import logging
from hashlib import sha256
from secrets import compare_digest

from aiogram.enums import ParseMode
from aiohttp import web

from tgbot.core.config import get_settings
from tgbot.telegram_format import render_html_message

logger = logging.getLogger(__name__)

_bot_instance = None
TELEGRAM_MAX_MESSAGE_LENGTH = 4096
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

    subject = data.get("subject", "")
    body = data.get("body", "")
    text = f"{subject}\n\n{body}".strip() if subject else body

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

    paragraphs = text.split("\n\n")
    if len(paragraphs) < 2:
        return _split_hard(text, max_length)

    # Split into N roughly equal parts so each fits within max_length.
    num_parts = _min_parts(paragraphs, max_length)
    return _balanced_split(paragraphs, num_parts)


def _min_parts(paragraphs: list[str], max_length: int) -> int:
    total = sum(len(p) for p in paragraphs) + 2 * (len(paragraphs) - 1)
    parts = max(2, -(-total // max_length))  # ceil division, at least 2
    while parts <= len(paragraphs):
        chunks = _balanced_split(paragraphs, parts)
        if all(len(c) <= max_length for c in chunks):
            return parts
        parts += 1
    return len(paragraphs)


def _balanced_split(paragraphs: list[str], num_parts: int) -> list[str]:
    total_len = sum(len(p) for p in paragraphs) + 2 * (len(paragraphs) - 1)
    target = total_len / num_parts

    chunks: list[str] = []
    current_paragraphs: list[str] = []
    current_len = 0

    for para in paragraphs:
        separator_len = 2 if current_paragraphs else 0
        new_len = current_len + separator_len + len(para)

        if current_paragraphs and len(chunks) < num_parts - 1 and new_len > target:
            chunks.append("\n\n".join(current_paragraphs))
            current_paragraphs = [para]
            current_len = len(para)
        else:
            current_paragraphs.append(para)
            current_len = new_len

    if current_paragraphs:
        chunks.append("\n\n".join(current_paragraphs))
    return chunks


def _split_hard(text: str, max_length: int) -> list[str]:
    """Fallback: split on newlines, then sentences, then hard character boundary."""
    chunks: list[str] = []
    current = ""
    for line in text.split("\n"):
        candidate = f"{current}\n{line}" if current else line
        if len(candidate) > max_length and current:
            chunks.append(current)
            current = line
        else:
            current = candidate
    if current:
        chunks.append(current)
    # If any chunk still exceeds max_length, split on nearest sentence boundary
    result: list[str] = []
    for chunk in chunks:
        while len(chunk) > max_length:
            part = _split_at_sentence(chunk, max_length)
            result.append(part)
            chunk = chunk[len(part) :].lstrip()
        if chunk:
            result.append(chunk)
    return result


def _split_at_sentence(text: str, max_length: int) -> str:
    """Split at the sentence-ending '.' closest to the middle, within max_length."""
    window = text[:max_length]
    mid = len(window) // 2
    # Search outward from the middle for a '. ' or '.\n' boundary
    best = -1
    for i in range(mid + 1):
        for pos in (mid + i, mid - i):
            if 0 <= pos < len(window) - 1 and window[pos] == ".":
                next_ch = window[pos + 1]
                if next_ch in (" ", "\n"):
                    best = pos
                    break
        if best != -1:
            break
    if best != -1:
        return text[: best + 1]
    return text[:max_length]


def _delivery_token() -> str:
    payload = f"deliver:{settings.bot_token}".encode()
    return sha256(payload).hexdigest()[:32]
