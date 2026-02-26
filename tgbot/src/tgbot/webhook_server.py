import logging

from aiohttp import web

logger = logging.getLogger(__name__)

_bot_instance = None


def set_bot(bot) -> None:  # noqa: ANN001
    global _bot_instance  # noqa: PLW0603
    _bot_instance = bot


async def handle_deliver(request: web.Request) -> web.Response:
    chat_id = request.match_info["chat_id"]

    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid json"}, status=400)

    subject = data.get("subject", "News Digest")
    body = data.get("body", "")
    text = f"*{subject}*\n\n{body}"

    if _bot_instance is None:
        logger.error("Bot instance not set, cannot deliver to %s", chat_id)
        return web.json_response({"error": "bot not ready"}, status=503)

    try:
        await _bot_instance.send_message(chat_id=int(chat_id), text=text, parse_mode="Markdown")
        logger.info("Digest delivered to chat_id=%s", chat_id)
        return web.json_response({"status": "delivered"})
    except Exception:
        logger.exception("Failed to deliver to chat_id=%s", chat_id)
        return web.json_response({"error": "delivery failed"}, status=500)


def create_webhook_app() -> web.Application:
    app = web.Application()
    app.router.add_post("/deliver/{chat_id}", handle_deliver)
    return app
