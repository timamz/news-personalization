import asyncio
import logging

from aiogram import Bot, Dispatcher
from aiohttp import web

from tgbot.core.config import get_settings
from tgbot.handlers import start, subscribe, subscriptions
from tgbot.storage import init_db
from tgbot.webhook_server import create_webhook_app, set_bot

logger = logging.getLogger(__name__)


async def run_webhook_server(host: str, port: int) -> None:
    app = create_webhook_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    logger.info("Webhook server started on %s:%d", host, port)


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    settings = get_settings()

    bot = Bot(token=settings.bot_token)
    dp = Dispatcher()

    await init_db(settings.bot_storage_path)

    dp.include_router(start.router)
    dp.include_router(subscribe.router)
    dp.include_router(subscriptions.router)

    set_bot(bot)

    await run_webhook_server(settings.webhook_host, settings.webhook_port)

    logger.info("Starting bot polling...")
    try:
        await dp.start_polling(bot)
    finally:
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
