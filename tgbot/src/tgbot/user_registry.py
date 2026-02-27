import logging

from tgbot.client import BackendClient
from tgbot.storage import get_api_key, save_api_key

logger = logging.getLogger(__name__)


async def ensure_api_key(telegram_id: int, backend: BackendClient) -> str:
    existing_api_key = await get_api_key(telegram_id)
    if existing_api_key is not None:
        return existing_api_key

    new_api_key = await backend.register_user()
    await save_api_key(telegram_id, new_api_key)
    logger.info("Registered new user for telegram_id=%d", telegram_id)
    return new_api_key
