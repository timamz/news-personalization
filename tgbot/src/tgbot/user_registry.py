import logging

from tgbot.client import BackendClient
from tgbot.storage import get_api_key, save_api_key

logger = logging.getLogger(__name__)

_VALIDATED_KEYS: set[str] = set()


async def ensure_api_key(telegram_id: int, backend: BackendClient) -> str:
    existing_api_key = await get_api_key(telegram_id)
    if existing_api_key is not None:
        if existing_api_key in _VALIDATED_KEYS or await backend.api_key_is_valid(existing_api_key):
            _VALIDATED_KEYS.add(existing_api_key)
            return existing_api_key
        logger.warning(
            "Cached api_key for telegram_id=%d rejected by backend, re-registering",
            telegram_id,
        )

    new_api_key = await backend.register_user()
    await save_api_key(telegram_id, new_api_key)
    _VALIDATED_KEYS.add(new_api_key)
    logger.info("Registered new user for telegram_id=%d", telegram_id)
    return new_api_key
