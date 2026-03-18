import redis.asyncio as aioredis

from news_service.core.config import get_settings


def get_redis_client() -> aioredis.Redis:
    """Create an async Redis client from the configured URL."""
    settings = get_settings()
    return aioredis.from_url(settings.redis_url, decode_responses=True)
