import redis.asyncio as aioredis

from news_service.core.config import get_settings

_client: aioredis.Redis | None = None


def get_redis_client() -> aioredis.Redis:
    """Return the process-wide async Redis client, creating it on first use.

    The client owns its own connection pool; reusing it avoids opening a
    fresh socket for every request. Call ``close_redis_client()`` during
    application shutdown to release the pool.
    """
    global _client
    if _client is None:
        _client = aioredis.from_url(get_settings().redis_url, decode_responses=True)
    return _client


async def close_redis_client() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
