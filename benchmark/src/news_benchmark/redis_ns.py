"""
Prefixed Redis wrapper for the benchmark.

All Redis keys the conversational agent writes (e.g. `conv:user:{user_id}`)
get transparently prefixed with `bench_<run_id>:` so multiple concurrent
runs against the same devbox Redis never collide. flush_prefix() scans
and deletes every key carrying the prefix at teardown.

The wrapper is installed by monkey-patching `news_service.core.redis.get_redis_client`
(or equivalent) to return a Redis instance whose set/get/delete/expire
prepend the prefix. This keeps news_service code unmodified.

Usage:

    ns = NamespacedRedis.from_url(cfg.benchmark_redis_url, prefix=f"bench_{run_id}:")
    await ns.ping()
    ...
    await ns.flush_prefix()
"""

from __future__ import annotations

from typing import Any

import redis.asyncio as aioredis


class NamespacedRedis:
    """Thin wrapper around redis.asyncio.Redis that prefixes all keys."""

    def __init__(self, client: aioredis.Redis, prefix: str) -> None:
        self._client = client
        self._prefix = prefix

    @classmethod
    def from_url(cls, url: str, prefix: str) -> NamespacedRedis:
        return cls(aioredis.from_url(url, decode_responses=True), prefix)

    def _k(self, key: str) -> str:
        return f"{self._prefix}{key}"

    async def ping(self) -> bool:
        return await self._client.ping()

    async def get(self, key: str) -> Any:
        return await self._client.get(self._k(key))

    async def set(
        self,
        key: str,
        value: str,
        ex: int | None = None,
    ) -> Any:
        return await self._client.set(self._k(key), value, ex=ex)

    async def delete(self, *keys: str) -> int:
        return await self._client.delete(*(self._k(k) for k in keys))

    async def expire(self, key: str, seconds: int) -> Any:
        return await self._client.expire(self._k(key), seconds)

    async def flush_prefix(self) -> int:
        """Scan and delete every key carrying this run's prefix."""
        deleted = 0
        async for key in self._client.scan_iter(match=f"{self._prefix}*", count=500):
            await self._client.delete(key)
            deleted += 1
        return deleted

    async def close(self) -> None:
        await self._client.aclose()
