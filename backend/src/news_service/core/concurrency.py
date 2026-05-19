"""Global concurrency limits for LLM-heavy operations.

Semaphores cap how many expensive operations run simultaneously within
a single app process. This prevents overwhelming the LLM provider and
the Yandex Search API when multiple users trigger heavy operations at once.

Semaphores are created lazily per running event loop so that Celery
workers (which call asyncio.run() per task, creating a fresh loop each
time) do not inherit a semaphore bound to a previous loop.
"""

import asyncio

from news_service.core.config import get_settings

settings = get_settings()

_search_semaphore: asyncio.Semaphore | None = None
_search_semaphore_loop: asyncio.AbstractEventLoop | None = None

_discovery_semaphore: asyncio.Semaphore | None = None
_discovery_semaphore_loop: asyncio.AbstractEventLoop | None = None


def get_search_semaphore() -> asyncio.Semaphore:
    global _search_semaphore, _search_semaphore_loop
    loop = asyncio.get_running_loop()
    if _search_semaphore is None or _search_semaphore_loop is not loop:
        _search_semaphore = asyncio.Semaphore(settings.max_concurrent_web_searches)
        _search_semaphore_loop = loop
    return _search_semaphore


def get_discovery_semaphore() -> asyncio.Semaphore:
    global _discovery_semaphore, _discovery_semaphore_loop
    loop = asyncio.get_running_loop()
    if _discovery_semaphore is None or _discovery_semaphore_loop is not loop:
        _discovery_semaphore = asyncio.Semaphore(settings.max_concurrent_discoveries)
        _discovery_semaphore_loop = loop
    return _discovery_semaphore
