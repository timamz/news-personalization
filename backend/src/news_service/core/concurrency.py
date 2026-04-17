"""Global concurrency limits for LLM-heavy operations.

Semaphores cap how many expensive operations run simultaneously within
a single app process. This prevents overwhelming the LLM provider and
SearXNG instance when multiple users trigger heavy operations at once.
"""

import asyncio

from news_service.core.config import get_settings

settings = get_settings()

discovery_semaphore = asyncio.Semaphore(settings.max_concurrent_discoveries)
