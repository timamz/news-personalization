"""
agent_tag context manager pushes an identifier onto a ContextVar stack.

Cost ledger wrappers read the top of this stack to attribute every
litellm call to the agent that invoked it. contextvars.copy_context()
naturally isolates parallel coroutines (e.g. parallel spawn_finder calls
from the Discovery agent), so each concurrent agent sees its own top-of-stack.

Usage:

    async with agent_tag("digest.writer"):
        ... anything here that calls litellm is attributed to digest.writer

    # Parallel spawn keeps tags isolated:
    async def one_finder():
        async with agent_tag("discovery.finder.broad"):
            await ...

    await asyncio.gather(one_finder(), one_finder())
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from contextvars import ContextVar

_TAG_STACK: ContextVar[tuple[str, ...]] = ContextVar("news_benchmark_agent_tag_stack", default=())


def current_tag() -> str:
    """Return the topmost agent tag or 'untagged' if the stack is empty."""
    stack = _TAG_STACK.get()
    return stack[-1] if stack else "untagged"


@asynccontextmanager
async def agent_tag(tag: str):
    """Push `tag` for the duration of the async block."""
    token = _TAG_STACK.set(_TAG_STACK.get() + (tag,))
    try:
        yield
    finally:
        _TAG_STACK.reset(token)
