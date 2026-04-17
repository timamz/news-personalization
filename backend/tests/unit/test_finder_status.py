"""Tests for the finder agent's status events carrying status_text."""

import asyncio
import logging
import uuid
from unittest.mock import AsyncMock, patch

import google.adk.agents
import pytest

logging.disable(logging.CRITICAL)


async def _capture_tools_from_run_finder(queue: asyncio.Queue) -> dict:
    """Boot run_finder with a no-op agent runner, capturing its registered tools."""
    captured: dict = {}
    original_init = google.adk.agents.Agent.__init__

    def capturing_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        for tool in self.tools:
            if callable(tool):
                captured[tool.__name__] = tool

    with (
        patch(
            "news_service.agents.source_discovery.finder.run_agent_text",
            new_callable=AsyncMock,
            return_value="Done.",
        ),
        patch.object(google.adk.agents.Agent, "__init__", capturing_init),
    ):
        from news_service.agents.source_discovery.finder import run_finder

        await run_finder(
            strategy=f"Find sources about {uuid.uuid4().hex[:6]}",
            session=AsyncMock(),
            prompt_embedding=[0.1] * 10,
            exclude_urls=[],
            status_queue=queue,
        )
    return captured


@pytest.mark.asyncio
async def test_search_web_tool_emits_status_with_query_text() -> None:
    queue: asyncio.Queue = asyncio.Queue()
    query = f"best RSS feeds about neuroscience {uuid.uuid4().hex[:6]}"

    with patch(
        "news_service.agents.source_discovery.finder.search_web",
        new_callable=AsyncMock,
        return_value="No results found.",
    ):
        tools = await _capture_tools_from_run_finder(queue)
        await tools["tool_search_web"](query)

    event = queue.get_nowait()
    assert event["status_key"] == "status_searching_web"
    assert query[:60] in event["status_text"]


@pytest.mark.asyncio
async def test_validate_source_tool_emits_status_with_url() -> None:
    queue: asyncio.Queue = asyncio.Queue()
    source_url = f"https://{uuid.uuid4().hex[:8]}.example.com/feed"

    with patch(
        "news_service.agents.source_discovery.finder.score_candidate",
        new_callable=AsyncMock,
        return_value=(0.75, ["Sample text"]),
    ):
        tools = await _capture_tools_from_run_finder(queue)
        await tools["validate_and_score_source"](source_url, "rss")

    event = queue.get_nowait()
    assert event["status_key"] == "status_validating_source"
    assert source_url[:60] in event["status_text"]
