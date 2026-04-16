"""Tests for the finder agent's status events with status_text field."""

import asyncio
import logging
import uuid
from unittest.mock import AsyncMock, patch

import pytest

logging.disable(logging.CRITICAL)


@pytest.mark.asyncio
async def test_finder_without_status_queue_does_not_raise() -> None:
    with patch(
        "news_service.agents.source_discovery.finder.run_agent_text",
        new_callable=AsyncMock,
        return_value="Done.",
    ):
        from news_service.agents.source_discovery.finder import run_finder

        result = await run_finder(
            strategy=f"Find sources about {uuid.uuid4().hex[:6]}",
            session=AsyncMock(),
            prompt_embedding=[0.1] * 10,
            exclude_urls=[],
            status_queue=None,
        )

    assert isinstance(result, list), "run_finder without queue did not return a list"


@pytest.mark.asyncio
async def test_finder_accepts_status_queue_parameter() -> None:
    queue: asyncio.Queue = asyncio.Queue()

    with patch(
        "news_service.agents.source_discovery.finder.run_agent_text",
        new_callable=AsyncMock,
        return_value="Done.",
    ):
        from news_service.agents.source_discovery.finder import run_finder

        result = await run_finder(
            strategy=f"Find Telegram channels about {uuid.uuid4().hex[:6]}",
            session=AsyncMock(),
            prompt_embedding=[0.1] * 10,
            exclude_urls=[],
            status_queue=queue,
        )

    assert isinstance(result, list), "run_finder with queue did not return a list"


@pytest.mark.asyncio
async def test_search_web_tool_emits_status_text() -> None:
    """Directly invoke the tool_search_web closure to verify status_text is emitted."""
    queue: asyncio.Queue = asyncio.Queue()
    search_query = f"best RSS feeds about neuroscience {uuid.uuid4().hex[:6]}"

    with (
        patch(
            "news_service.agents.source_discovery.finder.search_web",
            new_callable=AsyncMock,
            return_value="No results found.",
        ),
        patch(
            "news_service.agents.source_discovery.finder.run_agent_text",
            new_callable=AsyncMock,
            return_value="Done.",
        ),
    ):
        from news_service.agents.source_discovery.finder import run_finder

        # We need to capture the tool_search_web closure. Patch run_agent_text
        # to intercept agent creation and extract tools.
        captured_tools: dict = {}

        original_agent_init = None

        import google.adk.agents

        original_agent_init = google.adk.agents.Agent.__init__

        def capturing_init(self, *args, **kwargs):
            original_agent_init(self, *args, **kwargs)
            for tool in self.tools:
                if callable(tool):
                    captured_tools[tool.__name__] = tool

        with patch.object(google.adk.agents.Agent, "__init__", capturing_init):
            await run_finder(
                strategy="Search web for neuroscience feeds",
                session=AsyncMock(),
                prompt_embedding=[0.1] * 10,
                exclude_urls=[],
                status_queue=queue,
            )

        assert "tool_search_web" in captured_tools, "tool_search_web was not registered as a tool"

        web_tool = captured_tools["tool_search_web"]
        await web_tool(search_query)

    assert not queue.empty(), "tool_search_web did not emit any status event"
    event = queue.get_nowait()
    assert event["status_key"] == "status_searching_web", (
        "status event key is not 'status_searching_web'"
    )
    assert "status_text" in event, "status event does not contain status_text field"
    assert search_query[:60] in event["status_text"], "status_text does not contain the query"


@pytest.mark.asyncio
async def test_validate_source_tool_emits_status_text() -> None:
    """Directly invoke validate_and_score_source closure to verify status_text."""
    queue: asyncio.Queue = asyncio.Queue()
    source_url = f"https://{uuid.uuid4().hex[:8]}.example.com/feed"

    with (
        patch(
            "news_service.agents.source_discovery.finder.score_candidate",
            new_callable=AsyncMock,
            return_value=(0.75, ["Sample text about neuroscience"]),
        ),
        patch(
            "news_service.agents.source_discovery.finder.run_agent_text",
            new_callable=AsyncMock,
            return_value="Done.",
        ),
    ):
        from news_service.agents.source_discovery.finder import run_finder

        captured_tools: dict = {}

        import google.adk.agents

        original_agent_init = google.adk.agents.Agent.__init__

        def capturing_init(self, *args, **kwargs):
            original_agent_init(self, *args, **kwargs)
            for tool in self.tools:
                if callable(tool):
                    captured_tools[tool.__name__] = tool

        with patch.object(google.adk.agents.Agent, "__init__", capturing_init):
            await run_finder(
                strategy="Validate a specific source",
                session=AsyncMock(),
                prompt_embedding=[0.1] * 10,
                exclude_urls=[],
                status_queue=queue,
            )

        assert "validate_and_score_source" in captured_tools, (
            "validate_and_score_source was not registered as a tool"
        )

        validate_tool = captured_tools["validate_and_score_source"]
        await validate_tool(source_url, "rss")

    assert not queue.empty(), "validate_and_score_source did not emit any status event"
    event = queue.get_nowait()
    assert event["status_key"] == "status_validating_source", (
        "status event key is not 'status_validating_source'"
    )
    assert "status_text" in event, "status event does not contain status_text field"
    assert source_url[:60] in event["status_text"], "status_text does not contain the URL"
