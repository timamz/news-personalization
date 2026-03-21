"""Tests for the agentic source discovery module."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.agents.source_discovery import (
    ScoredSource,
    SourceDiscoveryResult,
    _create_source_discovery_agent,
    run_source_discovery,
    tool_search_web,
)


@pytest.mark.asyncio
async def test_tool_search_web_wraps_search(mocker):
    mocker.patch(
        "news_service.agents.source_discovery.search_web",
        new=AsyncMock(return_value="Found: https://a.com/rss - AI news feed"),
    )

    result = await tool_search_web.on_invoke_tool(
        MagicMock(), '{"query": "best RSS feeds about AI"}'
    )
    assert "https://a.com/rss" in result


@pytest.mark.asyncio
async def test_create_agent_has_correct_tools():
    session = AsyncMock()
    agent = _create_source_discovery_agent(session, [0.1] * 10)

    tool_names = {t.name for t in agent.tools}
    assert "search_existing_sources" in tool_names
    assert "tool_search_web" in tool_names
    assert "validate_and_score_source" in tool_names
    assert len(agent.tools) == 3


@pytest.mark.asyncio
async def test_create_agent_uses_structured_output():
    session = AsyncMock()
    agent = _create_source_discovery_agent(session, [0.1] * 10)
    assert agent.output_type is SourceDiscoveryResult


@pytest.mark.asyncio
async def test_run_source_discovery_returns_agent_result(mocker):
    expected = SourceDiscoveryResult(
        sources=[
            ScoredSource(url="https://a.com", title="A", source_kind="rss", relevance_score=0.9)
        ]
    )

    mock_run_result = MagicMock()
    mock_run_result.final_output = expected

    mocker.patch(
        "news_service.agents.source_discovery.Runner.run",
        new=AsyncMock(return_value=mock_run_result),
    )

    session = AsyncMock()
    result = await run_source_discovery(
        session=session,
        raw_prompt="AI news",
        prompt_embedding=[0.1] * 10,
    )

    assert result == expected
    assert len(result.sources) == 1
    assert result.sources[0].url == "https://a.com"


@pytest.mark.asyncio
async def test_search_existing_sources_tool(mocker):
    """The closure-based search tool calls find_similar_sources."""
    session = AsyncMock()
    prompt_embedding = [0.1] * 10

    mock_source = SimpleNamespace(
        url="https://existing.com/feed",
        title="Existing",
        source_description="Covers AI news",
    )
    mocker.patch(
        "news_service.agents.source_discovery.embed_text",
        new=AsyncMock(return_value=[0.2] * 10),
    )
    mocker.patch(
        "news_service.agents.source_discovery.find_similar_sources",
        new=AsyncMock(return_value=[mock_source]),
    )

    agent = _create_source_discovery_agent(session, prompt_embedding)
    search_tool = next(t for t in agent.tools if t.name == "search_existing_sources")

    result = await search_tool.on_invoke_tool(MagicMock(), '{"query": "AI"}')
    assert "https://existing.com/feed" in result
    assert "Existing" in result


@pytest.mark.asyncio
async def test_validate_and_score_source_tool(mocker):
    """The closure-based scoring tool calls score_candidate."""
    session = AsyncMock()
    prompt_embedding = [0.1] * 10

    mocker.patch(
        "news_service.agents.source_discovery.score_candidate",
        new=AsyncMock(return_value=(0.85, ["sample post text"])),
    )

    agent = _create_source_discovery_agent(session, prompt_embedding)
    score_tool = next(t for t in agent.tools if t.name == "validate_and_score_source")

    result = await score_tool.on_invoke_tool(
        MagicMock(), '{"url": "https://a.com/feed", "source_kind": "rss"}'
    )
    assert "0.850" in result
    assert "https://a.com/feed" in result
