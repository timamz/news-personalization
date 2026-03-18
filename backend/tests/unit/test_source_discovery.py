"""Tests for the agentic source discovery module."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.agents.discovery import DiscoveredSourceItem
from news_service.agents.source_discovery import (
    ScoredSource,
    SourceDiscoveryResult,
    _create_source_discovery_agent,
    _format_discovered_sources,
    run_source_discovery,
    tool_discover_rss_feeds,
    tool_discover_telegram_channels,
)


def test_format_discovered_sources_empty():
    assert _format_discovered_sources([]) == "No sources found."


def test_format_discovered_sources_formats_items():
    items = [
        DiscoveredSourceItem(url="https://a.com/feed", title="Feed A", source_kind="rss"),
        DiscoveredSourceItem(
            url="https://t.me/s/chan", title="Chan", source_kind="telegram_channel"
        ),
    ]
    result = _format_discovered_sources(items)
    assert "https://a.com/feed" in result
    assert "rss" in result
    assert "https://t.me/s/chan" in result
    assert "telegram_channel" in result


@pytest.mark.asyncio
async def test_tool_discover_rss_feeds_wraps_discovery(mocker):
    items = [DiscoveredSourceItem(url="https://a.com/rss", title="A", source_kind="rss")]
    mocker.patch(
        "news_service.agents.source_discovery.discover_rss_feeds",
        new=AsyncMock(return_value=items),
    )

    result = await tool_discover_rss_feeds.on_invoke_tool(MagicMock(), '{"query": "AI news"}')
    assert "https://a.com/rss" in result


@pytest.mark.asyncio
async def test_tool_discover_telegram_channels_wraps_discovery(mocker):
    items = [
        DiscoveredSourceItem(
            url="https://t.me/s/aichan", title="AI Chan", source_kind="telegram_channel"
        )
    ]
    mocker.patch(
        "news_service.agents.source_discovery.discover_telegram_channels",
        new=AsyncMock(return_value=items),
    )

    result = await tool_discover_telegram_channels.on_invoke_tool(
        MagicMock(), '{"query": "AI news"}'
    )
    assert "https://t.me/s/aichan" in result


@pytest.mark.asyncio
async def test_create_agent_has_correct_tools():
    session = AsyncMock()
    agent = _create_source_discovery_agent(session, [0.1] * 10)

    tool_names = {t.name for t in agent.tools}
    assert "search_existing_sources" in tool_names
    assert "tool_discover_rss_feeds" in tool_names
    assert "tool_discover_telegram_channels" in tool_names
    assert "tool_discover_reddit_subreddits" in tool_names
    # tool_discover_twitter_accounts disabled until Twitter rate limits stabilize
    assert "validate_and_score_source" in tool_names
    assert len(agent.tools) == 5


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
    """The closure-based search tool calls find_similar_feeds."""
    session = AsyncMock()
    prompt_embedding = [0.1] * 10

    mock_feed = SimpleNamespace(
        url="https://existing.com/feed",
        title="Existing",
        source_description="Covers AI news",
    )
    mocker.patch(
        "news_service.agents.source_discovery.embed_text",
        new=AsyncMock(return_value=[0.2] * 10),
    )
    mocker.patch(
        "news_service.agents.source_discovery.find_similar_feeds",
        new=AsyncMock(return_value=[mock_feed]),
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
