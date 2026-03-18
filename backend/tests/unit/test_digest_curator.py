"""Tests for the agentic digest curation module."""

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.agents.digest_curator import (
    DigestCurationResult,
    _create_digest_curator_agent,
    _format_news_item,
    run_digest_curator,
)


def _make_news_item(
    headline: str = "Test Headline",
    body: str = "Test body content",
    url: str = "https://example.com/article",
) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        headline=headline,
        body=body,
        url=url,
        published_at=None,
        fetched_at=None,
    )


def test_format_news_item_includes_id_and_fields():
    item = _make_news_item()
    result = _format_news_item(item)
    assert f"[ID: {item.id}]" in result
    assert "Test Headline" in result
    assert "Test body content" in result
    assert "https://example.com/article" in result


@pytest.mark.asyncio
async def test_create_agent_has_correct_tools():
    session = AsyncMock()
    agent = _create_digest_curator_agent(
        session=session,
        query_embedding=[0.1] * 10,
        exclude_ids=set(),
        allowed_feed_ids={uuid.uuid4()},
        published_after=MagicMock(),
        format_instructions="brief summary",
        digest_language="en",
    )

    tool_names = {t.name for t in agent.tools}
    assert "search_news_by_relevance" in tool_names
    assert "search_news_by_recency" in tool_names
    assert "get_article_details" in tool_names
    assert len(agent.tools) == 3


@pytest.mark.asyncio
async def test_create_agent_uses_structured_output():
    session = AsyncMock()
    agent = _create_digest_curator_agent(
        session=session,
        query_embedding=[0.1] * 10,
        exclude_ids=set(),
        allowed_feed_ids={uuid.uuid4()},
        published_after=MagicMock(),
        format_instructions="brief summary",
        digest_language="en",
    )
    assert agent.output_type is DigestCurationResult


@pytest.mark.asyncio
async def test_create_agent_uses_russian_label_for_ru():
    session = AsyncMock()
    agent = _create_digest_curator_agent(
        session=session,
        query_embedding=[0.1] * 10,
        exclude_ids=set(),
        allowed_feed_ids={uuid.uuid4()},
        published_after=MagicMock(),
        format_instructions="краткое содержание",
        digest_language="ru",
    )
    assert "Источник" in agent.instructions


@pytest.mark.asyncio
async def test_create_agent_uses_english_label_for_en():
    session = AsyncMock()
    agent = _create_digest_curator_agent(
        session=session,
        query_embedding=[0.1] * 10,
        exclude_ids=set(),
        allowed_feed_ids={uuid.uuid4()},
        published_after=MagicMock(),
        format_instructions="brief summary",
        digest_language="en",
    )
    assert "Source" in agent.instructions


@pytest.mark.asyncio
async def test_run_digest_curator_returns_result(mocker):
    expected = DigestCurationResult(
        digest_text="Here is your digest...",
        used_item_ids=[str(uuid.uuid4())],
    )
    mock_run_result = MagicMock()
    mock_run_result.final_output = expected

    mocker.patch(
        "news_service.agents.digest_curator.Runner.run",
        new=AsyncMock(return_value=mock_run_result),
    )

    result = await run_digest_curator(
        session=AsyncMock(),
        query_embedding=[0.1] * 10,
        exclude_ids=set(),
        allowed_feed_ids={uuid.uuid4()},
        published_after=MagicMock(),
        format_instructions="brief summary",
        digest_language="en",
    )

    assert result is not None
    assert result.digest_text == "Here is your digest..."
    assert len(result.used_item_ids) == 1


@pytest.mark.asyncio
async def test_run_digest_curator_returns_none_when_no_items(mocker):
    empty_result = DigestCurationResult(digest_text="", used_item_ids=[])
    mock_run_result = MagicMock()
    mock_run_result.final_output = empty_result

    mocker.patch(
        "news_service.agents.digest_curator.Runner.run",
        new=AsyncMock(return_value=mock_run_result),
    )

    result = await run_digest_curator(
        session=AsyncMock(),
        query_embedding=[0.1] * 10,
        exclude_ids=set(),
        allowed_feed_ids={uuid.uuid4()},
        published_after=MagicMock(),
        format_instructions="brief summary",
        digest_language="en",
    )

    assert result is None


@pytest.mark.asyncio
async def test_search_news_by_relevance_tool(mocker):
    session = AsyncMock()
    item = _make_news_item()

    mocker.patch(
        "news_service.agents.digest_curator.find_similar_news",
        new=AsyncMock(return_value=[item]),
    )

    agent = _create_digest_curator_agent(
        session=session,
        query_embedding=[0.1] * 10,
        exclude_ids=set(),
        allowed_feed_ids={uuid.uuid4()},
        published_after=MagicMock(),
        format_instructions="brief summary",
        digest_language="en",
    )
    relevance_tool = next(t for t in agent.tools if t.name == "search_news_by_relevance")

    result = await relevance_tool.on_invoke_tool(MagicMock(), '{"limit": 10}')
    assert "Test Headline" in result
    assert str(item.id) in result
