"""Tests for the Digest Writer ADK agent (writer.py).

Verifies tool behaviour, budget enforcement, prompt construction,
and the overall write_digest function contract.
"""

import logging
import uuid
from unittest.mock import AsyncMock, patch

import pytest

logging.disable(logging.CRITICAL)

_WRITER_MODULE = "news_service.agents.digest.writer"


def _random_html(body_text: str) -> str:
    tag_id = uuid.uuid4().hex[:6]
    return (
        f"<html><head><title>T-{tag_id}</title></head>"
        f"<body><nav>skip</nav><p>{body_text}</p></body></html>"
    )


def _fake_run_agent_text_with_submit(digest_text: str, used_ids: str):
    """Return an AsyncMock for run_agent_text that calls submit_digest internally."""

    async def _side_effect(*, agent, message, user_id):
        for tool_fn in agent.tools:
            if getattr(tool_fn, "__name__", "") == "submit_digest":
                await tool_fn(digest_text, used_ids)
                return "done"
        raise AssertionError("submit_digest tool not found on agent")

    return AsyncMock(side_effect=_side_effect)


def _fake_run_agent_text_no_submit():
    """Return an AsyncMock for run_agent_text that never calls submit_digest."""

    async def _side_effect(*, agent, message, user_id):
        return "I forgot to submit."

    return AsyncMock(side_effect=_side_effect)


def _fake_run_agent_text_with_tools(digest_text: str, used_ids: str, fetch_urls: list[str]):
    """Return a mock that calls fetch_article for given URLs then submit_digest."""

    async def _side_effect(*, agent, message, user_id):
        tools_by_name = {}
        for tool_fn in agent.tools:
            name = getattr(tool_fn, "__name__", "")
            tools_by_name[name] = tool_fn
        for url in fetch_urls:
            await tools_by_name["fetch_article"](url)
        await tools_by_name["submit_digest"](digest_text, used_ids)
        return "done"

    return AsyncMock(side_effect=_side_effect)


def _fake_run_agent_text_with_searches(digest_text: str, used_ids: str, queries: list[str]):
    """Return a mock that calls search_web for given queries then submit_digest."""

    async def _side_effect(*, agent, message, user_id):
        tools_by_name = {}
        for tool_fn in agent.tools:
            name = getattr(tool_fn, "__name__", "")
            tools_by_name[name] = tool_fn
        for q in queries:
            await tools_by_name["search_web"](q)
        await tools_by_name["submit_digest"](digest_text, used_ids)
        return "done"

    return AsyncMock(side_effect=_side_effect)


@pytest.mark.asyncio
async def test_write_digest_returns_composition_with_text_and_item_ids() -> None:
    item_a = str(uuid.uuid4())
    item_b = str(uuid.uuid4())
    digest_body = f"Digest {uuid.uuid4().hex[:8]}"

    with patch(
        f"{_WRITER_MODULE}.run_agent_text",
        new=_fake_run_agent_text_with_submit(digest_body, f"{item_a},{item_b}"),
    ):
        from news_service.agents.digest.writer import write_digest

        result = await write_digest(
            items_text=f"[ID: {item_a}] Headline: A\n\n[ID: {item_b}] Headline: B",
            user_spec="## Topic\nAI \u043d\u043e\u0432\u043e\u0441\u0442\u0438",
            digest_language="ru",
            format_instructions="\u043a\u0440\u0430\u0442\u043a\u043e",
            recent_digest_summaries="",
        )

    assert result.digest_text == digest_body, (
        "write_digest did not return the submitted digest text"
    )
    assert item_a in result.used_item_ids, (
        "write_digest did not include first item ID in used_item_ids"
    )
    assert item_b in result.used_item_ids, (
        "write_digest did not include second item ID in used_item_ids"
    )


@pytest.mark.asyncio
async def test_write_digest_raises_when_agent_does_not_submit() -> None:
    with patch(
        f"{_WRITER_MODULE}.run_agent_text",
        new=_fake_run_agent_text_no_submit(),
    ):
        from news_service.agents.digest.writer import write_digest

        with pytest.raises(RuntimeError, match="submit_digest"):
            await write_digest(
                items_text="[ID: x] Headline: Y",
                user_spec="## Topic\nTest",
                digest_language="en",
                format_instructions="brief",
                recent_digest_summaries="",
            )


@pytest.mark.asyncio
async def test_write_digest_respects_fetch_budget() -> None:
    item_id = str(uuid.uuid4())
    digest_body = f"Digest {uuid.uuid4().hex[:8]}"
    urls = [f"https://example.com/{i}" for i in range(10)]

    async def _side_effect(*, agent, message, user_id):
        tools_by_name = {}
        for tool_fn in agent.tools:
            name = getattr(tool_fn, "__name__", "")
            tools_by_name[name] = tool_fn

        results = []
        for url in urls:
            r = await tools_by_name["fetch_article"](url)
            results.append(r)
        await tools_by_name["submit_digest"](digest_body, item_id)
        return "done"

    with (
        patch(f"{_WRITER_MODULE}.run_agent_text", new=AsyncMock(side_effect=_side_effect)),
        patch(f"{_WRITER_MODULE}.settings") as mock_settings,
    ):
        mock_settings.writer_max_article_fetches = 2
        mock_settings.writer_max_web_searches = 1
        mock_settings.writer_article_fetch_timeout_seconds = 5.0
        mock_settings.writer_article_max_chars = 500
        mock_settings.proxy_url = None
        mock_settings.litellm_model = "openai/gpt-test"

        from news_service.agents.digest.writer import write_digest

        result = await write_digest(
            items_text=f"[ID: {item_id}] Headline: Z",
            user_spec="## Topic\nBudget test \u00e9\u00e8",
            digest_language="en",
            format_instructions="brief",
            recent_digest_summaries="",
        )

    assert result.digest_text == digest_body, (
        "write_digest did not return the expected digest text after budget enforcement"
    )


@pytest.mark.asyncio
async def test_write_digest_respects_search_budget() -> None:
    item_id = str(uuid.uuid4())
    digest_body = f"Digest {uuid.uuid4().hex[:8]}"
    queries = [f"query-{uuid.uuid4().hex[:4]}" for _ in range(8)]

    async def _side_effect(*, agent, message, user_id):
        tools_by_name = {}
        for tool_fn in agent.tools:
            name = getattr(tool_fn, "__name__", "")
            tools_by_name[name] = tool_fn

        results = []
        for q in queries:
            r = await tools_by_name["search_web"](q)
            results.append(r)
        await tools_by_name["submit_digest"](digest_body, item_id)
        return "done"

    with (
        patch(f"{_WRITER_MODULE}.run_agent_text", new=AsyncMock(side_effect=_side_effect)),
        patch(f"{_WRITER_MODULE}._search_web", new=AsyncMock(return_value="search results")),
        patch(f"{_WRITER_MODULE}.settings") as mock_settings,
    ):
        mock_settings.writer_max_article_fetches = 1
        mock_settings.writer_max_web_searches = 2
        mock_settings.writer_article_fetch_timeout_seconds = 5.0
        mock_settings.writer_article_max_chars = 500
        mock_settings.proxy_url = None
        mock_settings.litellm_model = "openai/gpt-test"

        from news_service.agents.digest.writer import write_digest

        result = await write_digest(
            items_text=f"[ID: {item_id}] Headline: Z",
            user_spec="## Topic\nSearch budget \u00fc\u00f6\u00e4",
            digest_language="de",
            format_instructions="kurz",
            recent_digest_summaries="",
        )

    assert result.digest_text == digest_body, (
        "write_digest did not return the expected digest text after search budget enforcement"
    )


@pytest.mark.asyncio
async def test_write_digest_includes_recent_summaries_in_prompt() -> None:
    item_id = str(uuid.uuid4())
    digest_body = f"Digest {uuid.uuid4().hex[:8]}"
    recent_summaries = "Recent digests:\n- Apr 15: GPT-5 release, NIST AI framework"
    captured_message = None

    async def _capture_message(*, agent, message, user_id):
        nonlocal captured_message
        captured_message = message
        for tool_fn in agent.tools:
            if getattr(tool_fn, "__name__", "") == "submit_digest":
                await tool_fn(digest_body, item_id)
                return "done"
        return "done"

    with patch(f"{_WRITER_MODULE}.run_agent_text", new=AsyncMock(side_effect=_capture_message)):
        from news_service.agents.digest.writer import write_digest

        await write_digest(
            items_text=f"[ID: {item_id}] Headline: Something",
            user_spec="## Topic\nAI news",
            digest_language="en",
            format_instructions="brief",
            recent_digest_summaries=recent_summaries,
        )

    assert captured_message is not None, (
        "run_agent_text was not called, cannot verify message content"
    )
    assert "GPT-5 release" in captured_message, (
        "write_digest did not include recent_digest_summaries in the agent prompt"
    )


@pytest.mark.asyncio
async def test_write_digest_incorporates_feedback_on_revision() -> None:
    item_id = str(uuid.uuid4())
    digest_body = f"Revised {uuid.uuid4().hex[:8]}"
    feedback_text = f"Too verbose, shorten section about {uuid.uuid4().hex[:6]}"
    captured_message = None

    async def _capture_message(*, agent, message, user_id):
        nonlocal captured_message
        captured_message = message
        for tool_fn in agent.tools:
            if getattr(tool_fn, "__name__", "") == "submit_digest":
                await tool_fn(digest_body, item_id)
                return "done"
        return "done"

    with patch(f"{_WRITER_MODULE}.run_agent_text", new=AsyncMock(side_effect=_capture_message)):
        from news_service.agents.digest.writer import write_digest

        await write_digest(
            items_text=f"[ID: {item_id}] Headline: X",
            user_spec="## Topic\nTest \u00e9",
            digest_language="fr",
            format_instructions="r\u00e9sum\u00e9 bref",
            recent_digest_summaries="",
            feedback=feedback_text,
        )

    assert captured_message is not None, (
        "run_agent_text was not called, cannot verify feedback inclusion"
    )
    assert feedback_text in captured_message, (
        "write_digest did not include judge feedback in the revision prompt"
    )


@pytest.mark.asyncio
async def test_fetch_article_extracts_text_from_html() -> None:
    body_content = f"Important news about {uuid.uuid4().hex[:8]}"
    html = _random_html(body_content)
    item_id = str(uuid.uuid4())
    fetched_texts: list[str] = []

    async def _side_effect(*, agent, message, user_id):
        tools_by_name = {}
        for tool_fn in agent.tools:
            name = getattr(tool_fn, "__name__", "")
            tools_by_name[name] = tool_fn
        text = await tools_by_name["fetch_article"]("https://example.com/article")
        fetched_texts.append(text)
        await tools_by_name["submit_digest"]("digest", item_id)
        return "done"

    fake_response = AsyncMock()
    fake_response.text = html
    fake_response.status_code = 200
    fake_response.raise_for_status = lambda: None

    fake_client = AsyncMock()
    fake_client.get = AsyncMock(return_value=fake_response)
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(f"{_WRITER_MODULE}.run_agent_text", new=AsyncMock(side_effect=_side_effect)),
        patch(f"{_WRITER_MODULE}.httpx.AsyncClient", return_value=fake_client),
    ):
        from news_service.agents.digest.writer import write_digest

        await write_digest(
            items_text=f"[ID: {item_id}] Headline: Test",
            user_spec="## Topic\nNews",
            digest_language="en",
            format_instructions="brief",
            recent_digest_summaries="",
        )

    assert len(fetched_texts) == 1, "fetch_article was not called exactly once"
    assert body_content in fetched_texts[0], (
        "fetch_article did not extract the body text from the HTML"
    )
    assert "skip" not in fetched_texts[0], "fetch_article did not strip nav elements from the HTML"


@pytest.mark.asyncio
async def test_fetch_article_truncates_long_content() -> None:
    long_body = "W" * 10_000
    html = _random_html(long_body)
    item_id = str(uuid.uuid4())
    fetched_texts: list[str] = []

    async def _side_effect(*, agent, message, user_id):
        tools_by_name = {}
        for tool_fn in agent.tools:
            name = getattr(tool_fn, "__name__", "")
            tools_by_name[name] = tool_fn
        text = await tools_by_name["fetch_article"]("https://example.com/long")
        fetched_texts.append(text)
        await tools_by_name["submit_digest"]("digest", item_id)
        return "done"

    fake_response = AsyncMock()
    fake_response.text = html
    fake_response.status_code = 200
    fake_response.raise_for_status = lambda: None

    fake_client = AsyncMock()
    fake_client.get = AsyncMock(return_value=fake_response)
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(f"{_WRITER_MODULE}.run_agent_text", new=AsyncMock(side_effect=_side_effect)),
        patch(f"{_WRITER_MODULE}.httpx.AsyncClient", return_value=fake_client),
        patch(f"{_WRITER_MODULE}.settings") as mock_settings,
    ):
        mock_settings.writer_max_article_fetches = 5
        mock_settings.writer_max_web_searches = 2
        mock_settings.writer_article_fetch_timeout_seconds = 5.0
        mock_settings.writer_article_max_chars = 500
        mock_settings.proxy_url = None
        mock_settings.litellm_model = "openai/gpt-test"

        from news_service.agents.digest.writer import write_digest

        await write_digest(
            items_text=f"[ID: {item_id}] Headline: Long",
            user_spec="## Topic\nTest",
            digest_language="en",
            format_instructions="brief",
            recent_digest_summaries="",
        )

    assert len(fetched_texts) == 1, "fetch_article was not called exactly once"
    assert len(fetched_texts[0]) <= 500, (
        "fetch_article did not truncate content to writer_article_max_chars"
    )


@pytest.mark.asyncio
async def test_fetch_article_handles_unreachable_url() -> None:
    item_id = str(uuid.uuid4())
    fetched_texts: list[str] = []

    async def _side_effect(*, agent, message, user_id):
        tools_by_name = {}
        for tool_fn in agent.tools:
            name = getattr(tool_fn, "__name__", "")
            tools_by_name[name] = tool_fn
        text = await tools_by_name["fetch_article"]("https://nonexistent.invalid/page")
        fetched_texts.append(text)
        await tools_by_name["submit_digest"]("digest", item_id)
        return "done"

    fake_client = AsyncMock()
    fake_client.get = AsyncMock(side_effect=ConnectionError("DNS resolution failed"))
    fake_client.__aenter__ = AsyncMock(return_value=fake_client)
    fake_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(f"{_WRITER_MODULE}.run_agent_text", new=AsyncMock(side_effect=_side_effect)),
        patch(f"{_WRITER_MODULE}.httpx.AsyncClient", return_value=fake_client),
    ):
        from news_service.agents.digest.writer import write_digest

        await write_digest(
            items_text=f"[ID: {item_id}] Headline: Unreachable",
            user_spec="## Topic\nTest \u00e9\u00e8\u00ea",
            digest_language="en",
            format_instructions="brief",
            recent_digest_summaries="",
        )

    assert len(fetched_texts) == 1, "fetch_article was not called exactly once"
    assert "Failed to fetch" in fetched_texts[0], (
        "fetch_article did not return an error message for an unreachable URL"
    )
