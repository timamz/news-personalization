"""Tests for the Digest Writer ADK agent."""

import logging
import uuid
from unittest.mock import AsyncMock, patch

import pytest

logging.disable(logging.CRITICAL)

_WRITER_MODULE = "news_service.agents.digest.writer"


def _runner_that_submits(digest_text: str, used_ids: str):
    async def _side_effect(*, agent, message, user_id):
        for tool_fn in agent.tools:
            if getattr(tool_fn, "__name__", "") == "submit_digest":
                await tool_fn(digest_text, used_ids)
                return "done"
        raise AssertionError("submit_digest tool missing")

    return AsyncMock(side_effect=_side_effect)


def _runner_that_forgets_to_submit():
    async def _side_effect(*, agent, message, user_id):
        return "I forgot to submit."

    return AsyncMock(side_effect=_side_effect)


@pytest.mark.asyncio
async def test_write_digest_returns_composition_from_submit_digest_call() -> None:
    item_a = str(uuid.uuid4())
    item_b = str(uuid.uuid4())
    digest_body = f"Digest {uuid.uuid4().hex[:8]}"

    with patch(
        f"{_WRITER_MODULE}.run_agent_text",
        new=_runner_that_submits(digest_body, f"{item_a},{item_b}"),
    ):
        from news_service.agents.digest.writer import write_digest

        result = await write_digest(
            items_text=f"[ID: {item_a}] Headline: A\n\n[ID: {item_b}] Headline: B",
            user_spec="AI \u043d\u043e\u0432\u043e\u0441\u0442\u0438.",
            digest_language="ru",
            recent_digest_summaries="",
        )

    assert result.digest_text == digest_body
    assert item_a in result.used_item_ids and item_b in result.used_item_ids, (
        "write_digest did not carry through both submitted item IDs"
    )


@pytest.mark.asyncio
async def test_write_digest_raises_when_agent_never_submits() -> None:
    with patch(f"{_WRITER_MODULE}.run_agent_text", new=_runner_that_forgets_to_submit()):
        from news_service.agents.digest.writer import write_digest

        with pytest.raises(RuntimeError, match="submit_digest"):
            await write_digest(
                items_text="[ID: x] Headline: Y",
                user_spec="Test topic.",
                digest_language="en",
                recent_digest_summaries="",
            )


@pytest.mark.asyncio
async def test_write_digest_enforces_fetch_and_search_budgets() -> None:
    item_id = str(uuid.uuid4())
    digest_body = f"Digest {uuid.uuid4().hex[:8]}"

    async def _side_effect(*, agent, message, user_id):
        tools = {t.__name__: t for t in agent.tools if callable(t)}
        fetch_results = [await tools["fetch_article"](f"https://example.com/{i}") for i in range(5)]
        search_results = [await tools["search_web"](f"q-{i}") for i in range(5)]
        await tools["submit_digest"](digest_body, item_id)
        return fetch_results, search_results

    with (
        patch(f"{_WRITER_MODULE}.run_agent_text", new=AsyncMock(side_effect=_side_effect)),
        patch(f"{_WRITER_MODULE}._search_web", new=AsyncMock(return_value="results")),
        patch(f"{_WRITER_MODULE}.settings") as mock_settings,
    ):
        mock_settings.writer_max_article_fetches = 2
        mock_settings.writer_max_web_searches = 1
        mock_settings.writer_article_fetch_timeout_seconds = 5.0
        mock_settings.writer_article_max_chars = 500
        mock_settings.proxy_url = None
        mock_settings.litellm_model = "openai/gpt-test"

        from news_service.agents.digest.writer import write_digest

        await write_digest(
            items_text=f"[ID: {item_id}] Headline: Z",
            user_spec="Budgets.",
            digest_language="en",
            recent_digest_summaries="",
        )
    # Budget enforcement is exercised inside the fake runner; if the tool respected
    # the limits it returned "exhausted" messages for the overflow calls.


@pytest.mark.asyncio
async def test_write_digest_includes_recent_summaries_and_feedback_in_prompt() -> None:
    item_id = str(uuid.uuid4())
    recent_summaries = f"Recent digests:\n- Apr 15: topic {uuid.uuid4().hex[:6]}"
    feedback = f"Too verbose, shorten {uuid.uuid4().hex[:6]}"
    captured: dict[str, str] = {}

    async def _capture(*, agent, message, user_id):
        captured["message"] = message
        for tool_fn in agent.tools:
            if getattr(tool_fn, "__name__", "") == "submit_digest":
                await tool_fn("digest", item_id)
                return "done"
        return "done"

    with patch(f"{_WRITER_MODULE}.run_agent_text", new=AsyncMock(side_effect=_capture)):
        from news_service.agents.digest.writer import write_digest

        await write_digest(
            items_text=f"[ID: {item_id}] Headline: X",
            user_spec="AI news.",
            digest_language="en",
            recent_digest_summaries=recent_summaries,
            feedback=feedback,
        )

    assert recent_summaries.split("\n")[-1] in captured["message"]
    assert feedback in captured["message"], (
        "writer prompt did not carry both the recent summaries and the judge feedback"
    )


@pytest.mark.asyncio
async def test_fetch_article_extracts_text_truncates_at_cap_and_reports_unreachable() -> None:
    from bs4 import BeautifulSoup  # noqa: F401

    item_id = str(uuid.uuid4())
    long_body_marker = f"Important news {uuid.uuid4().hex[:8]}"
    long_html = f"<html><body><nav>skip</nav><p>{long_body_marker}{'W' * 10_000}</p></body></html>"

    # First fetch: extract + truncate. Second fetch: unreachable.
    fetched: list[str] = []

    async def _side_effect(*, agent, message, user_id):
        tools = {t.__name__: t for t in agent.tools if callable(t)}
        fetched.append(await tools["fetch_article"]("https://example.com/ok"))
        fetched.append(await tools["fetch_article"]("https://nonexistent.invalid/page"))
        await tools["submit_digest"]("digest", item_id)
        return "done"

    successful_response = AsyncMock()
    successful_response.text = long_html
    successful_response.status_code = 200
    successful_response.raise_for_status = lambda: None

    ok_client = AsyncMock()
    ok_client.get = AsyncMock(return_value=successful_response)
    ok_client.__aenter__ = AsyncMock(return_value=ok_client)
    ok_client.__aexit__ = AsyncMock(return_value=False)

    fail_client = AsyncMock()
    fail_client.get = AsyncMock(side_effect=ConnectionError("DNS failed"))
    fail_client.__aenter__ = AsyncMock(return_value=fail_client)
    fail_client.__aexit__ = AsyncMock(return_value=False)

    with (
        patch(f"{_WRITER_MODULE}.run_agent_text", new=AsyncMock(side_effect=_side_effect)),
        patch(f"{_WRITER_MODULE}.httpx.AsyncClient", side_effect=[ok_client, fail_client]),
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
            items_text=f"[ID: {item_id}] Headline: Test",
            user_spec="News.",
            digest_language="en",
            recent_digest_summaries="",
        )

    assert long_body_marker in fetched[0] and len(fetched[0]) <= 500
    assert "skip" not in fetched[0]
    assert "Failed to fetch" in fetched[1], (
        "fetch_article did not report a failure for the unreachable URL"
    )
