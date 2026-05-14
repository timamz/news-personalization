"""Tests for the Digest Writer ADK agent."""

import logging
import uuid
from unittest.mock import AsyncMock, patch

import pytest

logging.disable(logging.CRITICAL)

_WRITER_MODULE = "news_service.agents.digest.writer"


def _runner_that_submits(digest_text: str, used_ids: str):
    async def _side_effect(*, agent, message, user_id, **kwargs):
        for tool_fn in agent.tools:
            if getattr(tool_fn, "__name__", "") == "submit_digest":
                await tool_fn(digest_text, used_ids)
                return "done"
        raise AssertionError("submit_digest tool missing")

    return AsyncMock(side_effect=_side_effect)


def _runner_that_forgets_to_submit():
    async def _side_effect(*, agent, message, user_id, **kwargs):
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

    assert result.digest_text == digest_body and (
        item_a in result.used_item_ids and item_b in result.used_item_ids
    ), "write_digest did not carry through both submitted item IDs"


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
async def test_write_digest_exposes_only_search_fetch_and_submit_tools() -> None:
    item_id = str(uuid.uuid4())
    captured: dict[str, set[str]] = {}

    async def _capture(*, agent, message, user_id, **kwargs):
        captured["tool_names"] = {t.__name__ for t in agent.tools if callable(t)}
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
            recent_digest_summaries="",
        )

    assert captured["tool_names"] == {"search_web", "fetch_page_bounded", "submit_digest"}, (
        "writer must expose search_web, fetch_page_bounded, and submit_digest"
    )


@pytest.mark.asyncio
async def test_write_digest_includes_recent_summaries_and_feedback_in_prompt() -> None:
    item_id = str(uuid.uuid4())
    recent_summaries = f"Recent digests:\n- Apr 15: topic {uuid.uuid4().hex[:6]}"
    feedback = f"Too verbose, shorten {uuid.uuid4().hex[:6]}"
    captured: dict[str, str] = {}

    async def _capture(*, agent, message, user_id, **kwargs):
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

    assert recent_summaries.split("\n")[-1] in captured["message"] and (
        feedback in captured["message"]
    ), "writer prompt did not carry both the recent summaries and the judge feedback"


def test_writer_instruction_demands_verbatim_full_uuids() -> None:
    from news_service.agents.digest.writer import _WRITER_PROMPT

    text = _WRITER_PROMPT.lower()
    assert "verbatim" in text and "[id:" in text and "do not shorten" in text, (
        "writer instruction must tell the model to copy full UUIDs verbatim "
        "from the [ID: ...] header and never shorten them"
    )
