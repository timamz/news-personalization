"""Tests for the digest pipeline reflector tools."""

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_session() -> AsyncMock:
    session = AsyncMock()
    session.delete = AsyncMock()
    session.add = MagicMock()
    session.flush = AsyncMock()
    return session


def _make_subscription() -> MagicMock:
    sub = MagicMock()
    sub.id = uuid.uuid4()
    sub.user_id = uuid.uuid4()
    sub.user_spec = "Neural network research."
    sub.last_reflected_at = None
    return sub


def _source_ctx(url: str = "https://example.com/feed", user_specified: bool = False) -> MagicMock:
    ctx = MagicMock()
    ctx.source_id = uuid.uuid4()
    ctx.url = url
    ctx.title = "T"
    ctx.is_user_specified = user_specified
    ctx.contribution_count = 0
    ctx.cosine_to_topic = 0.2
    ctx.last_published_at = None
    ctx.days_since_last_published = 40
    return ctx


@pytest.mark.asyncio
async def test_reflector_does_not_remove_user_specified_source() -> None:
    session = _make_session()
    subscription = _make_subscription()
    link = MagicMock()
    link.is_user_specified = True
    link.source_id = uuid.uuid4()

    lookup = MagicMock()
    lookup.scalar_one_or_none.return_value = link
    session.execute = AsyncMock(return_value=lookup)

    with patch(
        "news_service.agents.digest.reflector.run_agent_text",
        new_callable=AsyncMock,
        return_value="Reviewed pipeline. All healthy.",
    ):
        from news_service.agents.digest.reflector import run_reflector

        await run_reflector(
            db_session=session,
            subscription=subscription,
            digest_text="digest",
            user_spec=subscription.user_spec,
            quality_scores={},
            trigger_reasons=["Drift"],
            source_contexts=[_source_ctx(user_specified=True)],
            allowed_source_ids={uuid.uuid4()},
            topic_embedding=[0.1] * 1536,
        )

    session.delete.assert_not_called()


@pytest.mark.asyncio
async def test_reflector_returns_shared_state_with_discovery_and_observations() -> None:
    session = _make_session()
    subscription = _make_subscription()
    with patch(
        "news_service.agents.digest.reflector.run_agent_text",
        new_callable=AsyncMock,
        return_value="Coverage is thin. Triggered discovery.",
    ):
        from news_service.agents.digest.reflector import run_reflector

        shared_state = await run_reflector(
            db_session=session,
            subscription=subscription,
            digest_text=f"Digest {uuid.uuid4().hex[:6]}",
            user_spec=subscription.user_spec,
            quality_scores={"relevance": 3, "format_score": 3, "conciseness": 3},
            trigger_reasons=["Drift"],
            source_contexts=[_source_ctx()],
            allowed_source_ids={uuid.uuid4()},
            topic_embedding=[0.1] * 1536,
        )

    assert "discovery_triggered" in shared_state and "observations" in shared_state, (
        "reflector did not expose discovery_triggered and observations in shared state"
    )


@pytest.mark.asyncio
async def test_reflector_prompt_carries_trigger_reasons_and_source_contexts() -> None:
    session = _make_session()
    subscription = _make_subscription()
    reason_text = f"drift {uuid.uuid4().hex[:6]}"
    source_url = f"https://example.com/{uuid.uuid4().hex[:6]}"
    captured: dict[str, str] = {}

    async def _capture(*, agent, message, user_id):
        captured["message"] = message
        return "ok"

    with patch(
        "news_service.agents.digest.reflector.run_agent_text",
        new=AsyncMock(side_effect=_capture),
    ):
        from news_service.agents.digest.reflector import run_reflector

        await run_reflector(
            db_session=session,
            subscription=subscription,
            digest_text="digest",
            user_spec=subscription.user_spec,
            quality_scores={},
            trigger_reasons=[reason_text],
            source_contexts=[_source_ctx(url=source_url)],
            allowed_source_ids={uuid.uuid4()},
            topic_embedding=[0.1] * 1536,
        )

    assert reason_text in captured["message"] and source_url in captured["message"], (
        "reflector prompt did not carry both the trigger reason and the source URL"
    )


@pytest.mark.asyncio
async def test_fetch_source_items_refuses_sources_outside_this_subscription() -> None:
    session = _make_session()
    subscription = _make_subscription()
    allowed_ids = {uuid.uuid4()}
    foreign_source_id = uuid.uuid4()
    assert foreign_source_id not in allowed_ids

    captured_tool: dict[str, str] = {}

    async def _capture(*, agent, message, user_id):
        for tool in agent.tools:
            if getattr(tool, "__name__", "") == "fetch_source_items":
                captured_tool["result"] = await tool(str(foreign_source_id))
                return "ok"
        raise AssertionError("fetch_source_items tool missing")

    with patch(
        "news_service.agents.digest.reflector.run_agent_text",
        new=AsyncMock(side_effect=_capture),
    ):
        from news_service.agents.digest.reflector import run_reflector

        await run_reflector(
            db_session=session,
            subscription=subscription,
            digest_text="digest",
            user_spec=subscription.user_spec,
            quality_scores={},
            trigger_reasons=["Drift"],
            source_contexts=[],
            allowed_source_ids=allowed_ids,
            topic_embedding=[0.1] * 1536,
        )

    assert "not linked to this subscription" in captured_tool["result"], (
        "fetch_source_items did not refuse a source outside the subscription's allow-list"
    )


@pytest.mark.asyncio
async def test_fetch_source_items_returns_items_with_cosine_and_snippets() -> None:
    session = _make_session()
    subscription = _make_subscription()
    source_id = uuid.uuid4()
    allowed_ids = {source_id}

    headline = f"Headline {uuid.uuid4().hex[:6]}"
    body = f"Body prose {uuid.uuid4().hex[:8]} with paragraphs."
    fake_item = SimpleNamespace(
        id=uuid.uuid4(),
        source_id=source_id,
        headline=headline,
        body=body,
        url="http://x.test/a",
        published_at=datetime(2026, 4, 10, tzinfo=UTC),
        embedding=[0.1] * 1536,
    )

    scalars_result = MagicMock()
    scalars_result.all.return_value = [fake_item]
    query_result = MagicMock()
    query_result.scalars.return_value = scalars_result
    session.execute = AsyncMock(return_value=query_result)

    captured_tool: dict[str, str] = {}

    async def _capture(*, agent, message, user_id):
        for tool in agent.tools:
            if getattr(tool, "__name__", "") == "fetch_source_items":
                captured_tool["result"] = await tool(str(source_id), 30, 5)
                return "ok"
        raise AssertionError("fetch_source_items tool missing")

    with patch(
        "news_service.agents.digest.reflector.run_agent_text",
        new=AsyncMock(side_effect=_capture),
    ):
        from news_service.agents.digest.reflector import run_reflector

        await run_reflector(
            db_session=session,
            subscription=subscription,
            digest_text="digest",
            user_spec=subscription.user_spec,
            quality_scores={},
            trigger_reasons=["Drift"],
            source_contexts=[],
            allowed_source_ids=allowed_ids,
            topic_embedding=[0.1] * 1536,
        )

    result = captured_tool["result"]
    assert headline in result and body[:50] in result and "cos=" in result, (
        "fetch_source_items did not return headline, body snippet, and cosine similarity"
    )
