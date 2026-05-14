"""Tests for the event verifier ADK agent tools and shared-state flow.

The ADK runtime itself is not exercised here. Instead, ``run_agent_text`` is
patched with a scripted stub that pulls the tool functions from the agent's
``tools`` list and invokes them directly, simulating the agent's decisions
in a deterministic way. This lets us assert on tool-side effects
(shared_state recording, budget enforcement, source_id validation) without
needing a live LLM.
"""

import logging
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from news_service.agents.event import verifier as verifier_module
from news_service.agents.event.verifier import (
    VerifierSourceContext,
    run_event_verifier,
)

logging.disable(logging.CRITICAL)


def _tools_by_name(agent_tools: list) -> dict:
    return {t.__name__: t for t in agent_tools}


def _subscription() -> MagicMock:
    sub = MagicMock()
    sub.id = uuid.uuid4()
    sub.user_id = uuid.uuid4()
    sub.digest_language = "ru"
    return sub


def _source_ctx(source_id: uuid.UUID | None = None) -> VerifierSourceContext:
    return VerifierSourceContext(
        source_id=source_id or uuid.uuid4(),
        url=f"https://src-{uuid.uuid4().hex[:6]}.test/feed",
        title=f"Source {uuid.uuid4().hex[:4]}",
        is_user_specified=False,
        last_published_at=datetime.now(UTC) - timedelta(days=2),
        items_in_window=3,
    )


def _scripted_run_agent_text(script):
    """Build a stub for ``run_agent_text`` that calls scripted tools and returns text.

    ``script`` is a list of ``(tool_name, kwargs)`` pairs to invoke in order.
    """

    async def _stub(*, agent, message, user_id, max_llm_calls=None):  # noqa: ARG001
        tools = _tools_by_name(agent.tools)
        for tool_name, kwargs in script:
            await tools[tool_name](**kwargs)
        return f"scripted run complete ({len(script)} tool calls)"

    return _stub


@pytest.mark.asyncio
async def test_verifier_records_missed_event_when_agent_emits_one(mocker) -> None:
    session = AsyncMock()
    sub = _subscription()
    source_url = f"https://official-{uuid.uuid4().hex[:8]}.test/announcement"

    mocker.patch.object(
        verifier_module,
        "run_agent_text",
        new=_scripted_run_agent_text(
            [
                (
                    "emit_missed_event_tool",
                    {
                        "title": f"Новый анонс {uuid.uuid4().hex[:4]}",
                        "summary": "Official release date",
                        "source_url": source_url,
                        "happened_at": "2026-04-18",
                    },
                )
            ]
        ),
    )

    state = await run_event_verifier(
        db_session=session,
        subscription=sub,
        user_spec="спец",
        history_strings=[],
        source_contexts=[_source_ctx()],
        lookback_days=7,
    )

    assert (
        len(state["missed_events"]) == 1 and state["missed_events"][0].source_url == source_url
    ), "verifier did not record the miss the agent emitted"


@pytest.mark.asyncio
async def test_verifier_queues_discovery_when_agent_calls_trigger(mocker) -> None:
    session = AsyncMock()
    sub = _subscription()
    reason = f"event sub: missed {uuid.uuid4().hex[:6]} because source X did not cover it"

    mocker.patch.object(
        verifier_module,
        "run_agent_text",
        new=_scripted_run_agent_text([("trigger_source_discovery_tool", {"reason": reason})]),
    )

    state = await run_event_verifier(
        db_session=session,
        subscription=sub,
        user_spec="спец",
        history_strings=[],
        source_contexts=[_source_ctx()],
        lookback_days=7,
    )

    assert state["discovery_reasons"] == [reason], (
        "verifier did not collect the discovery reason the agent supplied"
    )


@pytest.mark.asyncio
async def test_verifier_rejects_empty_miss_title_and_url(mocker) -> None:
    session = AsyncMock()
    sub = _subscription()

    mocker.patch.object(
        verifier_module,
        "run_agent_text",
        new=_scripted_run_agent_text(
            [
                (
                    "emit_missed_event_tool",
                    {"title": "  ", "summary": "x", "source_url": "", "happened_at": ""},
                )
            ]
        ),
    )

    state = await run_event_verifier(
        db_session=session,
        subscription=sub,
        user_spec="спец",
        history_strings=[],
        source_contexts=[_source_ctx()],
        lookback_days=7,
    )

    assert state["missed_events"] == [], "verifier accepted a miss with empty title/source_url"


@pytest.mark.asyncio
async def test_verifier_enforces_search_budget(mocker, monkeypatch) -> None:
    session = AsyncMock()
    sub = _subscription()
    monkeypatch.setattr(verifier_module.settings, "event_verifier_max_searches", 2)
    search_spy = mocker.patch.object(
        verifier_module,
        "search_web",
        new=AsyncMock(return_value="canned results"),
    )

    script = [("web_search_tool", {"query": f"q{i}"}) for i in range(4)]
    mocker.patch.object(
        verifier_module,
        "run_agent_text",
        new=_scripted_run_agent_text(script),
    )

    state = await run_event_verifier(
        db_session=session,
        subscription=sub,
        user_spec="спец",
        history_strings=[],
        source_contexts=[_source_ctx()],
        lookback_days=7,
    )

    assert search_spy.call_count == 2 and state["search_budget_used"] == 4, (
        "verifier did not stop invoking search_web once budget was exhausted"
    )


@pytest.mark.asyncio
async def test_verifier_fetch_source_items_refuses_foreign_source(mocker) -> None:
    session = AsyncMock()
    sub = _subscription()
    allowed_ctx = _source_ctx()
    foreign_source_id = uuid.uuid4()

    captured: list[str] = []

    async def _stub(*, agent, message, user_id, max_llm_calls=None):  # noqa: ARG001
        tools = _tools_by_name(agent.tools)
        captured.append(
            await tools["fetch_source_items_tool"](
                source_id=str(foreign_source_id),
                since_days_ago=5,
                limit=5,
            )
        )
        return "done"

    mocker.patch.object(verifier_module, "run_agent_text", new=_stub)

    await run_event_verifier(
        db_session=session,
        subscription=sub,
        user_spec="спец",
        history_strings=[],
        source_contexts=[allowed_ctx],
        lookback_days=7,
    )

    assert "not linked" in captured[0].lower(), (
        "fetch_source_items did not refuse a source_id outside the sub's pool"
    )


@pytest.mark.asyncio
async def test_verifier_emit_status_collects_messages(mocker) -> None:
    session = AsyncMock()
    sub = _subscription()
    msg = f"Прогресс {uuid.uuid4().hex[:5]}"

    mocker.patch.object(
        verifier_module,
        "run_agent_text",
        new=_scripted_run_agent_text([("emit_status_tool", {"message": msg})]),
    )

    state = await run_event_verifier(
        db_session=session,
        subscription=sub,
        user_spec="спец",
        history_strings=[],
        source_contexts=[_source_ctx()],
        lookback_days=7,
    )

    assert state["status_messages"] == [msg], (
        "verifier did not collect the emit_status message the agent supplied"
    )
