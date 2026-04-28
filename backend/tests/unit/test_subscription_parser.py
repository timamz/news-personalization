"""Streaming-turn tests for the conversational agent.

Complements test_conversational_agent.py by exercising the top-level
streaming runner: status events for tool calls, the terminal done
event, and error propagation.
"""

import logging
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from news_service.agents.conversational import (
    _source_display_name,
    run_conversation_turn_streaming,
)

logging.disable(logging.CRITICAL)

_RUN_AGENT_PATH = "news_service.agents.conversational.agent.run_agent"


def _fake_user() -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        timezone="Europe/Moscow",
        language="en",
        conversation_summary="",
        has_onboarded=True,
    )


def _fake_streaming_agent(events_to_yield):
    async def fake(*, agent, message, user_id="system"):
        for ev in events_to_yield:
            yield ev

    return fake


def _fake_streaming_agent_error(error):
    async def fake(*, agent, message, user_id="system"):
        for _ in ():
            yield _
        raise error

    return fake


@pytest.mark.asyncio
async def test_streaming_yields_done_event_with_agent_message() -> None:
    agent_text = f"What kind of news? {uuid.uuid4().hex[:6]}"
    fake = _fake_streaming_agent([{"type": "final_response", "text": agent_text}])

    with patch(_RUN_AGENT_PATH, side_effect=fake):
        events = [
            ev
            async for ev in run_conversation_turn_streaming(
                [{"role": "user", "content": f"AI news {uuid.uuid4().hex[:6]}"}],
                db_session=AsyncMock(),
                user=_fake_user(),
                conversation_summary="",
                user_language="en",
            )
        ]

    assert len(events) == 1 and events[0]["event"] == "done"
    assert events[0]["output"]["message"] == agent_text, (
        "done event did not carry the agent's final text"
    )


@pytest.mark.asyncio
async def test_streaming_tool_call_emits_status_then_done() -> None:
    fake = _fake_streaming_agent(
        [
            {
                "type": "tool_call",
                "name": "add_source",
                "args": {
                    "subscription_id": str(uuid.uuid4()),
                    "identifier": "durov",
                    "source_kind": "telegram_channel",
                },
            },
            {"type": "final_response", "text": "Added."},
        ]
    )

    with patch(_RUN_AGENT_PATH, side_effect=fake):
        events = [
            ev
            async for ev in run_conversation_turn_streaming(
                [{"role": "user", "content": f"Add @durov {uuid.uuid4().hex[:6]}"}],
                db_session=AsyncMock(),
                user=_fake_user(),
                conversation_summary="",
                user_language="en",
            )
        ]

    assert events[0]["event"] == "status"
    assert events[0]["status_key"] == "status_adding_source"
    assert events[-1]["event"] == "done"


@pytest.mark.asyncio
async def test_streaming_yields_error_event_on_agent_failure() -> None:
    fake = _fake_streaming_agent_error(RuntimeError(f"Agent crashed {uuid.uuid4().hex[:6]}"))

    with patch(_RUN_AGENT_PATH, side_effect=fake):
        events = [
            ev
            async for ev in run_conversation_turn_streaming(
                [{"role": "user", "content": f"Test {uuid.uuid4().hex[:6]}"}],
                db_session=AsyncMock(),
                user=_fake_user(),
                conversation_summary="",
            )
        ]

    assert events[0]["event"] == "error"


def test_source_display_name_formats_each_kind_correctly() -> None:
    tg_handle = uuid.uuid4().hex[:8]
    sub = uuid.uuid4().hex[:8]
    assert (
        _source_display_name(f"https://t.me/s/{tg_handle}", "telegram_channel") == f"@{tg_handle}"
    )
    assert (
        _source_display_name(f"https://reddit.com/r/{sub}/new/", "reddit_subreddit") == f"r/{sub}"
    )
