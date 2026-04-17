"""Streaming-turn tests for the conversational agent.

Complements test_conversational_agent.py (which covers individual tool
definitions) by exercising the top-level streaming runner.
"""

import logging
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from news_service.agents.conversational import (
    _build_instruction,
    _source_display_name,
    run_conversation_turn_streaming,
)

logging.disable(logging.CRITICAL)

_RUN_AGENT_PATH = "news_service.agents.conversational.run_agent"


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
        raise error
        yield  # noqa: UP028

    return fake


@pytest.mark.asyncio
async def test_streaming_yields_done_event() -> None:
    agent_text = f"What kind of news? {uuid.uuid4().hex[:6]}"
    fake = _fake_streaming_agent([{"type": "final_response", "text": agent_text}])
    db_session = AsyncMock()
    user = _fake_user()

    with patch(_RUN_AGENT_PATH, side_effect=fake):
        events = []
        async for ev in run_conversation_turn_streaming(
            [{"role": "user", "content": f"AI news {uuid.uuid4().hex[:6]}"}],
            db_session=db_session,
            user=user,
            conversation_summary="",
            user_language="en",
        ):
            events.append(ev)

    assert len(events) == 1, "streaming did not yield exactly one event"
    assert events[0]["event"] == "done", "streaming did not yield a done event"


@pytest.mark.asyncio
async def test_streaming_done_event_contains_agent_message() -> None:
    agent_text = f"What kind of schedule? {uuid.uuid4().hex[:6]}"
    fake = _fake_streaming_agent([{"type": "final_response", "text": agent_text}])
    db_session = AsyncMock()
    user = _fake_user()

    with patch(_RUN_AGENT_PATH, side_effect=fake):
        events = []
        async for ev in run_conversation_turn_streaming(
            [{"role": "user", "content": f"AI news {uuid.uuid4().hex[:6]}"}],
            db_session=db_session,
            user=user,
            conversation_summary="",
            user_language="en",
        ):
            events.append(ev)

    assert events[0]["output"]["message"] == agent_text, (
        "streaming done event did not contain expected agent message"
    )


@pytest.mark.asyncio
async def test_streaming_yields_status_event_for_add_source_tool_call() -> None:
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
    db_session = AsyncMock()
    user = _fake_user()

    with patch(_RUN_AGENT_PATH, side_effect=fake):
        events = []
        async for ev in run_conversation_turn_streaming(
            [{"role": "user", "content": f"Add @durov {uuid.uuid4().hex[:6]}"}],
            db_session=db_session,
            user=user,
            conversation_summary="",
            user_language="en",
        ):
            events.append(ev)

    assert events[0]["event"] == "status", (
        "streaming did not yield status event for add_source tool call"
    )
    assert events[0]["status_key"] == "status_adding_source", (
        "status event did not have expected status_key"
    )


@pytest.mark.asyncio
async def test_streaming_tool_call_flow_yields_done_last() -> None:
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
    db_session = AsyncMock()
    user = _fake_user()

    with patch(_RUN_AGENT_PATH, side_effect=fake):
        events = []
        async for ev in run_conversation_turn_streaming(
            [{"role": "user", "content": f"Add @durov {uuid.uuid4().hex[:6]}"}],
            db_session=db_session,
            user=user,
            conversation_summary="",
            user_language="en",
        ):
            events.append(ev)

    assert events[-1]["event"] == "done", (
        "streaming tool call flow did not yield done event as last event"
    )


@pytest.mark.asyncio
async def test_streaming_yields_error_event_on_agent_failure() -> None:
    fake = _fake_streaming_agent_error(RuntimeError(f"Agent crashed {uuid.uuid4().hex[:6]}"))
    db_session = AsyncMock()
    user = _fake_user()

    with patch(_RUN_AGENT_PATH, side_effect=fake):
        events = []
        async for ev in run_conversation_turn_streaming(
            [{"role": "user", "content": f"Test {uuid.uuid4().hex[:6]}"}],
            db_session=db_session,
            user=user,
            conversation_summary="",
        ):
            events.append(ev)

    assert events[0]["event"] == "error", "streaming did not yield error event on agent failure"


def test_source_display_name_telegram_channel() -> None:
    result = _source_display_name(f"https://t.me/s/{uuid.uuid4().hex[:8]}", "telegram_channel")
    assert result.startswith("@"), "telegram display name does not start with @"


def test_source_display_name_reddit_subreddit() -> None:
    subreddit = uuid.uuid4().hex[:8]
    result = _source_display_name(f"https://www.reddit.com/r/{subreddit}/new/", "reddit_subreddit")
    assert result == f"r/{subreddit}", "reddit display name not formatted as r/name"


def test_source_display_name_twitter_account() -> None:
    handle = uuid.uuid4().hex[:8]
    result = _source_display_name(f"https://x.com/{handle}", "twitter_account")
    assert result == f"x.com/{handle}", "twitter display name not formatted as x.com/handle"


def test_build_instruction_includes_user_language() -> None:
    lang = f"lang_{uuid.uuid4().hex[:4]}"
    prompt = _build_instruction(
        conversation_summary="",
        user_language=lang,
        user_timezone=None,
    )
    assert lang in prompt, "instruction does not include user language"


def test_build_instruction_includes_conversation_history() -> None:
    marker = uuid.uuid4().hex[:8]
    history = [{"role": "user", "content": f"previous message {marker}"}]
    prompt = _build_instruction(
        conversation_summary="",
        user_language=None,
        user_timezone=None,
        conversation_history=history,
    )
    assert marker in prompt, "instruction does not include conversation history"


def test_build_instruction_includes_conversation_summary_when_present() -> None:
    marker = f"durable-fact-{uuid.uuid4().hex[:6]}"
    prompt = _build_instruction(
        conversation_summary=marker,
        user_language=None,
        user_timezone=None,
    )
    assert marker in prompt, "instruction does not carry conversation_summary into the prompt"
