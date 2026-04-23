"""
S-conv close_scenario stays silent: negative half of the close_scenario pair.

Paired with ``test_s_conv_close_scenario_fires.py``. Test B proves the
negative signal -- when the user's request is underspecified ("I want a
digest.") the agent must respond with a clarifying question and leave
``close_scenario`` uninvoked. The scenario is mid-flow; per the agent
prompt, "Do not close a scenario if something is still pending." The
observable signal is ``len(state.compacted_log) == 0``.

Two halves live in separate files because ADK/asyncpg connection state
leaks across pytest-asyncio function-scoped event loops when two
ADK-driving tests share a process. Separate files side-steps the
cross-loop Future error without changing pytest config.

The positive half asserts the agent DOES close on a completed,
acknowledged task. Both halves are required; see that file's docstring.
"""

from __future__ import annotations

import uuid

import pytest
from sqlalchemy import select


async def _drive_turn(state, *, user, db_session, message: str) -> str:
    from news_service.agents.conversational import run_conversation_turn_streaming
    from news_service.schemas.conversation import AgentTurnOutput

    state.messages.append({"role": "user", "content": message})

    agent_text = ""
    async for event in run_conversation_turn_streaming(
        state.messages,
        db_session=db_session,
        user=user,
        conversation_summary=user.conversation_summary or "",
        user_language=state.user_language,
        compacted_log=list(state.compacted_log),
    ):
        if event["event"] == "done":
            output = AgentTurnOutput.model_validate(event["output"])
            agent_text = output.message
            state.messages.extend(event["new_messages"])
            shared = event.get("shared_state") or {}
            close_summary = shared.get("scenario_close_summary")
            if close_summary:
                state.compacted_log.append(close_summary.strip())
    return agent_text


@pytest.mark.asyncio
async def test_s_conv_close_scenario_stays_silent_on_underspecified_request(world):
    """close_scenario stays silent when the agent is mid-flow asking a clarifying question."""
    from news_service.agents.conversational import tools as conv_tools
    from news_service.db.session import async_session_factory
    from news_service.models.failed_task import FailedTask
    from news_service.models.subscription import Subscription
    from news_service.models.user import User
    from news_service.schemas.conversation import ConversationState

    async def _noop_discovery(*_args, **_kwargs):
        return {"status": "skipped", "reason": "S-conv close_scenario test bypass"}

    original_discovery = conv_tools._run_inline_discovery
    conv_tools._run_inline_discovery = _noop_discovery  # type: ignore[assignment]

    try:
        user_id = uuid.uuid4()
        async with async_session_factory() as s:
            s.add(
                User(
                    id=user_id,
                    api_key=f"bench-{user_id.hex}",
                    language="en",
                    timezone="UTC",
                    has_onboarded=False,
                )
            )
            await s.commit()

        state = ConversationState(user_id=str(user_id), user_language="en")

        message = "I want a digest."
        async with async_session_factory() as s:
            user = await s.get(User, user_id)
            assert user is not None, "user disappeared before driving turn"
            agent_text = await _drive_turn(state, user=user, db_session=s, message=message)

        assert len(state.compacted_log) == 0, (
            f"close_scenario fired on an underspecified mid-flow turn: "
            f"compacted_log has {len(state.compacted_log)} entries, expected 0. "
            f"Agent said: {agent_text!r}. Compacted: {state.compacted_log!r}"
        )

        async with async_session_factory() as s:
            rows = await s.execute(
                select(Subscription).where(
                    Subscription.user_id == user_id,
                    Subscription.is_active.is_(True),
                )
            )
            subs = list(rows.scalars().all())
        assert len(subs) == 0, (
            f"no subscription should exist after an underspecified request, "
            f"got {len(subs)}. Agent said: {agent_text!r}"
        )

        assert agent_text.strip(), (
            "agent reply should be non-empty (expected a clarifying question), "
            "got empty string"
        )

        async with async_session_factory() as s:
            rows = await s.execute(select(FailedTask))
            failed = list(rows.scalars().all())
        assert not failed, f"expected 0 failed_tasks rows, got {len(failed)}: {failed!r}"
    finally:
        conv_tools._run_inline_discovery = original_discovery  # type: ignore[assignment]
