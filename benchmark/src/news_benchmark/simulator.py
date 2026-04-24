"""
Scripted-turn driver for the Conversational Agent.

Feeds a pre-authored list of user messages through
``run_conversation_turn_streaming`` one at a time, carrying the
``ConversationState`` forward so each turn sees the full transcript
plus any ``close_scenario`` compacted log entries the agent emitted.

The helper is intentionally thin. Tests schedule these turns at
specific virtual-clock instants (see FakeClock) -- the scheduler
advances time, this helper runs a single turn, then control returns
so the scheduler can advance to the next scheduled event.

Example:

    from news_benchmark.simulator import run_scripted_turns

    state = ConversationState(user_id=str(user_id), user_language="en")
    async for turn_text in run_scripted_turns(
        state=state,
        user=user,
        db_session=session,
        messages=["Hi, onboard me...", "Trigger my digest now"],
    ):
        print(turn_text)
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable
from typing import Any


async def run_one_turn(
    *,
    state: Any,
    user: Any,
    db_session: Any,
    message: str,
) -> str:
    """Run a single user turn, update ``state`` in place, return agent text.

    Mirrors the ``_drive_turn`` helper used in every S-conv test. Append
    the user message, stream the agent's events, accumulate new messages
    back onto the state, and fold any ``scenario_close_summary`` into
    ``state.compacted_log`` so subsequent turns see the condensed history
    instead of the raw transcript.
    """
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


async def run_scripted_turns(
    *,
    state: Any,
    user: Any,
    db_session: Any,
    messages: Iterable[str],
) -> AsyncIterator[str]:
    """Run every scripted user message in order, yielding each agent reply."""
    for msg in messages:
        yield await run_one_turn(state=state, user=user, db_session=db_session, message=msg)
