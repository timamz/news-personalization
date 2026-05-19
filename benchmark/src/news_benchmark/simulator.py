"""
Scripted-turn driver for the Conversational Agent.

Runs one user message through ``run_conversation_turn_streaming``,
carrying the ``ConversationState`` forward so the next turn sees the
updated transcript plus any compacted ``close_scenario`` entries.
"""

from __future__ import annotations

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
