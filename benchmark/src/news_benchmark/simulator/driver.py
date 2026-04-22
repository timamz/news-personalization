"""
Persona LLM helper for the user simulator.

Given a running transcript and the list of goals still unmet, asks the
persona LLM for the next user utterance. The orchestrator composes this
with its scripted-turn clock-advancing loop to drive a multi-turn
conversation against the production Conversational Agent without
forcing the agent to close everything in a single turn.
"""

from __future__ import annotations

from typing import Any

import litellm

from news_benchmark.scenarios.base import Persona, SubscriptionGoal
from news_benchmark.simulator.prompts import render_system


async def next_user_message(
    *,
    persona: Persona,
    remaining_goals: list[SubscriptionGoal],
    max_turns: int,
    simulator_model: str,
    simulator_temperature: float,
    transcript: list[dict[str, Any]],
) -> str:
    """Ask the persona LLM for the next user utterance given the running transcript.

    ``transcript`` is the orchestrator's running list of
    ``{"speaker": "user"|"agent", "text": str}`` entries (same shape it
    feeds to ``run_conversation_turn_streaming``).
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": render_system(persona, remaining_goals, max_turns)},
    ]
    for t in transcript:
        role = "user" if t["speaker"] == "user" else "assistant"
        messages.append({"role": role, "content": t["text"]})

    resp = await litellm.acompletion(
        model=simulator_model,
        messages=messages,
        temperature=simulator_temperature,
        max_tokens=300,
    )
    return (resp["choices"][0]["message"]["content"] or "").strip()
