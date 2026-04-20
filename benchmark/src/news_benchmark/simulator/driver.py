"""
Simulator driver: runs one conversation with the production Conversational Agent.

Opens a session, advances scripted turns from the scenario, and for each
scripted turn (or each time the agent's reply does not yet close the
goal), generates the next user message by calling the persona LLM with
the full transcript. Stops when the simulator emits <END>, every goal is
marked met, or max_turns is reached.

Wired against `run_conversation_turn_streaming` from
news_service.agents.conversational.agent. The driver accumulates the
agent's transcript state across turns exactly like the real streaming
API endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import litellm

from news_benchmark.scenarios.base import ConversationTurn, Persona, SubscriptionGoal
from news_benchmark.simulator.prompts import render_system
from news_benchmark.tagging import agent_tag


@dataclass
class ConversationLog:
    """Full transcript of one simulator-agent conversation."""

    turns: list[dict[str, Any]] = field(default_factory=list)
    goal_status: dict[str, bool] = field(default_factory=dict)
    terminated_reason: str = ""

    def append(self, speaker: str, text: str, meta: dict[str, Any] | None = None) -> None:
        self.turns.append({"speaker": speaker, "text": text, "meta": meta or {}})


@dataclass
class SimulatorDriver:
    """Drives one user-simulator conversation end-to-end."""

    persona: Persona
    goals: list[SubscriptionGoal]
    scripted: list[ConversationTurn]
    max_turns: int
    simulator_model: str
    simulator_temperature: float = 0.7

    async def run(
        self,
        *,
        agent_callable,
        goal_checker,
    ) -> ConversationLog:
        """Run the conversation. `agent_callable` takes a user message and returns
        the agent's final text + any side-effect state. `goal_checker(state)`
        returns an updated {goal_id: met} dict after each agent reply.
        """
        log = ConversationLog()
        remaining = list(self.goals)
        stall = 0
        scripted_idx = 0

        for _turn_idx in range(self.max_turns):
            if scripted_idx < len(self.scripted):
                user_msg = self.scripted[scripted_idx].message
                scripted_idx += 1
            else:
                user_msg = await self._next_user_message(log, remaining)

            if user_msg.strip().endswith("<END>") or user_msg.strip() == "<END>":
                log.terminated_reason = "simulator_end_sentinel"
                log.append("user", user_msg)
                break

            log.append("user", user_msg)
            async with agent_tag("simulator.conversational_call"):
                agent_reply, state = await agent_callable(user_msg)
            log.append("agent", agent_reply, meta={"state": state})

            met = goal_checker(state)
            log.goal_status = met
            previous_remaining = len(remaining)
            remaining = [g for g in self.goals if not met.get(g.goal_id, False)]
            if len(remaining) == previous_remaining:
                stall += 1
            else:
                stall = 0
            if not remaining:
                log.terminated_reason = "all_goals_met"
                break
            if stall >= 3:
                log.terminated_reason = "stall"
                break
        else:
            log.terminated_reason = "turn_budget"

        return log

    async def _next_user_message(
        self,
        log: ConversationLog,
        remaining: list[SubscriptionGoal],
    ) -> str:
        """Ask the persona LLM for the next user utterance."""
        messages: list[dict[str, str]] = [
            {"role": "system", "content": render_system(self.persona, remaining, self.max_turns)},
        ]
        for t in log.turns:
            role = "user" if t["speaker"] == "user" else "assistant"
            messages.append({"role": role, "content": t["text"]})

        resp = await litellm.acompletion(
            model=self.simulator_model,
            messages=messages,
            temperature=self.simulator_temperature,
            max_tokens=300,
        )
        return (resp["choices"][0]["message"]["content"] or "").strip()
