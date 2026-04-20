"""
LLM-as-judge rubric for conversation quality.

Per closed scenario: task_completion (binary) and info_surfaced (1-5).
task_completion is additionally checked deterministically against
goal_checker output in action_correctness.
"""

from __future__ import annotations

import json

import litellm
from pydantic import BaseModel, Field

from news_benchmark.tagging import agent_tag


class ConversationScores(BaseModel):
    info_surfaced: int = Field(ge=1, le=5)
    tone_matches_persona: int = Field(ge=1, le=5)
    rationale: str


SYSTEM = (
    "You are a quality judge for a conversational agent that sets up news "
    "subscriptions. Given the user's persona + goals and the full transcript, "
    "rate how well the agent surfaced what it did in terms the persona would "
    "understand, and whether its tone matched the persona. Return strict JSON."
)


async def judge_conversation(
    *,
    persona_description: str,
    goals_description: str,
    transcript: str,
    judge_model: str,
) -> ConversationScores:
    user = (
        f"PERSONA:\n{persona_description}\n\n"
        f"GOALS:\n{goals_description}\n\n"
        f"TRANSCRIPT:\n{transcript}\n\n"
        "Return JSON with info_surfaced (1-5), tone_matches_persona (1-5), rationale."
    )
    async with agent_tag("judge.conversation"):
        resp = await litellm.acompletion(
            model=judge_model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=400,
            response_format={"type": "json_object"},
        )
    raw = resp["choices"][0]["message"]["content"] or "{}"
    return ConversationScores.model_validate(json.loads(raw))
