"""
LLM-as-judge rubric for event notifications.

One rubric per delivered notification. Computed dedup_correctness is
checked deterministically across the run rather than by the LLM.
"""

from __future__ import annotations

import json

import litellm
from pydantic import BaseModel, Field

from news_benchmark.tagging import agent_tag


class EventScores(BaseModel):
    event_relevance: int = Field(ge=1, le=5)
    notification_body_quality: int = Field(ge=1, le=5)
    rationale: str


SYSTEM = (
    "You are a quality judge for event-notification quality. Rate a single "
    "delivered notification against the user's spec. Return strict JSON "
    "with keys event_relevance (1-5), notification_body_quality (1-5), rationale."
)


async def judge_event(
    *,
    user_spec: str,
    notification_body: str,
    source_headline: str,
    judge_model: str,
) -> EventScores:
    user = (
        f"USER_SPEC:\n{user_spec}\n\n"
        f"SOURCE_HEADLINE:\n{source_headline}\n\n"
        f"NOTIFICATION_BODY:\n{notification_body}\n\n"
        "Return the JSON object."
    )
    async with agent_tag("judge.event"):
        resp = await litellm.acompletion(
            model=judge_model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            temperature=0,
            max_tokens=300,
            response_format={"type": "json_object"},
        )
    raw = resp["choices"][0]["message"]["content"] or "{}"
    return EventScores.model_validate(json.loads(raw))
