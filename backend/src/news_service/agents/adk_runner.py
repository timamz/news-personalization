"""Shared ADK runner helper — eliminates InMemorySessionService boilerplate.

All ADK agents (conversational, finder, writer, reflector, verifier) use this
helper. It wraps the Agent + InMemorySessionService + Runner ceremony into a
single function call.

ADK is used for agents that need multi-turn tool-calling loops. Single-shot
structured output agents (judge, batch_assessor, orchestrator) use direct
chat_completion() with response_format instead, because ADK does not support
Pydantic structured output (response_format).

For multi-turn conversational flows, callers pass ``conversation_history`` and
this helper pre-populates the in-memory session with one ``Event`` per prior
turn BEFORE invoking the runner. The model then receives the prior turns as
real ``messages[]`` (alternating user/model roles) instead of as a flat text
block embedded in the system instruction. That matches how Anthropic / OpenAI
SDKs and the underlying chat-completions APIs expect dialogue state to be
delivered, and dramatically improves coreference / pronoun resolution on
small models.
"""

import time
import uuid
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import Agent
from google.adk.agents.run_config import RunConfig
from google.adk.events import Event
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from news_service.core.llm import thinking_kwargs


def make_adk_model(model: str, *, reasoning: bool | None = None) -> LiteLlm:
    """Instantiate the ADK ``LiteLlm`` wrapper with optional reasoning toggle.

    Centralizes the construction so every agent goes through the same
    ``thinking_kwargs`` translation. Pass ``reasoning=True`` for agents
    that benefit from chain-of-thought (digest writer, judges, reflector,
    discovery orchestrator, event verifier) and ``reasoning=False`` for
    latency-sensitive or high-volume call sites (conversational agent,
    source finder, batch event assessor).
    """
    extra = thinking_kwargs(model, reasoning)
    return LiteLlm(model=model, **extra)


def _make_history_event(turn: dict[str, Any], agent_name: str) -> Event | None:
    """Translate one stored ``{role, content}`` dict into an ADK ``Event``.

    Returns ``None`` for unknown roles or empty content so we never feed the
    model an empty turn (which some providers reject with a 400). The role
    mapping is the standard one: stored ``"user"`` becomes a user-authored
    Event with ``content.role="user"``; stored ``"assistant"`` becomes an
    Event authored by this agent with ``content.role="model"`` (the value
    Gemini / google-genai expects for the assistant side).
    """
    role = turn.get("role")
    text = (turn.get("content") or "").strip()
    if not text:
        return None
    if role == "user":
        author = "user"
        content_role = "user"
    elif role == "assistant":
        author = agent_name
        content_role = "model"
    else:
        return None
    content = types.Content(role=content_role, parts=[types.Part(text=text)])
    return Event(
        author=author,
        content=content,
        invocation_id=uuid.uuid4().hex,
        id=uuid.uuid4().hex,
        timestamp=time.time(),
    )


async def run_agent(
    *,
    agent: Agent,
    message: str,
    user_id: str = "system",
    max_llm_calls: int | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Run an ADK agent and yield events as they happen.

    Creates a fresh in-memory session per invocation; the session's only job
    is to carry prior turns from ``conversation_history`` into the LLM call.
    No state survives between invocations.

    ``max_llm_calls`` caps the number of LLM turns inside the ADK loop.
    Defaults to ADK's own default (500) when left None. Agents that can
    safely be forced to finish early (e.g. the digest writer) should pass
    an explicit smaller cap to stop runaway tool-calling loops.

    ``conversation_history`` is an ordered list of ``{"role": "user"|"assistant",
    "content": str}`` dicts representing the prior turns in this dialogue.
    Each is appended to the in-memory session before the runner starts so the
    LLM receives a proper alternating ``messages[]`` array. Pass an empty list
    or omit for single-shot agents.

    Yields dicts with:
      {"type": "tool_call", "name": str, "args": dict}
      {"type": "tool_result", "name": str, "result": str}
      {"type": "final_response", "text": str}
    """
    session_service = InMemorySessionService()
    run_id = uuid.uuid4().hex[:12]
    session = await session_service.create_session(
        app_name=agent.name,
        user_id=user_id,
        session_id=run_id,
    )

    if conversation_history:
        for turn in conversation_history:
            event = _make_history_event(turn, agent.name)
            if event is not None:
                await session_service.append_event(session, event)

    runner = Runner(
        agent=agent,
        app_name=agent.name,
        session_service=session_service,
    )

    content = types.Content(role="user", parts=[types.Part(text=message)])

    run_kwargs: dict[str, Any] = {
        "user_id": user_id,
        "session_id": run_id,
        "new_message": content,
    }
    if max_llm_calls is not None:
        run_kwargs["run_config"] = RunConfig(max_llm_calls=max_llm_calls)

    async for event in runner.run_async(**run_kwargs):
        if event.is_final_response():
            text = ""
            if event.content and event.content.parts:
                text = event.content.parts[0].text or ""
            yield {"type": "final_response", "text": text}
            break

        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    yield {
                        "type": "tool_call",
                        "name": part.function_call.name,
                        "args": dict(part.function_call.args) if part.function_call.args else {},
                    }
                if hasattr(part, "function_response") and part.function_response:
                    yield {
                        "type": "tool_result",
                        "name": part.function_response.name,
                        "result": str(part.function_response.response),
                    }


async def run_agent_text(
    *,
    agent: Agent,
    message: str,
    user_id: str = "system",
    max_llm_calls: int | None = None,
    conversation_history: list[dict[str, Any]] | None = None,
) -> str:
    """Convenience wrapper: run an agent and return the final text response."""
    async for event in run_agent(
        agent=agent,
        message=message,
        user_id=user_id,
        max_llm_calls=max_llm_calls,
        conversation_history=conversation_history,
    ):
        if event["type"] == "final_response":
            return event["text"]
    return ""
