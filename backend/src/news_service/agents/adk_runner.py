"""Shared ADK runner helper — eliminates InMemorySessionService boilerplate.

All ADK agents (conversational, finder, subscription_parser) use this helper
to run agents. It wraps the Agent + InMemorySessionService + Runner ceremony
into a single function call.

ADK is used for agents that need multi-turn tool-calling loops. Single-shot
structured output agents (planner, composer, judge, reflector, batch_assessor,
orchestrator) use direct chat_completion() with response_format instead, because
ADK does not support Pydantic structured output (response_format).
"""

import uuid
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types


async def run_agent(
    *,
    agent: Agent,
    message: str,
    user_id: str = "system",
    streaming: bool = False,
) -> str | AsyncGenerator[dict[str, Any], None]:
    """Run an ADK agent with a single user message.

    Creates a fresh in-memory session per invocation. ADK session state is not
    persisted between calls — conversation history lives in Redis or the caller's
    context, not in ADK sessions. We use ADK purely for its ReAct tool-calling
    loop, not for session persistence.

    Args:
        agent: The ADK Agent to run.
        message: The user message to send.
        user_id: User identifier for the ADK session.
        streaming: When False (default), returns the final text response as str.
            When True, returns an async generator yielding event dicts:
              {"type": "tool_call", "name": str, "args": dict}
              {"type": "tool_result", "name": str, "result": str}
              {"type": "final_response", "text": str}
    """
    if streaming:
        return _run_streaming(agent=agent, message=message, user_id=user_id)
    return await _run_blocking(agent=agent, message=message, user_id=user_id)


async def _setup_runner(agent: Agent, user_id: str) -> tuple[Runner, str]:
    """Create a Runner with a fresh in-memory session. Returns (runner, session_id)."""
    session_service = InMemorySessionService()
    run_id = uuid.uuid4().hex[:12]
    await session_service.create_session(
        app_name=agent.name,
        user_id=user_id,
        session_id=run_id,
    )
    runner = Runner(
        agent=agent,
        app_name=agent.name,
        session_service=session_service,
    )
    return runner, run_id


async def _run_blocking(*, agent: Agent, message: str, user_id: str) -> str:
    runner, run_id = await _setup_runner(agent, user_id)
    content = types.Content(role="user", parts=[types.Part(text=message)])

    response_text = ""
    async for event in runner.run_async(
        user_id=user_id,
        session_id=run_id,
        new_message=content,
    ):
        if event.is_final_response():
            if event.content and event.content.parts:
                response_text = event.content.parts[0].text or ""
            break

    return response_text


async def _run_streaming(
    *, agent: Agent, message: str, user_id: str
) -> AsyncGenerator[dict[str, Any], None]:
    runner, run_id = await _setup_runner(agent, user_id)
    content = types.Content(role="user", parts=[types.Part(text=message)])

    async for event in runner.run_async(
        user_id=user_id,
        session_id=run_id,
        new_message=content,
    ):
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
