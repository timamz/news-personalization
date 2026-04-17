"""ADK agent wiring and streaming runner for the conversational agent."""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.genai import types
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from news_service.agents.adk_runner import run_agent, run_agent_text
from news_service.agents.conversational.helpers import (
    _load_subscription_summaries,
    _status_for_tool_call,
)
from news_service.agents.conversational.prompt import _build_instruction
from news_service.agents.conversational.tools import build_tools
from news_service.core.config import get_settings
from news_service.db.session import async_session_factory
from news_service.models.user import User
from news_service.schemas.conversation import AgentTurnOutput

logger = logging.getLogger(__name__)
settings = get_settings()


def create_conversational_agent(
    *,
    db_session: AsyncSession,
    user: User,
    conversation_summary: str,
    user_language: str | None = None,
    conversation_history: list[dict] | None = None,
    status_queue: asyncio.Queue[dict[str, Any]] | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    subscription_summaries: list[str] | None = None,
    compacted_log: list[str] | None = None,
) -> tuple[Agent, dict[str, Any]]:
    """Build a fresh ADK agent bound to this turn's DB session and user.

    Returns the agent and a shared_state dict. All mutation tools open their
    own DB session via session_factory so the model can emit parallel tool
    calls safely. ``shared_state['scenario_close_summary']`` is set when
    the agent calls close_scenario; the caller uses that signal to compact
    the hot transcript after the turn.
    """
    scoped_factory = session_factory or async_session_factory
    shared_state: dict[str, Any] = {
        "status": "in_progress",
        "created_subscription_id": None,
        "scenario_close_summary": None,
    }

    tools = build_tools(
        user=user,
        db_session=db_session,
        scoped_factory=scoped_factory,
        shared_state=shared_state,
    )

    instruction = _build_instruction(
        conversation_summary=conversation_summary,
        user_language=user_language or user.language,
        user_timezone=user.timezone,
        conversation_history=conversation_history,
        subscription_summaries=subscription_summaries,
        compacted_log=compacted_log,
        has_onboarded=bool(getattr(user, "has_onboarded", False)),
    )

    agent = Agent(
        name="conversational_agent",
        model=LiteLlm(model=settings.litellm_model),
        instruction=instruction,
        tools=tools,
        generate_content_config=types.GenerateContentConfig(temperature=0.2),
    )
    _ = status_queue  # reserved for future per-tool UI events; not used by tools.
    return agent, shared_state


async def run_conversational_turn(
    *,
    db_session: AsyncSession,
    user: User,
    user_message: str,
    conversation_summary: str,
    user_language: str | None = None,
) -> dict[str, Any]:
    """Run a single non-streaming turn and return a simple result dict.

    Used by tests and non-streaming callers.
    """
    agent, shared_state = create_conversational_agent(
        db_session=db_session,
        user=user,
        conversation_summary=conversation_summary,
        user_language=user_language,
    )
    agent_message = await run_agent_text(
        agent=agent,
        message=user_message,
        user_id=str(user.id),
    )
    return {
        "agent_message": agent_message,
        "created_subscription_id": shared_state["created_subscription_id"],
    }


async def run_conversation_turn_streaming(
    messages: list[dict],
    *,
    db_session: AsyncSession,
    user: User,
    conversation_summary: str,
    user_language: str | None = None,
    compacted_log: list[str] | None = None,
) -> AsyncGenerator[dict[str, Any], None]:
    """Streaming variant: yields status events, then one final done event.

    The done event carries the agent's shared_state so the caller can
    react to ``scenario_close_summary`` and compact the hot transcript.

    Events:
      {"event": "status", "status_key": ..., ...kwargs}
      {"event": "done", "output": {...}, "new_messages": [...], "shared_state": {...}}
      {"event": "error", "detail": "..."}
    """
    previous_messages = messages[:-1] if len(messages) > 1 else []
    current_message = messages[-1]["content"] if messages else ""

    status_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    try:
        subscription_summaries = await _load_subscription_summaries(db_session, user.id)
    except Exception:
        logger.exception("Failed to load subscription summaries; continuing without context")
        subscription_summaries = None

    agent, shared_state = create_conversational_agent(
        db_session=db_session,
        user=user,
        conversation_summary=conversation_summary,
        user_language=user_language,
        conversation_history=previous_messages,
        status_queue=status_queue,
        subscription_summaries=subscription_summaries,
        compacted_log=compacted_log,
    )

    agent_text = ""
    try:
        async for event in run_agent(
            agent=agent,
            message=current_message,
            user_id=str(user.id),
        ):
            while not status_queue.empty():
                yield status_queue.get_nowait()
            if event["type"] == "tool_call":
                emitted = _status_for_tool_call(event)
                if emitted is not None:
                    yield emitted
            elif event["type"] == "final_response":
                agent_text = event["text"]
        while not status_queue.empty():
            yield status_queue.get_nowait()
    except Exception as exc:
        logger.exception("Conversational agent streaming failed")
        yield {"event": "error", "detail": f"Agent error: {exc}"}
        return

    output = AgentTurnOutput(
        message=agent_text,
        status=shared_state["status"],
    )
    yield {
        "event": "done",
        "output": output.model_dump(),
        "new_messages": [{"role": "assistant", "content": agent_text}],
        "shared_state": {
            "scenario_close_summary": shared_state.get("scenario_close_summary"),
        },
    }
