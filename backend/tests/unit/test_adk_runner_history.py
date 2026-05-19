"""Behavioral tests for ADK runner conversation-history replay.

These tests exercise the contract that ``run_agent`` must convert a flat
``[{role, content}]`` history into properly authored ADK ``Event`` records
on the in-memory session, so the LLM call sees a real alternating
``messages[]`` array instead of a system-prompt blob. The previous design
embedded the history inside the system instruction, which tanked
coreference resolution on small models.
"""

import logging

import pytest

from news_service.agents.adk_runner import _make_history_event

logging.disable(logging.CRITICAL)


def test_user_history_turn_becomes_event_authored_user_with_user_role_content() -> None:
    event = _make_history_event(
        {"role": "user", "content": "what subscriptions do I have"},
        agent_name="conversational_agent",
    )

    assert (
        event is not None
        and event.author == "user"
        and event.content is not None
        and event.content.role == "user"
    ), (
        "user-side history turns must be Event(author='user') with content.role='user' "
        "so the LLM provider receives a properly-shaped user message"
    )


def test_assistant_history_turn_becomes_event_authored_by_agent_with_model_role_content() -> None:
    event = _make_history_event(
        {"role": "assistant", "content": "you have 3 subscriptions"},
        agent_name="conversational_agent",
    )

    assert (
        event is not None
        and event.author == "conversational_agent"
        and event.content is not None
        and event.content.role == "model"
    ), (
        "assistant-side history turns must be Event(author=agent_name) with "
        "content.role='model' so google-genai treats them as the assistant turn"
    )


def test_history_turn_text_content_is_preserved_verbatim_into_the_event() -> None:
    text = "Подписка “Цель” -- event-уведомления по аниме-тайтлам."
    event = _make_history_event(
        {"role": "assistant", "content": text},
        agent_name="conversational_agent",
    )

    assert event is not None and event.content.parts[0].text == text, (
        "history replay must not mutate or transcode the message text; cyrillic and "
        "punctuation in the original turn must reach the LLM byte-for-byte"
    )


def test_empty_or_whitespace_only_history_turn_is_dropped_to_avoid_provider_400() -> None:
    event = _make_history_event(
        {"role": "user", "content": "   \n\t "},
        agent_name="conversational_agent",
    )

    assert event is None, (
        "empty turns must not become Events; some chat-completions providers reject "
        "messages with empty string content with a 400 InvalidRequestError"
    )


def test_unknown_role_in_history_is_skipped_rather_than_mapped_to_user() -> None:
    event = _make_history_event(
        {"role": "tool", "content": '{"status": "ok"}'},
        agent_name="conversational_agent",
    )

    assert event is None, (
        "unknown roles must be skipped, not silently coerced into a user turn; "
        "stuffing tool output into the user role would mislead the model about "
        "which side actually said it"
    )


@pytest.mark.asyncio
async def test_run_agent_appends_one_session_event_per_history_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from news_service.agents import adk_runner

    appended: list = []

    class _RecordingSessionService:
        async def create_session(self, *, app_name, user_id, session_id, state=None):  # noqa: D401, ANN001, ARG002
            return _Session(app_name=app_name, user_id=user_id, session_id=session_id)

        async def append_event(self, session, event):  # noqa: ANN001
            appended.append((session.session_id, event.author, event.content.role))
            return event

    class _Session:
        def __init__(self, *, app_name, user_id, session_id):  # noqa: ANN001
            self.app_name = app_name
            self.user_id = user_id
            self.session_id = session_id

    class _RunnerStub:
        def __init__(self, *, agent, app_name, session_service):  # noqa: ANN001, ARG002
            pass

        async def run_async(self, *, user_id, session_id, new_message, **kwargs):  # noqa: ANN001, ARG002
            assert new_message.parts[0].text == "call it 'anime feed'", (
                "run_agent did not pass the current user message to ADK as new_message"
            )

            class _FinalEvent:
                content = type(
                    "C",
                    (),
                    {"parts": [type("P", (), {"text": "ok"})()]},
                )()

                def is_final_response(self) -> bool:
                    return True

            yield _FinalEvent()

    class _AgentStub:
        name = "conversational_agent"

    monkeypatch.setattr(adk_runner, "InMemorySessionService", _RecordingSessionService)
    monkeypatch.setattr(adk_runner, "Runner", _RunnerStub)

    history = [
        {"role": "user", "content": "what subs do I have"},
        {"role": "assistant", "content": "you have 3"},
        {"role": "user", "content": "rename the anime one"},
        {"role": "assistant", "content": "ok, what name"},
    ]
    events_received: list = []
    async for event in adk_runner.run_agent(
        agent=_AgentStub(),
        message="call it 'anime feed'",
        user_id="user-123",
        conversation_history=history,
    ):
        events_received.append(event)

    expected = [
        ("user", "user"),
        ("conversational_agent", "model"),
        ("user", "user"),
        ("conversational_agent", "model"),
    ]
    actual = [(author, role) for _, author, role in appended]
    assert actual == expected, (
        f"run_agent must append one Event per non-empty history turn with the right "
        f"author/role mapping; got {actual} instead of {expected}"
    )
