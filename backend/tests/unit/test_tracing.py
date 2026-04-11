"""Tests for pipeline tracing and event recording."""

import logging
import uuid
from unittest.mock import AsyncMock

import pytest

from news_service.orchestration.tracing import (
    generate_trace_id,
    record_evaluation_result,
    record_pipeline_event,
    trace_agent_call,
)

logging.disable(logging.CRITICAL)


def test_generate_trace_id_returns_hex_string() -> None:
    tid = generate_trace_id()
    assert len(tid) == 16, "trace_id should be 16 hex characters"
    assert tid.isalnum(), "trace_id should be alphanumeric"


def test_generate_trace_id_is_unique() -> None:
    ids = {generate_trace_id() for _ in range(100)}
    assert len(ids) == 100, "generate_trace_id did not produce unique IDs"


@pytest.mark.asyncio
async def test_record_pipeline_event_adds_to_session() -> None:
    session = AsyncMock()
    session.add = lambda x: None
    session.flush = AsyncMock()

    event = await record_pipeline_event(
        session,
        trace_id=generate_trace_id(),
        pipeline_type="digest",
        agent_name=f"TestAgent-{uuid.uuid4().hex[:6]}",
        event_type="llm_call",
        subscription_id=uuid.uuid4(),
        latency_ms=1200,
    )

    assert event.pipeline_type == "digest", "event did not preserve pipeline_type"
    assert event.latency_ms == 1200, "event did not preserve latency_ms"


@pytest.mark.asyncio
async def test_record_evaluation_result_computes_overall_score() -> None:
    session = AsyncMock()
    session.add = lambda x: None
    session.flush = AsyncMock()

    result = await record_evaluation_result(
        session,
        trace_id=generate_trace_id(),
        subscription_id=uuid.uuid4(),
        delivery_type="digest",
        relevance_score=4.0,
        coverage_score=3.0,
        dedup_score=5.0,
        quality_score=4.0,
        judge_model="openai/gpt-5.4-nano",
    )

    assert result.overall_score == 4.0, "evaluation did not compute correct overall score"
    assert result.verdict == "PASS", "evaluation did not default to PASS verdict"


@pytest.mark.asyncio
async def test_trace_agent_call_records_latency() -> None:
    recorded_events = []
    session = AsyncMock()
    session.add = lambda x: recorded_events.append(x)
    session.flush = AsyncMock()

    async with trace_agent_call(
        session,
        trace_id=generate_trace_id(),
        pipeline_type="digest",
        agent_name="TestAgent",
    ) as ctx:
        ctx["output_summary"] = {"result": "ok"}

    assert len(recorded_events) == 1, "trace_agent_call did not record an event"
    assert recorded_events[0].latency_ms >= 0, "recorded event has no latency"
    assert recorded_events[0].event_type == "llm_call", "event_type should be llm_call on success"


@pytest.mark.asyncio
async def test_trace_agent_call_records_error_on_exception() -> None:
    recorded_events = []
    session = AsyncMock()
    session.add = lambda x: recorded_events.append(x)
    session.flush = AsyncMock()

    with pytest.raises(RuntimeError):
        async with trace_agent_call(
            session,
            trace_id=generate_trace_id(),
            pipeline_type="digest",
            agent_name="FailAgent",
        ):
            raise RuntimeError(f"test error {uuid.uuid4().hex[:6]}")

    assert len(recorded_events) == 1, "trace_agent_call did not record event on error"
    assert recorded_events[0].event_type == "error", "event_type should be error on failure"
    assert "test error" in recorded_events[0].error, "error field did not capture exception message"
