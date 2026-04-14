"""Pipeline tracing — records events for observability and replay.

Each pipeline run gets a unique trace_id. Every agent call within
that run is recorded as a PipelineEvent with timing, token usage,
and input/output summaries.
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from news_service.models.evaluation_result import EvaluationResult
from news_service.models.pipeline_event import PipelineEvent

logger = logging.getLogger(__name__)


def generate_trace_id() -> str:
    return uuid.uuid4().hex[:16]


async def record_pipeline_event(
    session: AsyncSession,
    *,
    trace_id: str,
    pipeline_type: str,
    agent_name: str,
    event_type: str,
    subscription_id: uuid.UUID | None = None,
    input_summary: dict[str, Any] | None = None,
    output_summary: dict[str, Any] | None = None,
    token_usage: dict[str, int] | None = None,
    latency_ms: int | None = None,
    model_name: str | None = None,
    error: str | None = None,
) -> PipelineEvent:
    """Record a single pipeline event to the database."""
    event = PipelineEvent(
        trace_id=trace_id,
        pipeline_type=pipeline_type,
        agent_name=agent_name,
        event_type=event_type,
        subscription_id=subscription_id,
        input_summary=input_summary,
        output_summary=output_summary,
        token_usage=token_usage,
        latency_ms=latency_ms,
        model_name=model_name,
        error=error,
    )
    session.add(event)
    await session.flush()
    return event


async def record_evaluation_result(
    session: AsyncSession,
    *,
    trace_id: str,
    subscription_id: uuid.UUID,
    delivery_type: str,
    relevance_score: float,
    format_score: float,
    conciseness_score: float,
    judge_model: str,
    verdict: str = "PASS",
) -> EvaluationResult:
    """Record quality scores from the LLM-as-Judge."""
    overall = (relevance_score + format_score + conciseness_score) / 3.0
    result = EvaluationResult(
        trace_id=trace_id,
        subscription_id=subscription_id,
        delivery_type=delivery_type,
        relevance_score=relevance_score,
        format_score=format_score,
        conciseness_score=conciseness_score,
        overall_score=round(overall, 2),
        judge_model=judge_model,
        verdict=verdict,
    )
    session.add(result)
    await session.flush()
    return result


@asynccontextmanager
async def trace_agent_call(
    session: AsyncSession,
    *,
    trace_id: str,
    pipeline_type: str,
    agent_name: str,
    subscription_id: uuid.UUID | None = None,
    input_summary: dict[str, Any] | None = None,
    model_name: str | None = None,
):
    """Context manager that times an agent call and records the event.

    Usage:
        async with trace_agent_call(session, trace_id=tid, ...) as ctx:
            result = await some_llm_call()
            ctx["output_summary"] = {"key": "value"}
            ctx["token_usage"] = {"prompt": 100, "completion": 50}
    """
    ctx: dict[str, Any] = {
        "output_summary": None,
        "token_usage": None,
        "error": None,
    }
    start = time.monotonic()
    try:
        yield ctx
    except Exception as exc:
        ctx["error"] = str(exc)[:500]
        raise
    finally:
        latency_ms = int((time.monotonic() - start) * 1000)
        try:
            await record_pipeline_event(
                session,
                trace_id=trace_id,
                pipeline_type=pipeline_type,
                agent_name=agent_name,
                event_type="llm_call" if ctx["error"] is None else "error",
                subscription_id=subscription_id,
                input_summary=input_summary,
                output_summary=ctx["output_summary"],
                token_usage=ctx["token_usage"],
                latency_ms=latency_ms,
                model_name=model_name,
                error=ctx["error"],
            )
        except Exception:
            logger.exception("Failed to record pipeline event (non-blocking)")
