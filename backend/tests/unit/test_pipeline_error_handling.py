"""Tests for digest pipeline error handling tiers and reflector trigger logic."""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from news_service.core.exceptions import DigestPipelineError


@pytest.fixture
def _subscription():
    sub = MagicMock()
    sub.id = uuid.uuid4()
    sub.user_id = uuid.uuid4()
    sub.user_spec = "## Topic\nMachine learning Neuigkeiten"
    sub.raw_prompt = "Machine learning Neuigkeiten"
    sub.topic_embedding = [0.1] * 1536
    sub.digest_language = "de"
    sub.format_instructions = "brief summary"
    sub.schedule_cron = "0 8 * * *"
    sub.last_reflected_at = datetime.now(UTC)
    return sub


@pytest.fixture
def _session():
    session = AsyncMock()
    result = MagicMock()
    result.all.return_value = []
    session.execute = AsyncMock(return_value=result)
    return session


@pytest.mark.asyncio
async def test_generate_digest_raises_pipeline_error_when_planner_fails(_subscription, _session):
    candidate = MagicMock()
    candidate.id = uuid.uuid4()
    candidate.source_id = uuid.uuid4()
    candidate.embedding = [0.2] * 1536

    with (
        patch(
            "news_service.agents.digest.pipeline.fetch_candidate_items",
            new_callable=AsyncMock,
            return_value=[candidate],
        ),
        patch(
            "news_service.agents.digest.pipeline.build_items_text",
            return_value="item text",
        ),
        patch(
            "news_service.agents.digest.pipeline._source_ids_for_digest",
            new_callable=AsyncMock,
            return_value={candidate.source_id},
        ),
        patch(
            "news_service.agents.digest.pipeline.plan_digest",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM timeout"),
        ),
    ):
        from news_service.agents.digest.pipeline import generate_digest

        with pytest.raises(DigestPipelineError, match="Planner failed"):
            await generate_digest(_session, _subscription)


@pytest.mark.asyncio
async def test_generate_digest_raises_pipeline_error_when_composer_fails(_subscription, _session):
    candidate = MagicMock()
    candidate.id = uuid.uuid4()
    candidate.source_id = uuid.uuid4()
    candidate.embedding = [0.2] * 1536

    plan_result = MagicMock()
    plan_result.plan = "Write about ML"

    with (
        patch(
            "news_service.agents.digest.pipeline.fetch_candidate_items",
            new_callable=AsyncMock,
            return_value=[candidate],
        ),
        patch(
            "news_service.agents.digest.pipeline.build_items_text",
            return_value="item text",
        ),
        patch(
            "news_service.agents.digest.pipeline._source_ids_for_digest",
            new_callable=AsyncMock,
            return_value={candidate.source_id},
        ),
        patch(
            "news_service.agents.digest.pipeline.plan_digest",
            new_callable=AsyncMock,
            return_value=plan_result,
        ),
        patch(
            "news_service.agents.digest.pipeline.compose_digest",
            new_callable=AsyncMock,
            side_effect=RuntimeError("LLM connection refused"),
        ),
    ):
        from news_service.agents.digest.pipeline import generate_digest

        with pytest.raises(DigestPipelineError, match="Composer failed"):
            await generate_digest(_session, _subscription)


@pytest.mark.asyncio
async def test_generate_digest_uses_unreviewed_draft_when_judge_fails(_subscription, _session):
    candidate = MagicMock()
    candidate.id = uuid.uuid4()
    candidate.source_id = uuid.uuid4()
    candidate.embedding = [0.2] * 1536

    plan_result = MagicMock()
    plan_result.plan = "Write about ML"

    composition = MagicMock()
    composition.digest_text = "Erstaunliche ML-Nachrichten heute"
    composition.used_item_ids = [str(candidate.id)]

    with (
        patch(
            "news_service.agents.digest.pipeline.fetch_candidate_items",
            new_callable=AsyncMock,
            return_value=[candidate],
        ),
        patch(
            "news_service.agents.digest.pipeline.build_items_text",
            return_value="item text",
        ),
        patch(
            "news_service.agents.digest.pipeline._source_ids_for_digest",
            new_callable=AsyncMock,
            return_value={candidate.source_id},
        ),
        patch(
            "news_service.agents.digest.pipeline.plan_digest",
            new_callable=AsyncMock,
            return_value=plan_result,
        ),
        patch(
            "news_service.agents.digest.pipeline.compose_digest",
            new_callable=AsyncMock,
            return_value=composition,
        ),
        patch(
            "news_service.agents.digest.pipeline.judge_digest",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Judge model unavailable"),
        ),
        patch(
            "news_service.agents.digest.pipeline.run_reflector",
            new_callable=AsyncMock,
            return_value={"discovery_triggered": False},
        ),
        patch(
            "news_service.agents.digest.pipeline._mark_as_sent",
            new_callable=AsyncMock,
        ),
    ):
        from news_service.agents.digest.pipeline import generate_digest

        result = await generate_digest(_session, _subscription)

        assert result is not None, "digest should be returned even when judge fails"
        assert result == "Erstaunliche ML-Nachrichten heute", "draft text should be used as-is"


@pytest.mark.asyncio
async def test_generate_digest_succeeds_when_reflector_fails(_subscription, _session):
    _subscription.last_reflected_at = None

    candidate = MagicMock()
    candidate.id = uuid.uuid4()
    candidate.source_id = uuid.uuid4()
    candidate.embedding = [0.2] * 1536

    plan_result = MagicMock()
    plan_result.plan = "Write about ML"

    composition = MagicMock()
    composition.digest_text = "Великие новости по ML"
    composition.used_item_ids = [str(candidate.id)]

    quality = MagicMock()
    quality.verdict = "PASS"
    quality.relevance = 5
    quality.format_score = 5
    quality.conciseness = 5
    quality.model_dump.return_value = {}

    with (
        patch(
            "news_service.agents.digest.pipeline.fetch_candidate_items",
            new_callable=AsyncMock,
            return_value=[candidate],
        ),
        patch(
            "news_service.agents.digest.pipeline.build_items_text",
            return_value="item text",
        ),
        patch(
            "news_service.agents.digest.pipeline._source_ids_for_digest",
            new_callable=AsyncMock,
            return_value={candidate.source_id},
        ),
        patch(
            "news_service.agents.digest.pipeline.plan_digest",
            new_callable=AsyncMock,
            return_value=plan_result,
        ),
        patch(
            "news_service.agents.digest.pipeline.compose_digest",
            new_callable=AsyncMock,
            return_value=composition,
        ),
        patch(
            "news_service.agents.digest.pipeline.judge_digest",
            new_callable=AsyncMock,
            return_value=quality,
        ),
        patch(
            "news_service.agents.digest.pipeline._build_source_info",
            new_callable=AsyncMock,
            return_value="- source info",
        ),
        patch(
            "news_service.agents.digest.pipeline.run_reflector",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Reflector crashed"),
        ),
        patch(
            "news_service.agents.digest.pipeline._mark_as_sent",
            new_callable=AsyncMock,
        ),
    ):
        from news_service.agents.digest.pipeline import generate_digest

        result = await generate_digest(_session, _subscription)

        assert result is not None, "digest should be returned even when reflector crashes"
        assert result == "Великие новости по ML", (
            "digest text should not be affected by reflector failure"
        )


def test_should_reflect_returns_true_when_pipeline_struggled():
    from news_service.agents.digest.pipeline import _should_reflect

    sub = MagicMock()
    sub.last_reflected_at = datetime.now(UTC)

    assert _should_reflect(subscription=sub, quality=None, candidates=[], source_ids=set()), (
        "should reflect when quality is None (judge failed)"
    )


def test_should_reflect_returns_true_when_source_coverage_below_threshold():
    from news_service.agents.digest.pipeline import _should_reflect

    sub = MagicMock()
    sub.last_reflected_at = datetime.now(UTC)

    quality = MagicMock()
    quality.verdict = "PASS"
    quality.relevance = 5
    quality.format_score = 5
    quality.conciseness = 5

    source_a = uuid.uuid4()
    source_b = uuid.uuid4()
    source_c = uuid.uuid4()

    candidate = MagicMock()
    candidate.source_id = source_a

    assert _should_reflect(
        subscription=sub,
        quality=quality,
        candidates=[candidate],
        source_ids={source_a, source_b, source_c},
    ), "should reflect when only 1 of 3 sources contributed (33% < 50%)"


def test_should_reflect_returns_true_when_quality_scores_mediocre():
    from news_service.agents.digest.pipeline import _should_reflect

    sub = MagicMock()
    sub.last_reflected_at = datetime.now(UTC)

    quality = MagicMock()
    quality.verdict = "PASS"
    quality.relevance = 3
    quality.format_score = 3
    quality.conciseness = 3

    source_id = uuid.uuid4()
    candidate = MagicMock()
    candidate.source_id = source_id

    assert _should_reflect(
        subscription=sub,
        quality=quality,
        candidates=[candidate],
        source_ids={source_id},
    ), "should reflect when average score (3.0) is below threshold (4.0)"


def test_should_reflect_returns_true_when_never_reflected_before():
    from news_service.agents.digest.pipeline import _should_reflect

    sub = MagicMock()
    sub.last_reflected_at = None

    quality = MagicMock()
    quality.verdict = "PASS"
    quality.relevance = 5
    quality.format_score = 5
    quality.conciseness = 5

    source_id = uuid.uuid4()
    candidate = MagicMock()
    candidate.source_id = source_id

    assert _should_reflect(
        subscription=sub,
        quality=quality,
        candidates=[candidate],
        source_ids={source_id},
    ), "should reflect when last_reflected_at is None"


def test_should_reflect_returns_false_when_healthy_and_recently_reflected():
    from news_service.agents.digest.pipeline import _should_reflect

    sub = MagicMock()
    sub.last_reflected_at = datetime.now(UTC) - timedelta(days=2)

    quality = MagicMock()
    quality.verdict = "PASS"
    quality.relevance = 5
    quality.format_score = 5
    quality.conciseness = 5

    source_id = uuid.uuid4()
    candidate = MagicMock()
    candidate.source_id = source_id

    assert not _should_reflect(
        subscription=sub,
        quality=quality,
        candidates=[candidate],
        source_ids={source_id},
    ), "should not reflect when all signals are healthy and recently reflected"
