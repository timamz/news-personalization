"""Tests for the digest pipeline: happy path, revision loop, empty cases."""

import logging
import random
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from news_service.agents.digest import pipeline as digest_pipeline
from news_service.agents.digest.judge import QualityScores
from news_service.agents.digest.writer import DigestComposition

logging.disable(logging.CRITICAL)

_PIPELINE = "news_service.agents.digest.pipeline"


def _make_subscription(prompt: str, embedding: list[float] | None) -> SimpleNamespace:
    return SimpleNamespace(
        id=uuid.uuid4(),
        user_id=uuid.uuid4(),
        topic_embedding=embedding,
        user_spec=f"{prompt}. Short digest, a few bullets.",
        digest_language="en",
        last_reflected_at=datetime.now(UTC),
    )


def _make_session_with_sources(source_ids: list[uuid.UUID]) -> SimpleNamespace:
    sent_result = SimpleNamespace(all=lambda: [])
    source_result = SimpleNamespace(all=lambda: [(sid,) for sid in source_ids])
    recent_digest_result = SimpleNamespace(all=lambda: [])
    return SimpleNamespace(
        execute=AsyncMock(side_effect=[sent_result, source_result, recent_digest_result]),
        flush=AsyncMock(),
        add=lambda x: None,
    )


def _composition(text: str, ids: list[str]) -> DigestComposition:
    return DigestComposition(digest_text=text, used_item_ids=ids)


def _quality(verdict: str = "PASS", feedback: str = "") -> QualityScores:
    return QualityScores(
        relevance=4, format_score=4, conciseness=5, verdict=verdict, feedback=feedback
    )


def _patch_happy_path(mocker, digest_text: str, item_ids: list[str]) -> None:
    fake_item = SimpleNamespace(
        id=uuid.UUID(item_ids[0]) if item_ids else uuid.uuid4(),
        source_id=uuid.uuid4(),
        headline="T",
        body="B",
        url="http://t.com",
        embedding=[0.1] * 10,
        published_at=datetime(2026, 3, 10, tzinfo=UTC),
        fetched_at=datetime(2026, 3, 10, tzinfo=UTC),
    )
    mocker.patch(f"{_PIPELINE}.fetch_candidate_items", new=AsyncMock(return_value=[fake_item]))
    mocker.patch(f"{_PIPELINE}.build_items_text", return_value="[ID: ...] Headline: T")
    mocker.patch(
        f"{_PIPELINE}.write_digest",
        new=AsyncMock(return_value=_composition(digest_text, item_ids)),
    )
    mocker.patch(f"{_PIPELINE}.judge_digest", new=AsyncMock(return_value=_quality()))
    mocker.patch(f"{_PIPELINE}._load_source_contexts", new=AsyncMock(return_value=[]))


@pytest.mark.asyncio
async def test_generate_digest_returns_composed_digest_and_passes_user_spec_to_writer(
    mocker,
) -> None:
    source_id = uuid.uuid4()
    embedding = [random.random() for _ in range(1536)]
    prompt = f"AI {uuid.uuid4().hex[:6]}"
    session = _make_session_with_sources([source_id])
    subscription = _make_subscription(prompt, embedding)
    digest_text = f"Digest {uuid.uuid4().hex[:6]}"
    _patch_happy_path(mocker, digest_text, [str(uuid.uuid4())])

    write_mock = mocker.patch(
        f"{_PIPELINE}.write_digest",
        new=AsyncMock(return_value=_composition(digest_text, [str(uuid.uuid4())])),
    )
    # Re-patch fetch candidate with the same item id as composition
    ids = [str(uuid.uuid4())]
    _patch_happy_path(mocker, digest_text, ids)
    write_mock = mocker.patch(
        f"{_PIPELINE}.write_digest",
        new=AsyncMock(return_value=_composition(digest_text, ids)),
    )

    result = await digest_pipeline.generate_digest(session, subscription)

    assert result == digest_text, "pipeline did not return the composed digest text"
    passed_spec = write_mock.await_args.kwargs["user_spec"]
    assert prompt in passed_spec, "writer was not called with the user_spec"


@pytest.mark.asyncio
async def test_generate_digest_returns_none_without_fixed_sources() -> None:
    session = _make_session_with_sources([])
    subscription = _make_subscription(f"Topic {uuid.uuid4().hex[:4]}", [0.0] * 1536)

    result = await digest_pipeline.generate_digest(session, subscription)
    assert result is None


@pytest.mark.asyncio
async def test_generate_digest_returns_none_when_no_candidates(mocker) -> None:
    session = _make_session_with_sources([uuid.uuid4()])
    subscription = _make_subscription(f"Topic {uuid.uuid4().hex[:4]}", [0.0] * 1536)
    mocker.patch(f"{_PIPELINE}.fetch_candidate_items", new=AsyncMock(return_value=[]))

    result = await digest_pipeline.generate_digest(session, subscription)
    assert result is None


@pytest.mark.asyncio
async def test_generate_digest_computes_and_stores_embedding_when_missing(mocker) -> None:
    session = _make_session_with_sources([uuid.uuid4()])
    subscription = _make_subscription(f"Topic {uuid.uuid4().hex[:4]}", None)
    computed = [random.random() for _ in range(1536)]
    embed_mock = mocker.patch(f"{_PIPELINE}.embed_text", new=AsyncMock(return_value=computed))
    _patch_happy_path(mocker, f"Text {uuid.uuid4().hex[:6]}", [str(uuid.uuid4())])

    await digest_pipeline.generate_digest(session, subscription)

    assert embed_mock.await_count == 1 and subscription.topic_embedding == computed, (
        "pipeline did not compute and store a topic embedding when missing"
    )


@pytest.mark.asyncio
async def test_generate_digest_revises_on_judge_feedback_and_returns_revised_text(mocker) -> None:
    source_id = uuid.uuid4()
    session = _make_session_with_sources([source_id])
    subscription = _make_subscription(
        f"Topic {uuid.uuid4().hex[:4]}", [random.random() for _ in range(1536)]
    )

    candidate_item_id = uuid.uuid4()
    fake_item = SimpleNamespace(
        id=candidate_item_id,
        source_id=source_id,
        headline="T",
        body="B",
        url="http://t.com",
        embedding=[0.1] * 10,
        published_at=datetime(2026, 3, 10, tzinfo=UTC),
        fetched_at=datetime(2026, 3, 10, tzinfo=UTC),
    )
    mocker.patch(f"{_PIPELINE}.fetch_candidate_items", new=AsyncMock(return_value=[fake_item]))
    mocker.patch(f"{_PIPELINE}.build_items_text", return_value="items")
    mocker.patch(f"{_PIPELINE}._load_source_contexts", new=AsyncMock(return_value=[]))
    mocker.patch(
        f"{_PIPELINE}.run_reflector",
        new=AsyncMock(return_value={"discovery_triggered": False}),
    )

    first = f"Draft {uuid.uuid4().hex[:6]}"
    revised = f"Revised {uuid.uuid4().hex[:6]}"
    cid = str(candidate_item_id)
    write_mock = mocker.patch(
        f"{_PIPELINE}.write_digest",
        new=AsyncMock(side_effect=[_composition(first, [cid]), _composition(revised, [cid])]),
    )
    mocker.patch(
        f"{_PIPELINE}.judge_digest",
        new=AsyncMock(
            side_effect=[_quality(verdict="REVISE", feedback="Too long"), _quality(verdict="PASS")]
        ),
    )

    result = await digest_pipeline.generate_digest(session, subscription)

    assert write_mock.await_count == 2 and result == revised, (
        "pipeline did not run exactly two writer attempts or return the revised text"
    )
