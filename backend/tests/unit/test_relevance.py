import logging
import random
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest

from news_service.services import relevance
from news_service.services.relevance import DatedPost, sample_recent_posts

logging.disable(logging.CRITICAL)


def _dated(text: str, days_ago: float | None) -> DatedPost:
    if days_ago is None:
        return DatedPost(text=text, published_at=None)
    return DatedPost(text=text, published_at=datetime.now(UTC) - timedelta(days=days_ago))


def test_sample_recent_posts_prefers_items_inside_the_window() -> None:
    recent = [_dated(f"recent-{i}", days_ago=i) for i in range(8)]
    old = [_dated(f"old-{i}", days_ago=200 + i) for i in range(6)]
    result = sample_recent_posts(recent + old, sample_size=5, window_days=30, rng=random.Random(0))
    assert len(result) == 5 and all(r.startswith("recent-") for r in result), (
        "sampler did not prefer in-window posts when enough were available"
    )


def test_sample_recent_posts_falls_back_to_full_pool_when_window_too_sparse() -> None:
    recent = [_dated("recent", days_ago=1)]
    old = [_dated(f"old-{i}", days_ago=200 + i) for i in range(6)]
    undated = [_dated(f"undated-{i}", days_ago=None) for i in range(3)]
    result = sample_recent_posts(
        recent + old + undated, sample_size=5, window_days=30, rng=random.Random(0)
    )
    assert len(result) == 5 and any(r.startswith("old-") for r in result), (
        "fallback should sample from the whole pool when the recent window is too small"
    )


def test_sample_recent_posts_returns_all_when_pool_under_sample_size() -> None:
    posts = [_dated(f"post-{i}", days_ago=None) for i in range(3)]
    result = sample_recent_posts(posts, sample_size=10, window_days=30)
    assert sorted(result) == ["post-0", "post-1", "post-2"], (
        "tiny pool should be returned whole rather than raising or truncating"
    )


@pytest.mark.asyncio
async def test_score_source_relevance_computes_top_k_average(mocker) -> None:
    prompt_emb = [1.0, 0.0, 0.0]
    post_embs = [[0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.7, 0.3, 0.0]]
    mocker.patch.object(relevance, "embed_texts", new=AsyncMock(return_value=post_embs))

    score = await relevance.score_source_relevance(
        [f"text-{uuid.uuid4().hex[:4]}" for _ in range(3)], prompt_emb, top_k=2
    )

    expected = (
        relevance.cosine_similarity(post_embs[0], prompt_emb)
        + relevance.cosine_similarity(post_embs[1], prompt_emb)
    ) / 2
    assert score == pytest.approx(expected, abs=1e-4), (
        "score_source_relevance did not average the top-k cosine similarities"
    )


@pytest.mark.asyncio
async def test_score_candidate_returns_score_and_samples_from_fetched_posts(mocker) -> None:
    posts = [_dated(f"post {i}", days_ago=i * 0.1) for i in range(20)]
    mocker.patch.object(
        relevance,
        "fetch_source_posts",
        new=AsyncMock(return_value=posts),
    )
    mocker.patch.object(relevance, "embed_texts", new=AsyncMock(return_value=[[0.9, 0.1]] * 10))

    score, samples, is_dormant = await relevance.score_candidate(
        f"https://example.com/{uuid.uuid4().hex}/feed", "rss", [1.0, 0.0]
    )
    assert score > 0 and len(samples) == 10 and not is_dormant, (
        "score_candidate did not produce a positive score with the expected sample count"
    )


@pytest.mark.asyncio
async def test_score_candidate_returns_zero_on_fetch_failure(mocker) -> None:
    mocker.patch.object(
        relevance,
        "fetch_source_posts",
        new=AsyncMock(side_effect=RuntimeError("network down")),
    )

    score, samples, is_dormant = await relevance.score_candidate(
        f"https://example.com/{uuid.uuid4().hex}/feed", "rss", [1.0, 0.0]
    )
    assert score == 0.0 and samples == [] and not is_dormant, (
        "score_candidate did not degrade to zero-score with empty samples on fetch failure"
    )


@pytest.mark.asyncio
async def test_score_candidate_flags_dormant_source_when_no_recent_posts(mocker) -> None:
    posts = [_dated(f"post {i}", days_ago=365 + i) for i in range(20)]
    mocker.patch.object(
        relevance,
        "fetch_source_posts",
        new=AsyncMock(return_value=posts),
    )
    mocker.patch.object(relevance, "embed_texts", new=AsyncMock(return_value=[[0.9, 0.1]] * 10))

    score, samples, is_dormant = await relevance.score_candidate(
        f"https://example.com/{uuid.uuid4().hex}/feed", "rss", [1.0, 0.0]
    )
    assert is_dormant and score == 0.0 and samples == [], (
        "source with no posts inside news_item_max_age_days was not flagged dormant"
    )


@pytest.mark.asyncio
async def test_score_candidate_scores_when_every_post_lacks_a_timestamp(mocker) -> None:
    posts = [_dated(f"post {i}", days_ago=None) for i in range(20)]
    mocker.patch.object(
        relevance,
        "fetch_source_posts",
        new=AsyncMock(return_value=posts),
    )
    mocker.patch.object(relevance, "embed_texts", new=AsyncMock(return_value=[[0.9, 0.1]] * 10))

    score, samples, is_dormant = await relevance.score_candidate(
        f"https://example.com/{uuid.uuid4().hex}/feed", "rss", [1.0, 0.0]
    )
    assert not is_dormant and score > 0 and len(samples) > 0, (
        "source whose posts have no published_at should be scored, not flagged dormant"
    )
