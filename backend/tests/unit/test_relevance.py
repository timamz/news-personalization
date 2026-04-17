import logging
import uuid
from unittest.mock import AsyncMock

import pytest

from news_service.services import relevance

logging.disable(logging.CRITICAL)


def test_sample_posts_picks_evenly_spaced_indices() -> None:
    posts = [f"post-{i}" for i in range(20)]
    assert relevance.sample_posts(posts, 5) == ["post-0", "post-4", "post-8", "post-12", "post-16"]


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
    mocker.patch.object(
        relevance,
        "fetch_source_posts",
        new=AsyncMock(return_value=[f"post {i}" for i in range(20)]),
    )
    mocker.patch.object(relevance, "embed_texts", new=AsyncMock(return_value=[[0.9, 0.1]] * 10))

    score, samples = await relevance.score_candidate(
        f"https://example.com/{uuid.uuid4().hex}/feed", "rss", [1.0, 0.0]
    )
    assert score > 0 and len(samples) == 10, (
        "score_candidate did not produce a positive score with the expected sample count"
    )


@pytest.mark.asyncio
async def test_score_candidate_returns_zero_on_fetch_failure(mocker) -> None:
    mocker.patch.object(
        relevance,
        "fetch_source_posts",
        new=AsyncMock(side_effect=RuntimeError("network down")),
    )

    score, samples = await relevance.score_candidate(
        f"https://example.com/{uuid.uuid4().hex}/feed", "rss", [1.0, 0.0]
    )
    assert score == 0.0 and samples == [], (
        "score_candidate did not degrade to zero-score with empty samples on fetch failure"
    )
