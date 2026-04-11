import logging
import uuid
from unittest.mock import AsyncMock

import pytest

from news_service.core import llm as llm_module

logging.disable(logging.CRITICAL)


def _embedding_response(size: int) -> dict:
    return {"data": [{"embedding": [float(i)]} for i in range(size)]}


@pytest.mark.asyncio
async def test_embed_texts_batches_thirteen_items_into_three_requests_with_correct_sizes(
    mocker,
) -> None:
    async def _mock_aembedding(**kwargs):
        return type(
            "R", (), {"data": [{"embedding": [float(i)]} for i in range(len(kwargs["input"]))]}
        )()

    mock_embed = mocker.patch(
        "news_service.core.llm.litellm.aembedding", new=AsyncMock(side_effect=_mock_aembedding)
    )

    contents = [f"текст {uuid.uuid4().hex[:4]}" for _ in range(13)]
    result = await llm_module.embed_texts(contents)

    assert len(result) == 13, "embed_texts did not return correct number of embeddings"
    assert mock_embed.await_count == 3, "embed_texts did not split 13 items into 3 batch requests"
    first_call_input = mock_embed.await_args_list[0].kwargs["input"]
    assert len(first_call_input) == 6, "embed_texts first batch did not have 6 items"
    third_call_input = mock_embed.await_args_list[2].kwargs["input"]
    assert len(third_call_input) == 1, "embed_texts last batch did not have 1 item"


@pytest.mark.asyncio
async def test_embed_texts_falls_back_to_per_item_after_batch_error(mocker) -> None:
    batch_failed = False

    async def _mock_aembedding(**kwargs):
        nonlocal batch_failed
        inp = kwargs["input"]
        if isinstance(inp, list) and len(inp) > 1 and not batch_failed:
            batch_failed = True
            raise RuntimeError("ошибка пакетного запроса")
        size = len(inp) if isinstance(inp, list) else 1
        return type("R", (), {"data": [{"embedding": [float(i)]} for i in range(size)]})()

    mocker.patch(
        "news_service.core.llm.litellm.aembedding", new=AsyncMock(side_effect=_mock_aembedding)
    )

    contents = [f"текст-{uuid.uuid4().hex[:4]}" for _ in range(3)]
    result = await llm_module.embed_texts(contents)

    assert len(result) == 3, (
        "embed_texts did not return correct count after falling back to per-item requests"
    )
