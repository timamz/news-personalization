import logging
import uuid
from types import SimpleNamespace

import pytest
from openai import OpenAIError

from news_service.db import vector_store

logging.disable(logging.CRITICAL)


def _response(size: int) -> SimpleNamespace:
    return SimpleNamespace(data=[SimpleNamespace(embedding=[float(i)]) for i in range(size)])


@pytest.mark.asyncio
async def test_embed_texts_returns_correct_count(mocker) -> None:
    create_mock = mocker.AsyncMock(side_effect=lambda **kwargs: _response(len(kwargs["input"])))
    mocker.patch.object(vector_store._client.embeddings, "create", create_mock)

    contents = [f"текст {uuid.uuid4().hex[:4]}" for _ in range(13)]
    result = await vector_store.embed_texts(contents)

    assert len(result) == 13, "embed_texts did not return correct number of embeddings"


@pytest.mark.asyncio
async def test_embed_texts_batches_into_three_requests(mocker) -> None:
    create_mock = mocker.AsyncMock(side_effect=lambda **kwargs: _response(len(kwargs["input"])))
    mocker.patch.object(vector_store._client.embeddings, "create", create_mock)

    contents = [f"текст {uuid.uuid4().hex[:4]}" for _ in range(13)]
    await vector_store.embed_texts(contents)

    assert create_mock.await_count == 3, "embed_texts did not split 13 items into 3 batch requests"


@pytest.mark.asyncio
async def test_embed_texts_first_batch_has_six_items(mocker) -> None:
    create_mock = mocker.AsyncMock(side_effect=lambda **kwargs: _response(len(kwargs["input"])))
    mocker.patch.object(vector_store._client.embeddings, "create", create_mock)

    contents = [f"текст {uuid.uuid4().hex[:4]}" for _ in range(13)]
    await vector_store.embed_texts(contents)

    first_call_input = create_mock.await_args_list[0].kwargs["input"]
    assert len(first_call_input) == 6, "embed_texts first batch did not have 6 items"


@pytest.mark.asyncio
async def test_embed_texts_last_batch_has_one_item(mocker) -> None:
    create_mock = mocker.AsyncMock(side_effect=lambda **kwargs: _response(len(kwargs["input"])))
    mocker.patch.object(vector_store._client.embeddings, "create", create_mock)

    contents = [f"текст {uuid.uuid4().hex[:4]}" for _ in range(13)]
    await vector_store.embed_texts(contents)

    third_call_input = create_mock.await_args_list[2].kwargs["input"]
    assert len(third_call_input) == 1, "embed_texts last batch did not have 1 item"


@pytest.mark.asyncio
async def test_embed_texts_falls_back_to_per_item_after_batch_error(mocker) -> None:
    batch_failed = False

    async def _create(**kwargs) -> SimpleNamespace:
        nonlocal batch_failed
        payload = kwargs["input"]
        if isinstance(payload, list):
            if not batch_failed:
                batch_failed = True
                raise OpenAIError("ошибка пакетного запроса")
            return _response(len(payload))
        return _response(1)

    create_mock = mocker.AsyncMock(side_effect=_create)
    mocker.patch.object(vector_store._client.embeddings, "create", create_mock)

    contents = [f"текст-{uuid.uuid4().hex[:4]}" for _ in range(3)]
    result = await vector_store.embed_texts(contents)

    assert len(result) == 3, (
        "embed_texts did not return correct count after falling back to per-item requests"
    )


@pytest.mark.asyncio
async def test_embed_texts_fallback_makes_correct_number_of_calls(mocker) -> None:
    batch_failed = False

    async def _create(**kwargs) -> SimpleNamespace:
        nonlocal batch_failed
        payload = kwargs["input"]
        if isinstance(payload, list):
            if not batch_failed:
                batch_failed = True
                raise OpenAIError("ошибка пакетного запроса")
            return _response(len(payload))
        return _response(1)

    create_mock = mocker.AsyncMock(side_effect=_create)
    mocker.patch.object(vector_store._client.embeddings, "create", create_mock)

    contents = [f"текст-{uuid.uuid4().hex[:4]}" for _ in range(3)]
    await vector_store.embed_texts(contents)

    assert create_mock.await_count == 4, (
        "embed_texts fallback did not make exactly 4 API calls (1 batch + 3 per-item)"
    )
