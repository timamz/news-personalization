from types import SimpleNamespace

import pytest
from openai import OpenAIError

from news_service.db import vector_store


def _response(size: int) -> SimpleNamespace:
    return SimpleNamespace(data=[SimpleNamespace(embedding=[float(i)]) for i in range(size)])


@pytest.mark.asyncio
async def test_embed_texts_batches_requests(mocker) -> None:
    create_mock = mocker.AsyncMock(side_effect=lambda **kwargs: _response(len(kwargs["input"])))
    mocker.patch.object(vector_store._client.embeddings, "create", create_mock)

    contents = [f"text {i}" for i in range(13)]
    result = await vector_store.embed_texts(contents)

    assert len(result) == 13
    assert create_mock.await_count == 3
    first_call_input = create_mock.await_args_list[0].kwargs["input"]
    second_call_input = create_mock.await_args_list[1].kwargs["input"]
    third_call_input = create_mock.await_args_list[2].kwargs["input"]
    assert len(first_call_input) == 6
    assert len(second_call_input) == 6
    assert len(third_call_input) == 1


@pytest.mark.asyncio
async def test_embed_texts_falls_back_to_per_item_after_batch_error(mocker) -> None:
    batch_failed = False

    async def _create(**kwargs) -> SimpleNamespace:
        nonlocal batch_failed
        payload = kwargs["input"]
        if isinstance(payload, list):
            if not batch_failed:
                batch_failed = True
                raise OpenAIError("batch failed")
            return _response(len(payload))
        return _response(1)

    create_mock = mocker.AsyncMock(side_effect=_create)
    mocker.patch.object(vector_store._client.embeddings, "create", create_mock)

    result = await vector_store.embed_texts(["a", "b", "c"])

    assert len(result) == 3
    assert create_mock.await_count == 4
