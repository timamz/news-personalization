"""Celery broker roundtrip smoke test.

Verifies that ``celery_app.send_task`` actually reaches a worker via the
configured Redis broker, not just that the function signature is callable.
Registers a throwaway echo task, starts an in-process worker thread via
``celery.contrib.testing.worker.start_worker``, dispatches a ``send_task``
with a random payload, and asserts the worker executed it and returned
the payload through the result backend.

If this test fails, ``send_task`` calls made from ``reflect_events`` or
``deliver_events`` may never reach a worker in prod.
"""

import uuid

import pytest
from celery.contrib.testing.worker import start_worker

from news_service.tasks.celery_app import celery_app


@celery_app.task(name="news_service.tests.celery_echo")
def _celery_echo(payload: str) -> str:
    """Throwaway task registered at import time for the broker smoke test."""
    return payload


@pytest.mark.asyncio(loop_scope="session")
async def test_send_task_reaches_worker_through_redis_broker() -> None:
    payload = f"echo-{uuid.uuid4().hex}-Привет"

    with start_worker(
        celery_app,
        perform_ping_check=False,
        shutdown_timeout=10.0,
        loglevel="ERROR",
    ):
        async_result = celery_app.send_task(
            "news_service.tests.celery_echo",
            args=[payload],
        )
        result = async_result.get(timeout=10)

    assert result == payload, (
        "celery broker roundtrip did not return the dispatched payload; "
        "send_task may not be reaching workers in prod"
    )
