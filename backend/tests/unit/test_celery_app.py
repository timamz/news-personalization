import logging

import pytest

from news_service.tasks import celery_app

logging.disable(logging.CRITICAL)


class _DisposalRecorder:
    """Engine stand-in that counts how many times its async dispose() was awaited.

    Used to assert that the Celery prefork init hook drops the inherited
    asyncpg connection pool. Example:

        recorder = _DisposalRecorder()
        monkeypatch.setattr("news_service.db.session.engine", recorder)
        celery_app._install_llm_usage_callback()
        assert recorder.dispose_calls == 1
    """

    def __init__(self) -> None:
        self.dispose_calls = 0

    async def dispose(self) -> None:
        self.dispose_calls += 1


def test_worker_process_init_disposes_inherited_engine_to_prevent_asyncpg_fork_deadlock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = _DisposalRecorder()
    monkeypatch.setattr("news_service.db.session.engine", recorder)
    monkeypatch.setattr(
        "news_service.tasks.celery_app.install_usage_callback",
        lambda: None,
    )

    celery_app._install_llm_usage_callback()

    assert recorder.dispose_calls == 1, (
        "worker_process_init does not dispose the parent's AsyncEngine; forked "
        "workers will share asyncpg sockets and deadlock with InterfaceError"
    )


def test_worker_process_init_still_registers_the_litellm_usage_callback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("news_service.db.session.engine", _DisposalRecorder())
    callback_invocations = 0

    def _record() -> None:
        nonlocal callback_invocations
        callback_invocations += 1

    monkeypatch.setattr("news_service.tasks.celery_app.install_usage_callback", _record)

    celery_app._install_llm_usage_callback()

    assert callback_invocations == 1, (
        "worker_process_init no longer registers the llm_usage callback; per-call "
        "cost attribution rows will silently stop being written"
    )
