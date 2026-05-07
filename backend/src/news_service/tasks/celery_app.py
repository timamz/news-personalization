import json
import logging
from datetime import UTC, datetime

from celery import Celery
from celery.signals import task_failure, worker_process_init

from news_service.core.config import get_settings
from news_service.core.llm_usage import install_usage_callback

logger = logging.getLogger(__name__)

settings = get_settings()

celery_app = Celery("news_service", broker=settings.redis_url, backend=settings.redis_url)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    beat_schedule={
        "poll-all-feeds": {
            "task": "news_service.tasks.poll_feeds.poll_all_feeds",
            "schedule": settings.rss_poll_interval_minutes * 60,
        },
        "schedule-due-digests": {
            "task": "news_service.tasks.schedule_digests.schedule_due_digests",
            "schedule": 60,
        },
        "update-source-embeddings": {
            "task": "news_service.tasks.update_source_embeddings.update_all_source_embeddings",
            "schedule": 24 * 60 * 60,
        },
        "update-subscription-source-stats": {
            "task": (
                "news_service.tasks.update_subscription_source_stats."
                "update_all_subscription_source_stats"
            ),
            "schedule": 24 * 60 * 60,
        },
        "reflect-event-subscriptions": {
            "task": "news_service.tasks.reflect_events.reflect_event_subscriptions",
            "schedule": 24 * 60 * 60,
        },
    },
)


@worker_process_init.connect
def _install_llm_usage_callback(**_: object) -> None:
    """Initialize each freshly forked Celery worker process.

    Two things happen here, both required exactly once per child:

    1. Dispose the SQLAlchemy AsyncEngine inherited from the prefork
       parent. The module-level ``engine`` in ``news_service.db.session``
       is constructed at import time, so the parent already holds asyncpg
       connections whose socket file descriptors get duplicated into every
       child by ``fork()``. Two processes writing to the same socket
       triggers ``asyncpg.InterfaceError: cannot perform operation:
       another operation is in progress`` and eventually deadlocks the
       worker on a corrupted recv. Disposing the inherited pool drops
       those FDs in the child; new connections are opened lazily on first
       use, scoped to this process.

    2. Register the LiteLLM success/failure callback that writes the
       per-call ``llm_usage`` ledger row. The callback is a side effect
       on the ``litellm`` module and does not survive ``fork()`` cleanly
       in every code path, so we re-register here per child.
    """
    import asyncio

    from news_service.db.session import engine

    asyncio.run(engine.dispose())
    install_usage_callback()


@task_failure.connect
def record_failed_task(
    sender: object = None,
    task_id: str | None = None,
    exception: BaseException | None = None,
    args: tuple | None = None,
    kwargs: dict | None = None,
    einfo: object = None,
    **_kw: object,
) -> None:
    """Record failed Celery tasks into the failed_tasks table (dead letter queue).

    Connected via Celery signal — fires after any task raises an unhandled exception.
    Uses a synchronous DB connection (Celery workers run sync event loops).
    """
    import asyncio

    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
    from sqlalchemy.pool import NullPool

    from news_service.models.failed_task import FailedTask

    task_name = getattr(sender, "name", "unknown")
    retries = getattr(sender, "request", None)
    retry_count = getattr(retries, "retries", 0) if retries else 0

    tb_text = ""
    if einfo is not None:
        tb_text = str(einfo)

    failed = FailedTask(
        task_name=task_name,
        task_args=json.dumps(list(args)) if args else "[]",
        task_kwargs=json.dumps(kwargs) if kwargs else "{}",
        exception_type=type(exception).__name__ if exception else "Unknown",
        exception_message=str(exception) if exception else "",
        traceback=tb_text,
        retries=retry_count,
        failed_at=datetime.now(UTC),
    )

    async def _persist() -> None:
        eng = create_async_engine(settings.database_url, poolclass=NullPool)
        factory = async_sessionmaker(eng, class_=AsyncSession, expire_on_commit=False)
        try:
            async with factory() as session:
                session.add(failed)
                await session.commit()
        finally:
            await eng.dispose()

    try:
        asyncio.run(_persist())
    except Exception:
        logger.exception(
            "Failed to record dead-letter entry for task %s (id=%s)",
            task_name,
            task_id,
        )


import news_service.tasks.deliver_digest  # noqa: E402, F401
import news_service.tasks.deliver_events  # noqa: E402, F401
import news_service.tasks.discover_sources  # noqa: E402, F401
import news_service.tasks.poll_feeds  # noqa: E402, F401
import news_service.tasks.reflect_events  # noqa: E402, F401
import news_service.tasks.schedule_digests  # noqa: E402, F401
import news_service.tasks.update_source_embeddings  # noqa: E402, F401
import news_service.tasks.update_subscription_source_stats  # noqa: E402, F401
