"""
CeleryShim routes Celery task dispatch to inline ``await`` calls.

The benchmark runs the production code in-process against a throwaway
Postgres, with no Celery worker running. Every place the production
code calls ``celery_app.send_task(NAME, args=...)`` or ``task.delay(...)``
would therefore sit in Redis forever: no worker picks it up, and the
pipeline it represents (digest delivery, event batch delivery, source
discovery triggered by a reflector, ...) never fires. This is the
single biggest coverage hole the benchmark had -- *everything* behind a
Celery dispatch was silently dead.

The shim fixes this by replacing ``celery_app.send_task`` with an inline
dispatcher that:

  1. Looks the task name up in a registry mapping to the underlying
     async coroutine (``_deliver_digest``, ``_poll_all_feeds``, etc.).
  2. Runs the coroutine on the current event loop as a
     ``asyncio.create_task`` and awaits its completion inside a
     short-lived background helper. The benchmark orchestrator is
     already running in an event loop, so we must NOT ``asyncio.run``.
  3. Logs unknown task names (instead of silently dropping them, which
     was the old behavior).

Task signatures use string UUIDs the way the Celery wrappers do
(``deliver_digest(subscription_id: str, ...)``), so the shim parses
``uuid.UUID(arg)`` before calling the underlying async implementation
where relevant. A small ``_TaskSpec`` table makes this per-task
coercion explicit.

Also patches ``deliver_digest.delay(subscription_id, notify_if_empty)``
for the ``trigger_digest_now`` conversational tool, which dispatches
by attribute access on the task decorator rather than by name.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class _TaskSpec:
    """One registered task: the async impl and how to coerce its positional args."""

    impl: Callable[..., Awaitable[Any]]
    coercions: tuple[Callable[[Any], Any], ...] = ()

    def coerce(self, args: tuple[Any, ...]) -> tuple[Any, ...]:
        out: list[Any] = []
        for i, raw in enumerate(args):
            coerce_fn = self.coercions[i] if i < len(self.coercions) else None
            out.append(coerce_fn(raw) if coerce_fn is not None and raw is not None else raw)
        return tuple(out)


def _uuid_or_passthrough(value: Any) -> Any:
    """Accept either a string or a UUID and return a UUID."""
    if isinstance(value, uuid.UUID):
        return value
    return uuid.UUID(str(value))


def _uuid_list(value: Any) -> list[uuid.UUID]:
    """Coerce a list of stringified UUIDs to ``list[uuid.UUID]``."""
    return [_uuid_or_passthrough(v) for v in value]


def _build_registry() -> dict[str, _TaskSpec]:
    """Lazy-imported registry; avoids importing news_service at module load."""
    from news_service.tasks.deliver_digest import _deliver_digest
    from news_service.tasks.deliver_events import _deliver_event_notifications_batch
    from news_service.tasks.discover_sources import _discover_in_task_session
    from news_service.tasks.poll_feeds import _poll_all_feeds
    from news_service.tasks.reflect_events import _reflect_event_subscriptions
    from news_service.tasks.schedule_digests import _schedule_due_digests

    # These exist but aren't triggered by any in-process send_task today.
    # Registered for completeness so an unexpected enqueue doesn't get
    # silently swallowed.
    from news_service.tasks.update_source_embeddings import _update_all as _upd_embeddings
    from news_service.tasks.update_subscription_source_stats import _update_all as _upd_stats

    return {
        "news_service.tasks.deliver_digest.deliver_digest": _TaskSpec(
            impl=_deliver_digest,
            coercions=(_uuid_or_passthrough,),
        ),
        "news_service.tasks.deliver_events.deliver_event_notifications_batch": _TaskSpec(
            impl=_deliver_event_notifications_batch,
            coercions=(_uuid_list,),
        ),
        "news_service.tasks.discover_sources.discover_sources_for_subscription": _TaskSpec(
            impl=_discover_in_task_session,
            coercions=(_uuid_or_passthrough,),
        ),
        "news_service.tasks.poll_feeds.poll_all_feeds": _TaskSpec(impl=_poll_all_feeds),
        "news_service.tasks.reflect_events.reflect_event_subscriptions": _TaskSpec(
            impl=_reflect_event_subscriptions
        ),
        "news_service.tasks.schedule_digests.schedule_due_digests": _TaskSpec(
            impl=_schedule_due_digests
        ),
        "news_service.tasks.update_source_embeddings.update_source_embeddings": _TaskSpec(
            impl=_upd_embeddings
        ),
        "news_service.tasks.update_subscription_source_stats."
        "update_subscription_source_stats": _TaskSpec(impl=_upd_stats),
    }


class CeleryShim:
    """Inline dispatcher for ``celery_app.send_task`` and ``task.delay``.

    Usage:

        shim = CeleryShim()
        shim.install()
        ...  # run benchmark
        shim.uninstall()
    """

    def __init__(self) -> None:
        self._registry: dict[str, _TaskSpec] = {}
        self._originals: dict[str, Any] = {}
        self._pending: set[asyncio.Task[Any]] = set()
        self._installed = False
        # Tasks dispatched via ``send_task`` / ``.delay`` run through a
        # single FIFO serializer rather than as parallel ``create_task``s.
        # The production Celery worker we're replacing serializes per
        # queue (prefetch=1 default), and more importantly, running
        # pipelines (digest, event batch, discovery) concurrently on the
        # SAME asyncpg engine pool triggers
        # ``InterfaceError: another operation is in progress`` during
        # session teardown. Serializing by default avoids that class of
        # race entirely.
        self._serializer_lock: asyncio.Lock | None = None

    def install(self) -> None:
        """Install send_task + .delay shims. Idempotent."""
        if self._installed:
            return
        self._registry = _build_registry()
        # Create the lock eagerly on install so every ``send_task`` /
        # ``.delay`` that follows shares the same instance. Lazy
        # creation in ``_run_serialized`` races -- two tasks dispatched
        # in quick succession each create their own lock, which
        # defeats the whole point of serialization.
        self._serializer_lock = asyncio.Lock()

        from news_service.tasks import celery_app as celery_mod
        from news_service.tasks import deliver_digest as deliver_digest_mod

        self._originals["celery_app.send_task"] = celery_mod.celery_app.send_task
        celery_mod.celery_app.send_task = self._send_task  # type: ignore[assignment]

        # ``trigger_digest_now`` imports deliver_digest lazily inside the
        # tool body and calls ``.delay(subscription_id, notify_if_empty=True)``.
        # Replace the task object's ``.delay`` method so the inline call
        # routes through our shim too.
        self._originals["deliver_digest.delay"] = deliver_digest_mod.deliver_digest.delay
        deliver_digest_mod.deliver_digest.delay = self._delay_for(  # type: ignore[assignment]
            "news_service.tasks.deliver_digest.deliver_digest"
        )
        self._installed = True

    def uninstall(self) -> None:
        """Restore originals. Safe to call without install()."""
        if not self._installed:
            return
        from news_service.tasks import celery_app as celery_mod
        from news_service.tasks import deliver_digest as deliver_digest_mod

        celery_mod.celery_app.send_task = self._originals[  # type: ignore[assignment]
            "celery_app.send_task"
        ]
        deliver_digest_mod.deliver_digest.delay = self._originals[  # type: ignore[assignment]
            "deliver_digest.delay"
        ]
        self._originals.clear()
        self._registry.clear()
        self._installed = False

    async def drain(self) -> None:
        """Await every background task this shim has spawned."""
        if not self._pending:
            return
        await asyncio.gather(*self._pending, return_exceptions=True)

    def _send_task(
        self,
        name: str,
        args: list[Any] | tuple[Any, ...] | None = None,
        kwargs: dict[str, Any] | None = None,
        **_ignored: Any,
    ) -> None:
        """Stand-in for ``celery_app.send_task(name, args=..., kwargs=...)``.

        The production call is fire-and-forget (returns an AsyncResult we
        ignore). We mirror that: spawn the coroutine as a background task
        on the current event loop so the caller doesn't block. Exceptions
        are logged, not raised, because Celery wouldn't surface them to
        the enqueuer either.
        """
        spec = self._registry.get(name)
        if spec is None:
            logger.warning("CeleryShim: unknown task %s -- dropping dispatch", name)
            return
        pos = spec.coerce(tuple(args or ()))
        kw = dict(kwargs or {})
        self._spawn(name, spec.impl(*pos, **kw))

    def _delay_for(self, name: str) -> Callable[..., None]:
        """Build a ``.delay(*args, **kwargs)`` shim bound to a specific task."""

        def _delay(*args: Any, **kwargs: Any) -> None:
            spec = self._registry.get(name)
            if spec is None:
                logger.warning("CeleryShim: unknown task %s -- dropping .delay call", name)
                return
            pos = spec.coerce(args)
            self._spawn(name, spec.impl(*pos, **kwargs))

        return _delay

    def _spawn(self, name: str, coro: Awaitable[Any]) -> None:
        task = asyncio.create_task(self._run_serialized(name, coro))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _run_serialized(self, name: str, coro: Awaitable[Any]) -> None:
        """Acquire the per-shim FIFO lock, then run the task logged."""
        assert self._serializer_lock is not None, "CeleryShim.install() not called"
        async with self._serializer_lock:
            try:
                await coro
            except Exception:
                logger.exception("CeleryShim: task %s raised", name)
