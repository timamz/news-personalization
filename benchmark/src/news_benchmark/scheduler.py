"""
VirtualScheduler is a priority-queue event loop driving the benchmark.

The production system has four time-driven things: RSS/TG/Reddit polling
every 30 minutes, digest cron schedules, the event verifier daily tick,
and the scenario's content-drop timeline. The scheduler keeps a min-heap
of (fire_at, callback) entries, advances the FakeClock to whichever fires
next, and awaits the callback. No asyncio sleep; wall time is spent only
inside real LLM calls.

Callbacks can enqueue follow-ups (e.g. a poll cycle schedules the next
poll 30 minutes later). Scheduling is open-ended; the scheduler stops
when the heap is empty or the simulated-days budget is exhausted.

Usage:

    sched = VirtualScheduler()
    sched.schedule(start, poll_tick)
    sched.schedule(start + timedelta(hours=9), digest_cron_tick)
    await sched.run(until=start + timedelta(days=30))
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime

from news_benchmark.clock import CLOCK

Callback = Callable[[], Awaitable[None]]


@dataclass(order=True)
class _Event:
    fire_at: datetime
    seq: int
    cb: Callback = field(compare=False)
    label: str = field(compare=False, default="")


class VirtualScheduler:
    """Priority-queue async scheduler that advances a FakeClock between events."""

    def __init__(self) -> None:
        self._heap: list[_Event] = []
        self._seq = itertools.count()
        self._cancelled: set[int] = set()

    def schedule(self, fire_at: datetime, cb: Callback, label: str = "") -> int:
        """Schedule `cb` to fire at `fire_at` (virtual time). Returns a handle id."""
        seq = next(self._seq)
        heapq.heappush(self._heap, _Event(fire_at, seq, cb, label))
        return seq

    def cancel(self, handle: int) -> None:
        """Mark a previously scheduled event as cancelled."""
        self._cancelled.add(handle)

    async def run(self, until: datetime) -> None:
        """Drain the heap, firing each event in order, stopping at `until`."""
        while self._heap:
            ev = heapq.heappop(self._heap)
            if ev.seq in self._cancelled:
                continue
            if ev.fire_at > until:
                heapq.heappush(self._heap, ev)
                break
            CLOCK.advance_to(ev.fire_at)
            await ev.cb()
