"""
FakeClock provides a deterministic virtual clock for the benchmark.

Monkey-patches datetime.datetime with a subclass whose now()/utcnow() read
from the shared FakeClock instance. MUST be installed via
install_clock_patch() before any news_service.* import, otherwise modules
that cached datetime.datetime at import time will see the real clock.

Usage:

    from news_benchmark.clock import CLOCK, install_clock_patch
    install_clock_patch(start=datetime(2026, 4, 1, tzinfo=UTC))
    # now every `datetime.datetime.now()` returns CLOCK.now()
    CLOCK.advance_to(CLOCK.now() + timedelta(minutes=30))
"""

from __future__ import annotations

import datetime as _dt_module
from datetime import UTC, datetime, timedelta


class FakeClock:
    """Shared mutable clock state read by the patched datetime class."""

    def __init__(self, start: datetime) -> None:
        if start.tzinfo is None:
            raise ValueError("FakeClock start must be timezone-aware")
        self._now: datetime = start

    def now(self) -> datetime:
        """Return the current virtual time as a timezone-aware datetime."""
        return self._now

    def advance_to(self, target: datetime) -> None:
        """Move the clock forward to `target`. Refuses to move backwards."""
        if target < self._now:
            raise ValueError(f"FakeClock cannot move backwards: {self._now} -> {target}")
        self._now = target

    def advance_by(self, delta: timedelta) -> None:
        """Move the clock forward by `delta`."""
        self.advance_to(self._now + delta)


CLOCK = FakeClock(datetime(2026, 1, 1, tzinfo=UTC))


class _FakeDatetime(_dt_module.datetime):
    """datetime.datetime subclass whose now() reads the shared FakeClock."""

    @classmethod
    def now(cls, tz: _dt_module.tzinfo | None = None) -> _FakeDatetime:  # type: ignore[override]
        current = CLOCK.now()
        if tz is None:
            return cls._wrap(current.replace(tzinfo=None))
        return cls._wrap(current.astimezone(tz))

    @classmethod
    def utcnow(cls) -> _FakeDatetime:  # type: ignore[override]
        return cls._wrap(CLOCK.now().replace(tzinfo=None))

    @classmethod
    def _wrap(cls, d: datetime) -> _FakeDatetime:
        return cls(d.year, d.month, d.day, d.hour, d.minute, d.second, d.microsecond, d.tzinfo)


_patched = False


def install_clock_patch(start: datetime | None = None) -> None:
    """Install the FakeClock datetime patch. Idempotent.

    MUST be called before any news_service.* module is imported. run.py
    enforces this by doing zero news_service imports at module level.
    """
    global _patched
    if start is not None:
        CLOCK.__init__(start)
    if _patched:
        return
    _dt_module.datetime = _FakeDatetime  # type: ignore[misc]
    _patched = True


def assert_patched() -> None:
    """Fail loudly if any scenario has drifted back to the real clock."""
    if _dt_module.datetime is not _FakeDatetime:
        raise RuntimeError(
            "FakeClock patch is not installed; a module likely imported "
            "datetime before install_clock_patch() ran"
        )
