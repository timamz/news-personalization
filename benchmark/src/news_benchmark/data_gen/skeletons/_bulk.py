"""
Helper for bulk-generating negative-tier timeline items from compact tuples.

Keeps scenario skeletons readable when dozens of negatives are needed to
match real-world feed volumes. Headlines are still hand-authored; this
only spares the boilerplate of one `_e(...)` call per row.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from itertools import cycle

from news_benchmark.scenarios.base import TimelineEntry


def bulk(
    sub_id: str,
    start: datetime,
    spread_days: float,
    difficulty: str,
    positive: bool,
    rows: list[tuple[str, str, str]],
    style_cycle: tuple[str, ...] = ("newsroom",),
) -> list[TimelineEntry]:
    """Expand (source_url, headline, _hint) rows into TimelineEntry objects.

    Items are distributed evenly across `spread_days`; style rotates
    through `style_cycle` to inject prose variation.
    """
    if not rows:
        return []
    step = spread_days / max(1, len(rows))
    styles = cycle(style_cycle)
    out: list[TimelineEntry] = []
    for i, (source, headline, _hint) in enumerate(rows):
        ts = start + timedelta(days=i * step, hours=(i * 3) % 24)
        out.append(
            TimelineEntry(
                fake_ts=ts.isoformat(),
                source_url=source,
                headline=headline,
                difficulty=difficulty,
                should_notify_per_sub={sub_id: positive},
                should_contribute_to_digest_per_sub={sub_id: positive},
                body_style_hint=next(styles),
                body_adversarial=False,
                body_language="en",
            )
        )
    return out
