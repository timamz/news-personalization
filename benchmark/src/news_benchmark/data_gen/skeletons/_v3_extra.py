"""V3 bulk-expansion loader.

Reads JSON banks written by ``scripts/gen_v3_banks.py`` and returns
``TimelineEntry`` objects via the existing ``bulk()`` helper. Kept
separate from the skeleton files so generated content doesn't bloat
the hand-authored scenario Python.

Banks live at ``data/scenarios/<scenario>/_v3_banks/<tier>__<theme>.json``.
If the directory doesn't exist (v2 state, or the generator hasn't run
yet) the function returns an empty list and the skeleton still works.

Usage:

    from news_benchmark.data_gen.skeletons import _v3_extra
    items += _v3_extra.load_for("s01", sub_id=SUB, start=START)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

from news_benchmark.data_gen.skeletons._bulk import bulk
from news_benchmark.scenarios.base import TimelineEntry

logger = logging.getLogger(__name__)

_BENCH_ROOT = Path(__file__).resolve().parents[4]
_DATA_DIR = _BENCH_ROOT / "data" / "scenarios"


def _bank_dir(scenario_key: str) -> Path:
    return _DATA_DIR / scenario_key / "_v3_banks"


def load_for(
    scenario_key: str,
    *,
    sub_id: str,
    start: datetime,
    hour_offset: int = 10,
    days_shift: int = 0,
) -> list[TimelineEntry]:
    """Load every bank for ``scenario_key`` into TimelineEntry objects.

    Arguments:
        scenario_key: folder name under data/scenarios/ (e.g. "s01",
            "s03", "s05-ai"). Must match what ``gen_v3_banks.py`` wrote.
        sub_id: subscription label to assign. Bulk-generated items are
            always labeled ``positive=False`` since v3 banks only produce
            easy_negative and near_miss_negative tiers.
        start: the scenario start datetime. Items are spread across 29
            days from this anchor.
        hour_offset: base hour-of-day (items drift further by index to
            avoid a single-timestamp stampede).
        days_shift: days to add to start; useful when one scenario
            loads banks for multiple tiers and wants timestamp disjoint
            blocks.
    """
    root = _bank_dir(scenario_key)
    if not root.is_dir():
        return []

    out: list[TimelineEntry] = []
    anchor = start + timedelta(days=days_shift, hours=hour_offset)
    for bank_path in sorted(root.glob("*.json")):
        try:
            payload = json.loads(bank_path.read_text())
        except Exception as exc:
            logger.warning("_v3_extra: failed to load %s: %s", bank_path, exc)
            continue
        tier = payload.get("tier")
        rows = payload.get("rows") or []
        if not rows or not tier:
            continue
        style_cycle = tuple(payload.get("style_cycle") or ("newsroom",))
        tuple_rows = [
            (str(r[0]), str(r[1]), str(r[2]) if len(r) > 2 else "")
            for r in rows
            if isinstance(r, list) and len(r) >= 2
        ]
        out.extend(
            bulk(
                sub_id,
                anchor,
                spread_days=29,
                difficulty=str(tier),
                positive=False,
                rows=tuple_rows,
                style_cycle=style_cycle,
            )
        )
    return out
