"""
JSON serializer for a single scenario run.

Emits a self-contained record: scenario id, model column, PASS/FAIL
outcomes, classification metrics, rubric scores with rationales, cost
ledger rollups, every captured webhook payload, and the full
conversation transcript. Consumed by the matrix writer and kept as the
raw artifact for post-mortem.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _json_default(o: Any) -> Any:
    if is_dataclass(o):
        return asdict(o)
    if hasattr(o, "model_dump"):
        return o.model_dump()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    raise TypeError(f"not serializable: {type(o).__name__}")


def write_scenario_json(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=_json_default, ensure_ascii=False))
