#!/usr/bin/env bash
# After s01 and s03 fabric have been regenerated at v3 scale, prime
# derived scenarios' body caches from them (hashes match because
# headline_hash depends on source+headline+style+adversarial+lang —
# not labels or scenario id). Then run data_gen for the derived
# scenarios, which will only need to generate the extension bodies
# (s02's 32 drift items, s04's 5 buried misses, s05's new AI-sub
# items that didn't exist in s01/s03).

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$BENCH_DIR/../../../../" && pwd)"
BACKEND_ENV="$REPO_ROOT/backend/.env"
PY="$BENCH_DIR/.venv/bin/python"

set -a
source "$BACKEND_ENV"
set +a

cd "$BENCH_DIR"

SCENARIOS_DIR="data/scenarios"

# --- s02: inherits s01 ---
if [[ -f "$SCENARIOS_DIR/s01/bodies_cache.json" ]]; then
  echo "[prime] s02 <- s01 bodies_cache"
  cp "$SCENARIOS_DIR/s01/bodies_cache.json" "$SCENARIOS_DIR/s02/bodies_cache.json"
else
  echo "[prime] WARN: s01 bodies_cache.json missing; s02 will regenerate from scratch"
fi

# --- s04: inherits s03 ---
if [[ -f "$SCENARIOS_DIR/s03/bodies_cache.json" ]]; then
  echo "[prime] s04 <- s03 bodies_cache"
  cp "$SCENARIOS_DIR/s03/bodies_cache.json" "$SCENARIOS_DIR/s04/bodies_cache.json"
else
  echo "[prime] WARN: s03 bodies_cache.json missing; s04 will regenerate from scratch"
fi

# --- s05: inherits both; merge s01 + s03 into s05 cache ---
"$PY" - <<'PY'
import json, os
from pathlib import Path
base = Path("data/scenarios")
c01 = base / "s01" / "bodies_cache.json"
c03 = base / "s03" / "bodies_cache.json"
s05_cache_path = base / "s05" / "bodies_cache.json"
merged: dict[str, str] = {}
if s05_cache_path.exists():
    merged.update(json.loads(s05_cache_path.read_text()))
for p in (c01, c03):
    if p.exists():
        merged.update(json.loads(p.read_text()))
        print(f"[prime] s05 <- {p} ({len(merged)} total after merge)")
s05_cache_path.parent.mkdir(parents=True, exist_ok=True)
s05_cache_path.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
PY

# --- run data_gen for derived scenarios ---
for sid in s02 s04 s05; do
  echo "==================== $sid ===================="
  date
  "$PY" -u -m news_benchmark.data_gen.cli "$sid" --no-judge
  echo
done
echo "[DONE] derived fabric regenerated"
