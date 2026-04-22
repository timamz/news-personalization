#!/usr/bin/env bash
set -eo pipefail

# Launch N scenarios in parallel. Each runs as its own subprocess with
# its own throwaway postgres DB (news_bench_<run_id>) and its own
# results dir; no shared state between runs.
#
# Usage:
#   ./run_parallel.sh                         # all known scenarios
#   ./run_parallel.sh s01 s02 s03 s04         # specific ones
#   VERBOSE=0 ./run_parallel.sh s01 s02       # suppress LLM-IN/OUT traces
#   OUT_DIR=results/parallel_X ./run_parallel.sh ...
#
# Logs stream to /tmp/bench_<scenario>.log. After all runs finish, the
# script greps each log for errors/rate-limits and prints the per-
# scenario classification report (tp/fp/fn/f1) from the written JSON.

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$BENCH_DIR/../../../../" && pwd)"
BACKEND_ENV="$REPO_ROOT/backend/.env"
PY="$BENCH_DIR/.venv/bin/python"
OUT_DIR="${OUT_DIR:-results/parallel_$(date +%Y%m%d_%H%M%S)}"
SEED="${SEED:-42}"
REPEAT="${REPEAT:-1}"
VERBOSE="${VERBOSE:-1}"

if [[ ! -f "$BACKEND_ENV" ]]; then
  echo "backend .env not found at $BACKEND_ENV" >&2
  exit 1
fi
if [[ ! -x "$PY" ]]; then
  echo "python venv missing at $PY (run 'uv sync' in $BENCH_DIR first)" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$BACKEND_ENV"
set +a

scenarios=("$@")
if [[ ${#scenarios[@]} -eq 0 ]]; then
  mapfile -t scenarios < <(ls "$BENCH_DIR/data/scenarios" | sort)
fi

echo "[parallel] scenarios: ${scenarios[*]}"
echo "[parallel] out_dir:   $OUT_DIR"
echo "[parallel] seed=$SEED repeat=$REPEAT verbose=$VERBOSE"
echo

verbose_flag=()
if [[ "$VERBOSE" == "1" ]]; then
  verbose_flag=(--verbose)
fi

pids=()
logs=()
for s in "${scenarios[@]}"; do
  log="/tmp/bench_${s}.log"
  logs+=("$log")
  cd "$BENCH_DIR"
  nohup "$PY" -u -m news_benchmark.run \
    --scenarios "$s" \
    --models default \
    --seed "$SEED" \
    --repeat "$REPEAT" \
    --out-dir "$OUT_DIR" \
    "${verbose_flag[@]}" \
    > "$log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "[parallel] $s -> pid=$pid log=$log"
done

echo
echo "[parallel] waiting for ${#pids[@]} runs to finish..."
failed=()
for i in "${!pids[@]}"; do
  s="${scenarios[$i]}"
  p="${pids[$i]}"
  l="${logs[$i]}"
  if wait "$p"; then
    echo "[parallel] $s: completed"
  else
    rc=$?
    echo "[parallel] $s: FAILED rc=$rc (see $l)"
    failed+=("$s")
  fi
done

echo
echo "=========================================================="
echo " per-scenario summary"
echo "=========================================================="
for i in "${!scenarios[@]}"; do
  s="${scenarios[$i]}"
  log="${logs[$i]}"
  err=$(grep -cE "ERROR|Traceback" "$log" 2>/dev/null || echo 0)
  rl=$(grep -cE "RateLimitError|429|rate_limit_exceeded" "$log" 2>/dev/null || echo 0)
  llm=$(grep -cE "LLM-OUT" "$log" 2>/dev/null || echo 0)
  echo
  echo "[$s] log=$log errors=$err rate_limits=$rl llm_calls=$llm"
  shopt -s nullglob
  for f in "$BENCH_DIR/$OUT_DIR"/*/scenarios/"${s}"__*.json; do
    "$PY" - "$f" <<'PY'
import json, sys
from pathlib import Path
p = Path(sys.argv[1])
rec = json.loads(p.read_text())
cls = rec.get("classification", {})
per = cls.get("per_sub", {})
print(f"  {p.name}")
for sub, m in per.items():
    tp = m.get("tp"); fp = m.get("fp"); fn = m.get("fn"); tn = m.get("tn")
    f1 = m.get("f1"); pr = m.get("precision"); rc = m.get("recall")
    print(f"    sub={sub[:8]} tp={tp} fp={fp} fn={fn} tn={tn} precision={pr} recall={rc} f1={f1}")
print(f"    cost_usd={rec.get('cost',{}).get('total_usd')}")
print(f"    notes={rec.get('notes')}")
PY
  done
  shopt -u nullglob
done

echo
if [[ ${#failed[@]} -gt 0 ]]; then
  echo "[parallel] ${#failed[@]} run(s) failed: ${failed[*]}"
  exit 1
fi
echo "[parallel] all runs completed"
