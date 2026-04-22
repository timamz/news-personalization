#!/usr/bin/env bash
# Regenerate v3 fabric for all 5 scenarios. Run after gen_v3_banks.py finishes.
#
# Usage: ./scripts/regen_v3_fabric.sh
#
# Each scenario runs through `python -m news_benchmark.data_gen.cli` with
# --no-judge (skip the LLM label-consistency audit; it's a cost/time hog
# at v3 scale and it's a warning gate, not a hard gate).
#
# Body generation is now semaphore-capped (25 concurrent) so the proxy
# doesn't get hammered. Scenarios run sequentially; within a scenario,
# body generation is parallel.

set -euo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "$BENCH_DIR/../../../../" && pwd)"
BACKEND_ENV="$REPO_ROOT/backend/.env"
PY="$BENCH_DIR/.venv/bin/python"

set -a
# shellcheck disable=SC1090
source "$BACKEND_ENV"
set +a

cd "$BENCH_DIR"
for sid in s01 s03 s02 s04 s05; do
  echo "==================== $sid ===================="
  date
  "$PY" -u -m news_benchmark.data_gen.cli "$sid" --no-judge
  echo
done
echo "[DONE] v3 fabric regenerated for all scenarios"
