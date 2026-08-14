#!/usr/bin/env bash
# Optimization smoke (500k) then optional formal 3×2M beat-heuristic run.
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-smoke}"  # smoke | formal
OUT_DIR="/root/autodl-tmp/RL-2048/results"
mkdir -p "$OUT_DIR/experiments" "$OUT_DIR/runs/opt" "$OUT_DIR/runs/opt_smoke"

# shellcheck disable=SC1091
source .venv/bin/activate

if [[ "$MODE" == "smoke" ]]; then
  CONFIG="configs/autodl/opt_smoke.yaml"
  SEEDS="${TRAIN_SEEDS:-1}"
  OUTPUT="$OUT_DIR/experiments/opt_smoke_latest.json"
  LOG="$OUT_DIR/experiments/opt_smoke_$(date -u +%Y%m%dT%H%M%S).log"
else
  CONFIG="configs/autodl/opt_beat_heuristic.yaml"
  SEEDS="${TRAIN_SEEDS:-3}"
  OUTPUT="$OUT_DIR/experiments/opt_beat_heuristic_latest.json"
  LOG="$OUT_DIR/experiments/opt_beat_heuristic_$(date -u +%Y%m%dT%H%M%S).log"
fi

echo "Mode=$MODE config=$CONFIG seeds=$SEEDS"
echo "Logging to $LOG"

python scripts/run_experiment.py \
  --config "$CONFIG" \
  --train-seeds "$SEEDS" \
  --output "$OUTPUT" \
  2>&1 | tee "$LOG"

echo "Done. Summary: $OUTPUT"
