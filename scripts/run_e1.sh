#!/usr/bin/env bash
# Formal E1: Masked Double DQN, 5 seeds × 500k steps, val 1000 eval.
# Compare against random / heuristic / greedy on the same seed set.
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_SEEDS="${TRAIN_SEEDS:-5}"
OUT_DIR="/root/autodl-tmp/RL-2048/results"
mkdir -p "$OUT_DIR/experiments" "$OUT_DIR/runs/e1"

# shellcheck disable=SC1091
source .venv/bin/activate

LOG="$OUT_DIR/experiments/e1_$(date -u +%Y%m%dT%H%M%S).log"
echo "Logging to $LOG"
echo "Config: configs/autodl/e1_min_dqn.yaml | seeds=$TRAIN_SEEDS | steps=500000 | eval=val/1000"

python scripts/run_experiment.py \
  --config configs/autodl/e1_min_dqn.yaml \
  --train-seeds "$TRAIN_SEEDS" \
  --output "$OUT_DIR/experiments/e1_latest.json" \
  2>&1 | tee "$LOG"

echo "E1 complete. Summary: $OUT_DIR/experiments/e1_latest.json"
