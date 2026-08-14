#!/usr/bin/env bash
# Phase A/B tile-horizon experiments.
# Usage: bash scripts/run_tile_opt.sh smoke|phaseA|phaseB
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-smoke}"
OUT_DIR="/root/autodl-tmp/RL-2048/results"
mkdir -p "$OUT_DIR/experiments"

# shellcheck disable=SC1091
source .venv/bin/activate

case "$MODE" in
  smoke)
    CONFIG="configs/autodl/opt_tile1024_smoke.yaml"
    SEEDS="${TRAIN_SEEDS:-1}"
    OUTPUT="$OUT_DIR/experiments/opt_tile1024_smoke_latest.json"
    ;;
  phaseA|a|A)
    CONFIG="configs/autodl/opt_tile1024.yaml"
    SEEDS="${TRAIN_SEEDS:-3}"
    OUTPUT="$OUT_DIR/experiments/opt_tile1024_latest.json"
    ;;
  phaseB|b|B)
    CONFIG="configs/autodl/opt_tile2048.yaml"
    SEEDS="${TRAIN_SEEDS:-3}"
    OUTPUT="$OUT_DIR/experiments/opt_tile2048_latest.json"
    ;;
  phaseB_v2|b2|B2)
    CONFIG="configs/autodl/opt_tile2048_v2.yaml"
    SEEDS="${TRAIN_SEEDS:-3}"
    OUTPUT="$OUT_DIR/experiments/opt_tile2048_v2_latest.json"
    ;;
  *)
    echo "Usage: $0 smoke|phaseA|phaseB|phaseB_v2"
    exit 1
    ;;
esac

LOG="${OUTPUT%.json}_$(date -u +%Y%m%dT%H%M%S).log"
echo "Mode=$MODE config=$CONFIG seeds=$SEEDS"
echo "Logging to $LOG"

python scripts/run_experiment.py \
  --config "$CONFIG" \
  --train-seeds "$SEEDS" \
  --output "$OUTPUT" \
  2>&1 | tee "$LOG"

echo "Done. Summary: $OUTPUT"
