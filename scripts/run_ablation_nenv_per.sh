#!/usr/bin/env bash
# 2×2 ablation: num_envs ∈ {1,8} × use_per ∈ {false,true}, Phase A budget (5M).
#
# Usage:
#   bash scripts/run_ablation_nenv_per.sh              # all 4 cells
#   bash scripts/run_ablation_nenv_per.sh n1_p0        # one cell
#   TRAIN_SEEDS=2 bash scripts/run_ablation_nenv_per.sh
#
# Cells:
#   n1_p0  num_envs=1 PER=off   (control ≈ Phase A)
#   n1_p1  num_envs=1 PER=on
#   n8_p0  num_envs=8 PER=off
#   n8_p1  num_envs=8 PER=on    (Phase B factors @ 5M)
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source .venv/bin/activate

OUT_DIR="/root/autodl-tmp/RL-2048/results"
mkdir -p "$OUT_DIR/experiments" "$OUT_DIR/logs"
SEEDS="${TRAIN_SEEDS:-1}"
SEED_START="${SEED_START:-0}"

ALL_CELLS=(n1_p0 n1_p1 n8_p0 n8_p1)

if [[ $# -eq 0 || "${1:-}" == "all" ]]; then
  CELLS=("${ALL_CELLS[@]}")
else
  CELLS=("$@")
fi

for cell in "${CELLS[@]}"; do
  CONFIG="configs/autodl/abl_${cell}.yaml"
  if [[ ! -f "$CONFIG" ]]; then
    echo "Unknown cell '$cell' (missing $CONFIG)"
    echo "Valid: ${ALL_CELLS[*]}"
    exit 1
  fi
  OUTPUT="$OUT_DIR/experiments/abl_${cell}_latest.json"
  LOG="$OUT_DIR/logs/abl_${cell}_$(date -u +%Y%m%dT%H%M%S).log"
  echo "============================================================"
  echo "[$(date -Is)] START cell=$cell seeds=$SEEDS config=$CONFIG"
  echo "Log: $LOG"
  echo "============================================================"
  python scripts/run_experiment.py \
    --config "$CONFIG" \
    --train-seeds "$SEEDS" \
    --seed-start "$SEED_START" \
    --skip-baselines \
    --output "$OUTPUT" \
    2>&1 | tee "$LOG"
  echo "[$(date -Is)] DONE cell=$cell → $OUTPUT"
done

python scripts/summarize_ablation.py || true
echo "[$(date -Is)] Ablation pipeline finished."
