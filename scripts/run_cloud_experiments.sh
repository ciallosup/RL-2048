#!/usr/bin/env bash
# Full baseline + E1/E2 multi-seed experiments. Run inside tmux on AutoDL.
set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN_SEEDS="${TRAIN_SEEDS:-5}"

# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== Baseline eval ==="
rl2048-eval --episodes 200 --seed-set dev
rl2048-eval --episodes 1000 --seed-set val

echo "=== E1 Double DQN (${TRAIN_SEEDS} seeds) ==="
python scripts/run_experiment.py \
  --config configs/autodl/e1_min_dqn.yaml \
  --train-seeds "$TRAIN_SEEDS" \
  --output /root/autodl-tmp/RL-2048/results/experiments/e1_latest.json

echo "=== E2 Vanilla DQN (${TRAIN_SEEDS} seeds) ==="
python scripts/run_experiment.py \
  --config configs/autodl/e2_vanilla_dqn.yaml \
  --train-seeds "$TRAIN_SEEDS" \
  --output /root/autodl-tmp/RL-2048/results/experiments/e2_latest.json

echo "All experiments complete."
