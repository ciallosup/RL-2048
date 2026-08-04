#!/usr/bin/env bash
# Full baseline + E1/E2 multi-seed experiments. Run inside tmux on AutoDL.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

TRAIN_SEEDS="${TRAIN_SEEDS:-5}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"

# shellcheck disable=SC1091
source .venv/bin/activate

mkdir -p "${ROOT}/results/experiments" "${ROOT}/results/runs/e1" "${ROOT}/results/runs/e2_vanilla"

echo "=== Multi-seed experiments ==="
echo "project_root=${ROOT}"
echo "train_seeds=${TRAIN_SEEDS}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"

if [[ "${SKIP_BASELINE}" != "1" ]]; then
  echo "=== Baseline eval ==="
  rl2048-eval --episodes 200 --seed-set dev
  rl2048-eval --episodes 1000 --seed-set val
fi

echo "=== E1 Double DQN (${TRAIN_SEEDS} seeds) ==="
python scripts/run_experiment.py \
  --config configs/autodl/e1_min_dqn.yaml \
  --train-seeds "$TRAIN_SEEDS" \
  --output "${ROOT}/results/experiments/e1_latest.json"

echo "=== E2 Vanilla DQN (${TRAIN_SEEDS} seeds) ==="
python scripts/run_experiment.py \
  --config configs/autodl/e2_vanilla_dqn.yaml \
  --train-seeds "$TRAIN_SEEDS" \
  --output "${ROOT}/results/experiments/e2_latest.json"

echo "=== Aggregate summary ==="
python scripts/summarize_experiments.py \
  --e1 "${ROOT}/results/experiments/e1_latest.json" \
  --e2 "${ROOT}/results/experiments/e2_latest.json" \
  --output "${ROOT}/results/experiments/e1_e2_compare.json"

echo "All experiments complete."
echo "Results:"
echo "  ${ROOT}/results/experiments/e1_latest.json"
echo "  ${ROOT}/results/experiments/e2_latest.json"
echo "  ${ROOT}/results/experiments/e1_e2_compare.json"
