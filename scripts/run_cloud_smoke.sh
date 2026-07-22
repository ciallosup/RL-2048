#!/usr/bin/env bash
# GPU smoke test (~100k steps). Run inside tmux on AutoDL.
set -euo pipefail
cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== Baseline eval (dev 200) ==="
rl2048-eval --episodes 200 --seed-set dev

echo "=== Smoke train (100k steps) ==="
rl2048-train --config configs/autodl/e1_smoke.yaml --train-seed 0

echo "=== Smoke eval ==="
LATEST=$(find /root/autodl-tmp/RL-2048/results/runs/e1_smoke -name 'checkpoint_final.pt' | sort | tail -1)
if [[ -n "$LATEST" ]]; then
  rl2048-eval --checkpoint "$LATEST" --episodes 200 --seed-set dev
fi

echo "Smoke test complete."
