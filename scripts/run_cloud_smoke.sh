#!/usr/bin/env bash
# GPU smoke test (~100k steps). Run inside tmux on AutoDL.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"

# shellcheck disable=SC1091
source .venv/bin/activate

echo "=== Baseline eval (dev 200) ==="
rl2048-eval --episodes 200 --seed-set dev

echo "=== Smoke train (100k steps) ==="
rl2048-train --config configs/autodl/e1_smoke.yaml --train-seed 0

echo "=== Smoke eval ==="
RUN_ROOT="${ROOT}/results/runs/e1_smoke"
LATEST=$(find "$RUN_ROOT" -name 'checkpoint_final.pt' 2>/dev/null | sort | tail -1 || true)
if [[ -z "${LATEST}" ]]; then
  LATEST=$(find "$RUN_ROOT" -name 'checkpoint_*.pt' 2>/dev/null | sort -V | tail -1 || true)
fi
if [[ -n "${LATEST}" ]]; then
  echo "Evaluating: $LATEST"
  rl2048-eval --checkpoint "$LATEST" --episodes 200 --seed-set dev
else
  echo "ERROR: no checkpoint found under $RUN_ROOT" >&2
  exit 1
fi

echo "Smoke test complete."
