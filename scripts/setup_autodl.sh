#!/usr/bin/env bash
# AutoDL cloud setup: reuse image PyTorch, install project deps only.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== AutoDL environment setup ==="

if ! command -v nvidia-smi &>/dev/null; then
  echo "WARNING: nvidia-smi not found; GPU may be unavailable."
else
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
fi

PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "ERROR: system PyTorch CUDA not available."
  echo "Use AutoDL image: PyTorch 2.5.1 / Python 3.12 / CUDA 12.4"
  "$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || true
  exit 1
fi

"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"

if [[ ! -d .venv ]]; then
  "$PYTHON" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

python -m pip install -U pip
# Do not reinstall torch — use the CUDA build from the AutoDL image.
pip install -e ".[train]" --no-deps
pip install "gymnasium>=1.0.0" "numpy>=1.26.0" "pyyaml>=6.0" "pytest>=8.0.0"

echo ""
echo "=== Smoke tests ==="
pytest -m "not slow" -q

echo ""
echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Quick GPU train check:"
echo "  rl2048-train --config configs/autodl/e1_smoke.yaml --train-seed 0"
