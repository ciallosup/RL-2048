#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

MODE="${1:-venv}"

if [[ "$MODE" == "venv" ]]; then
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  python -m pip install -U pip
  pip install -e ".[dev,train]"
  echo ""
  echo "Virtual environment ready: .venv"
  echo "Activate: source .venv/bin/activate"
else
  conda create -p ./.conda/env python=3.11 -y
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate ./.conda/env
  python -m pip install -U pip
  pip install -e ".[dev,train]"
  echo ""
  echo "Conda environment ready: .conda/env"
  echo "Activate: conda activate ./.conda/env"
fi

echo "Run visualizer: rl2048-play"
