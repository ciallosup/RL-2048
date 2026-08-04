#!/usr/bin/env bash
set -euo pipefail
echo "REMOTE_OK"
hostname
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
cd /root/autodl-tmp/RL-2048
git remote -v
git status -sb
git log -1 --oneline
test -f scripts/setup_autodl.sh && echo setup_script_ok
test -f configs/autodl/e1_smoke.yaml && echo autodl_config_ok
if [[ -d .venv ]]; then
  echo venv_ok
  source .venv/bin/activate
  rl2048-train --help >/dev/null && echo cli_ok
  pytest -m "not slow" -q
else
  echo venv_missing
fi
