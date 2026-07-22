#!/usr/bin/env bash
# Run on AutoDL after creating an instance. Clones repo and sets up the environment.
set -euo pipefail

REPO_URL="${1:-}"
TARGET="/root/autodl-tmp/RL-2048"

if [[ -z "$REPO_URL" ]]; then
  echo "Usage: bash scripts/autodl_bootstrap.sh <git-repo-url>"
  echo "Example: bash autodl_bootstrap.sh https://github.com/you/RL-2048.git"
  exit 1
fi

mkdir -p /root/autodl-tmp
cd /root/autodl-tmp

if [[ -d RL-2048 ]]; then
  echo "Directory exists; pulling latest..."
  cd RL-2048
  git pull
else
  git clone "$REPO_URL" RL-2048
  cd RL-2048
fi

bash scripts/setup_autodl.sh
echo ""
echo "Bootstrap complete. Project at: $TARGET"
echo "Next: tmux new -s train  (then run smoke or experiment scripts)"
