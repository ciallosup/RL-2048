#!/usr/bin/env bash
# Wait for Phase A formal to finish, then run Phase B (3×10M + PER).
set -euo pipefail
cd "$(dirname "$0")/.."
OUT_DIR="/root/autodl-tmp/RL-2048/results/experiments"
SUMMARY="$OUT_DIR/opt_tile1024_latest.json"
PHASEB_MARKER="$OUT_DIR/opt_tile2048_latest.json"

# shellcheck disable=SC1091
source .venv/bin/activate

echo "[$(date -Is)] Waiting for Phase A summary: $SUMMARY"
while true; do
  if [[ -f "$SUMMARY" ]] && python3 - <<'PY'
import json
from pathlib import Path
p=Path("/root/autodl-tmp/RL-2048/results/experiments/opt_tile1024_latest.json")
try:
    d=json.loads(p.read_text())
except Exception:
    raise SystemExit(1)
ok = bool(d.get("runs")) and "dqn_aggregate" in d and len(d["runs"]) >= 3
raise SystemExit(0 if ok else 1)
PY
  then
    # ensure trainer process for phaseA is gone
    if ! pgrep -f "configs/autodl/opt_tile1024.yaml" >/dev/null 2>&1; then
      break
    fi
  fi
  sleep 120
done

echo "[$(date -Is)] Phase A complete. Aggregate:"
python3 - <<'PY'
import json
from pathlib import Path
d=json.loads(Path("/root/autodl-tmp/RL-2048/results/experiments/opt_tile1024_latest.json").read_text())
a=d["dqn_aggregate"]
print(f"P1024={100*a['p_reach_1024']['mean']:.2f}% score={a['mean_score']['mean']:.1f} P2048={100*a['p_reach_2048']['mean']:.2f}%")
gate = a['p_reach_1024']['mean'] >= 0.05
print("PhaseA gate:", "PASS" if gate else "FAIL (still launching Phase B as planned pipeline)")
PY

if [[ -f "$PHASEB_MARKER" ]]; then
  echo "Phase B summary already exists; skip re-run."
  exit 0
fi

echo "[$(date -Is)] Starting Phase B (1 seed first for faster signal; set TRAIN_SEEDS=3 to full)..."
TRAIN_SEEDS="${TRAIN_SEEDS:-1}" bash scripts/run_tile_opt.sh phaseB

echo "[$(date -Is)] Writing comparison report..."
python3 scripts/summarize_tile_runs.py || true
echo "Pipeline complete."
