#!/usr/bin/env python3
"""C1 fine-tune vs Phase A: greedy val 200 + 2-ply play-out n=20."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rl2048.env import Game2048Env
from rl2048.eval.metrics import distribution_summary, tile_reach_probs, wilson_interval
from rl2048.eval.runner import evaluate_policy, run_episode, summary_to_dict
from rl2048.eval.seeds import val_seeds
from rl2048.policies.dqn_policy import DQNPolicy
from rl2048.policies.expectimax import ExpectimaxDQNPolicy

OUT = Path("/root/autodl-tmp/RL-2048/results/experiments/c1_vs_phaseA.json")
C1_DIR = Path("/root/autodl-tmp/RL-2048/results/runs/c1_finetune_4096/c1_finetune_4096_seed0_20260815T090841")
PHASE_A = Path("checkpoints/phaseA_dueling_seed0.pt")
CKPTS = [
    ("phaseA", PHASE_A),
    ("c1_200k", C1_DIR / "checkpoint_200000.pt"),
    ("c1_final", C1_DIR / "checkpoint_final.pt"),
]


def _print(label: str, summary: dict) -> None:
    tiles = summary.get("tile_reach_probs", {})
    print(
        f"  {label:<16} mean={summary['score_stats']['mean']:7.1f} "
        f"P1024={100 * summary['p_reach_1024']:5.1f}% "
        f"P2048={100 * summary['p_reach_2048']:5.1f}% "
        f"P4096={100 * float(tiles.get('P(>=4096)', 0.0)):5.1f}% "
        f"trunc={100 * summary['truncation_rate']:4.1f}%",
        flush=True,
    )


def playout_rows(policy, seeds, *, key: str, max_steps: int) -> list[dict]:
    env = Game2048Env(max_episode_steps=max_steps)
    rows = []
    t0 = time.perf_counter()
    for i, seed in enumerate(seeds, start=1):
        ep = run_episode(env, policy, seed=seed, policy_key=key, stop_on_2048=False)
        rows.append(
            {
                "seed": int(seed),
                "score": int(ep.game_score),
                "max_tile": int(ep.max_tile),
                "length": int(ep.episode_length),
                "truncated": bool(ep.truncated),
            }
        )
        n = len(rows)
        n2048 = sum(r["max_tile"] >= 2048 for r in rows)
        n4096 = sum(r["max_tile"] >= 4096 for r in rows)
        mean = float(np.mean([r["score"] for r in rows]))
        print(
            f"    {key} {i}/{len(seeds)} seed={seed} score={ep.game_score} "
            f"max={ep.max_tile} mean={mean:.0f} "
            f"P2048={100 * n2048 / n:.1f}% P4096={100 * n4096 / n:.1f}% "
            f"({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
    return rows


def summarize_playout(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    scores = [r["score"] for r in rows]
    tiles = [r["max_tile"] for r in rows]
    n2048 = sum(t >= 2048 for t in tiles)
    n4096 = sum(t >= 4096 for t in tiles)
    ci2048 = wilson_interval(n2048, n)
    ci4096 = wilson_interval(n4096, n)
    return {
        "name": name,
        "n": n,
        "mean_score": float(np.mean(scores)),
        "score_stats": distribution_summary(scores),
        "p_reach_1024": sum(t >= 1024 for t in tiles) / n,
        "p_reach_2048": n2048 / n,
        "p_reach_2048_ci": [ci2048.lower, ci2048.upper],
        "p_reach_4096": n4096 / n,
        "p_reach_4096_ci": [ci4096.lower, ci4096.upper],
        "tile_reach_probs": tile_reach_probs(tiles),
        "truncation_rate": float(np.mean([r["truncated"] for r in rows])),
        "episodes": rows,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "greedy": [],
        "playout_2ply": {},
    }
    greedy_seeds = val_seeds(200)
    print(f"=== greedy val {len(greedy_seeds)} max_steps=4000 ===", flush=True)
    for name, path in CKPTS:
        if not path.exists():
            print(f"  skip missing {path}", flush=True)
            continue
        t0 = time.perf_counter()
        policy = DQNPolicy.from_checkpoint(path, decode="greedy")
        summary = evaluate_policy(
            policy,
            policy_key=f"{name}_greedy",
            policy_label=name,
            seeds=greedy_seeds,
            max_episode_steps=4000,
            stop_on_2048=False,
        )
        row = summary_to_dict(summary)
        row["elapsed_sec"] = time.perf_counter() - t0
        row["checkpoint"] = str(path)
        _print(name, row)
        payload["greedy"].append(row)
        OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    play_seeds = val_seeds(20)
    print(f"\n=== 2-ply play-out n={len(play_seeds)} max_steps=8000 ===", flush=True)
    for name, path in (("phaseA", PHASE_A), ("c1_final", C1_DIR / "checkpoint_final.pt")):
        policy = ExpectimaxDQNPolicy.from_checkpoint(
            path, depth=2, adaptive=False, corner_tiebreak=True
        )
        rows = playout_rows(policy, play_seeds, key=f"{name}_d2", max_steps=8000)
        payload["playout_2ply"][name] = summarize_playout(name, rows)
        summary = payload["playout_2ply"][name]
        print(
            f"  DONE {name} mean={summary['mean_score']:.0f} "
            f"P2048={100 * summary['p_reach_2048']:.1f}% "
            f"P4096={100 * summary['p_reach_4096']:.1f}%",
            flush=True,
        )
        OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Saved", OUT, flush=True)


if __name__ == "__main__":
    main()
