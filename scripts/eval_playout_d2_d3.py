#!/usr/bin/env python3
"""Paired 2-ply vs 3-ply play-out (no stop on 2048). Resumes from JSON."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rl2048.env import Game2048Env
from rl2048.eval.metrics import distribution_summary, tile_reach_probs, wilson_interval
from rl2048.eval.runner import run_episode
from rl2048.eval.seeds import val_seeds
from rl2048.policies.expectimax import ExpectimaxDQNPolicy

CKPT = Path("checkpoints/phaseA_dueling_seed0.pt")
OUT = Path("/root/autodl-tmp/RL-2048/results/experiments/playout_d2_vs_d3_n20.json")
N = 20
MAX_STEPS = 8000
CONFIGS = ((2, "d2_corner"), (3, "d3_corner"))


def summarize(name: str, rows: list[dict], elapsed: float) -> dict:
    n = len(rows)
    n2048 = sum(r["max_tile"] >= 2048 for r in rows)
    n4096 = sum(r["max_tile"] >= 4096 for r in rows)
    n1024 = sum(r["max_tile"] >= 1024 for r in rows)
    ci2048 = wilson_interval(n2048, n)
    ci4096 = wilson_interval(n4096, n)
    scores = [r["score"] for r in rows]
    return {
        "name": name,
        "n": n,
        "mean_score": float(np.mean(scores)) if scores else 0.0,
        "score_stats": distribution_summary(scores) if scores else {},
        "p_reach_1024": n1024 / n if n else 0.0,
        "p_reach_2048": n2048 / n if n else 0.0,
        "p_reach_2048_ci": [ci2048.lower, ci2048.upper],
        "p_reach_4096": n4096 / n if n else 0.0,
        "p_reach_4096_ci": [ci4096.lower, ci4096.upper],
        "tile_reach_probs": tile_reach_probs([r["max_tile"] for r in rows]),
        "truncation_rate": float(np.mean([r["truncated"] for r in rows])) if rows else 0.0,
        "mean_length": float(np.mean([r["length"] for r in rows])) if rows else 0.0,
        "fail_max_tiles": sorted(r["max_tile"] for r in rows if r["max_tile"] < 2048),
        "elapsed_sec": elapsed,
        "episodes": rows,
    }


def load_payload() -> dict:
    if OUT.exists():
        try:
            payload = json.loads(OUT.read_text(encoding="utf-8"))
            payload.setdefault("results", {})
            return payload
        except json.JSONDecodeError:
            pass
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n": N,
        "max_steps": MAX_STEPS,
        "stop_on_2048": False,
        "checkpoint": str(CKPT),
        "results": {},
    }


def save_payload(payload: dict) -> None:
    payload["timestamp"] = datetime.now(timezone.utc).isoformat()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(OUT)


def main() -> None:
    seeds = val_seeds(N)
    payload = load_payload()
    print(f"checkpoint={CKPT} n={N} max_steps={MAX_STEPS} out={OUT}", flush=True)

    for depth, name in CONFIGS:
        existing = list(payload.get("results", {}).get(name, {}).get("episodes", []))
        done = {int(r["seed"]) for r in existing}
        remaining = [s for s in seeds if int(s) not in done]
        print(f"=== {name} resume {len(existing)}/{N}, remaining={len(remaining)} ===", flush=True)
        if not remaining:
            payload["results"][name] = summarize(
                name, existing, float(payload["results"].get(name, {}).get("elapsed_sec", 0.0))
            )
            save_payload(payload)
            continue

        policy = ExpectimaxDQNPolicy.from_checkpoint(
            CKPT, depth=depth, adaptive=False, corner_tiebreak=True
        )
        env = Game2048Env(max_episode_steps=MAX_STEPS)
        rows = existing
        t0 = time.perf_counter()
        prior = float(payload.get("results", {}).get(name, {}).get("elapsed_sec", 0.0) or 0.0)
        for i, seed in enumerate(remaining, start=1):
            ep = run_episode(env, policy, seed=seed, policy_key=name, stop_on_2048=False)
            rows.append(
                {
                    "seed": int(seed),
                    "score": int(ep.game_score),
                    "max_tile": int(ep.max_tile),
                    "reached_2048": bool(ep.reached_2048),
                    "length": int(ep.episode_length),
                    "truncated": bool(ep.truncated),
                    "terminated": bool(ep.terminated),
                }
            )
            n = len(rows)
            n2048 = sum(r["max_tile"] >= 2048 for r in rows)
            n4096 = sum(r["max_tile"] >= 4096 for r in rows)
            mean = float(np.mean([r["score"] for r in rows]))
            elapsed = prior + (time.perf_counter() - t0)
            print(
                f"  {name} {n}/{N} seed={seed} score={ep.game_score} max={ep.max_tile} "
                f"len={ep.episode_length} mean={mean:.0f} "
                f"P2048={100 * n2048 / n:.1f}% P4096={100 * n4096 / n:.1f}% "
                f"({elapsed:.0f}s)",
                flush=True,
            )
            payload["results"][name] = summarize(name, rows, elapsed)
            save_payload(payload)
        print(
            f"DONE {name} mean={payload['results'][name]['mean_score']:.0f} "
            f"P2048={100 * payload['results'][name]['p_reach_2048']:.1f}% "
            f"P4096={100 * payload['results'][name]['p_reach_4096']:.1f}%",
            flush=True,
        )

    print("Saved", OUT, flush=True)


if __name__ == "__main__":
    main()
