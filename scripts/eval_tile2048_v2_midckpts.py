#!/usr/bin/env python3
"""Val Phase B v2 mid-checkpoints on val/1000."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Allow `python scripts/...` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from scripts.run_experiment import evaluate_checkpoint, _tile_p

ROOT = Path("/root/autodl-tmp/RL-2048/results/runs/opt_tile2048_v2")
STEPS = [4_000_000, 5_000_000, 6_000_000, 8_000_000, 10_000_000]
OUT = Path("/root/autodl-tmp/RL-2048/results/experiments/opt_tile2048_v2_midckpt_val.json")


def main() -> None:
    runs = sorted(p for p in ROOT.iterdir() if p.is_dir())
    results: list[dict] = []

    for run in runs:
        seed = int(run.name.split("seed")[1].split("_")[0])
        for steps in STEPS:
            ckpt = run / f"checkpoint_{steps}.pt"
            if not ckpt.exists() and steps == 10_000_000:
                ckpt = run / "checkpoint_final.pt"
            if not ckpt.exists():
                print(f"MISSING {ckpt}", flush=True)
                continue
            print(f"=== eval seed={seed} steps={steps} {ckpt.name} ===", flush=True)
            summary = evaluate_checkpoint(
                ckpt,
                episodes=1000,
                seed_set="val",
                max_episode_steps=1200,
            )
            row = {
                "train_seed": seed,
                "env_steps": steps,
                "checkpoint": str(ckpt),
                "mean_score": summary["score_stats"]["mean"],
                "p_ge_512": _tile_p(summary, "P(>=512)"),
                "p_reach_1024": summary["p_reach_1024"],
                "p_reach_2048": summary["p_reach_2048"],
                "max_tile_mean": summary["max_tile_stats"]["mean"],
            }
            results.append(row)
            print(
                f"  mean={row['mean_score']:.1f} P512={100 * row['p_ge_512']:.1f}% "
                f"P1024={100 * row['p_reach_1024']:.2f}% P2048={100 * row['p_reach_2048']:.2f}%",
                flush=True,
            )
            # incremental save
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(
                json.dumps(
                    {"timestamp": datetime.now(timezone.utc).isoformat(), "runs": results},
                    indent=2,
                ),
                encoding="utf-8",
            )

    by_step: dict[int, list] = defaultdict(list)
    for r in results:
        by_step[r["env_steps"]].append(r)

    agg = {}
    print("\n=== Aggregate by checkpoint step ===", flush=True)
    print(f"{'steps':>10} | {'mean':>10} | {'P512':>8} | {'P1024':>8} | {'P2048':>8}", flush=True)
    print("-" * 56, flush=True)
    for steps in STEPS:
        rows = by_step.get(steps, [])
        if not rows:
            continue
        means = [r["mean_score"] for r in rows]
        p512 = [r["p_ge_512"] for r in rows]
        p1024 = [r["p_reach_1024"] for r in rows]
        p2048 = [r["p_reach_2048"] for r in rows]
        agg[steps] = {
            "n": len(rows),
            "mean_score": {"mean": float(np.mean(means)), "std": float(np.std(means))},
            "p_ge_512": {"mean": float(np.mean(p512)), "std": float(np.std(p512))},
            "p_reach_1024": {"mean": float(np.mean(p1024)), "std": float(np.std(p1024))},
            "p_reach_2048": {"mean": float(np.mean(p2048)), "std": float(np.std(p2048))},
        }
        a = agg[steps]
        print(
            f"{steps:10d} | {a['mean_score']['mean']:7.1f}±{a['mean_score']['std']:<4.0f} | "
            f"{100 * a['p_ge_512']['mean']:6.1f}% | {100 * a['p_reach_1024']['mean']:6.2f}% | "
            f"{100 * a['p_reach_2048']['mean']:6.2f}%",
            flush=True,
        )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoints_evaluated": STEPS,
        "runs": results,
        "aggregate_by_steps": {str(k): v for k, v in agg.items()},
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
