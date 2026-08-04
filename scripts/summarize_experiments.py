"""Aggregate E1/E2 multi-seed experiment JSONs into a short comparison table."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _seed_metrics(payload: dict) -> list[dict]:
    rows = []
    for run in payload.get("runs", []):
        ev = run.get("eval", {})
        rows.append(
            {
                "train_seed": run.get("train_seed"),
                "p_reach_2048": float(ev.get("p_reach_2048", 0.0)),
                "p_reach_1024": float(ev.get("p_reach_1024", 0.0)),
                "mean_score": float(ev.get("score_stats", {}).get("mean", 0.0)),
                "mean_max_tile": float(ev.get("max_tile_stats", {}).get("mean", 0.0)),
                "mean_length": float(ev.get("length_stats", {}).get("mean", 0.0)),
                "truncation_rate": float(ev.get("truncation_rate", 0.0)),
            }
        )
    return rows


def _agg(rows: list[dict], key: str) -> dict:
    vals = [r[key] for r in rows]
    if not vals:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(statistics.fmean(vals)),
        "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
        "min": float(min(vals)),
        "max": float(max(vals)),
    }


def summarize(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = _seed_metrics(payload)
    cfg = payload.get("config", {})
    return {
        "path": str(path),
        "run_name": cfg.get("run_name"),
        "use_double_dqn": cfg.get("use_double_dqn"),
        "train_seeds": len(rows),
        "total_env_steps": cfg.get("total_env_steps"),
        "eval_seed_set": cfg.get("eval_seed_set"),
        "num_eval_episodes": cfg.get("num_eval_episodes"),
        "per_seed": rows,
        "aggregate": {
            "p_reach_2048": _agg(rows, "p_reach_2048"),
            "p_reach_1024": _agg(rows, "p_reach_1024"),
            "mean_score": _agg(rows, "mean_score"),
            "mean_max_tile": _agg(rows, "mean_max_tile"),
            "mean_length": _agg(rows, "mean_length"),
            "truncation_rate": _agg(rows, "truncation_rate"),
        },
        "random_baseline": payload.get("random_baseline"),
    }


def _print_block(title: str, summary: dict) -> None:
    agg = summary["aggregate"]
    print(f"\n{title}")
    print(
        f"  double={summary['use_double_dqn']}  seeds={summary['train_seeds']}  "
        f"steps={summary['total_env_steps']}  eval={summary['num_eval_episodes']}/{summary['eval_seed_set']}"
    )
    print(
        f"  P(2048)={agg['p_reach_2048']['mean']:.1%}±{agg['p_reach_2048']['std']:.1%}  "
        f"P(1024)={agg['p_reach_1024']['mean']:.1%}±{agg['p_reach_1024']['std']:.1%}  "
        f"score={agg['mean_score']['mean']:.0f}±{agg['mean_score']['std']:.0f}  "
        f"max_tile={agg['mean_max_tile']['mean']:.0f}±{agg['mean_max_tile']['std']:.0f}"
    )
    for row in summary["per_seed"]:
        print(
            f"    seed={row['train_seed']}: "
            f"P(2048)={row['p_reach_2048']:.1%}  P(1024)={row['p_reach_1024']:.1%}  "
            f"score={row['mean_score']:.0f}  max_tile={row['mean_max_tile']:.0f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize E1/E2 experiment JSONs.")
    parser.add_argument("--e1", type=Path, required=True)
    parser.add_argument("--e2", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/experiments/e1_e2_compare.json"))
    args = parser.parse_args()

    e1 = summarize(args.e1)
    e2 = summarize(args.e2)
    payload = {
        "e1_double_dqn": e1,
        "e2_vanilla_dqn": e2,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_block("E1 Double DQN", e1)
    _print_block("E2 Vanilla DQN", e2)
    print(f"\nComparison saved to {args.output}")


if __name__ == "__main__":
    main()
