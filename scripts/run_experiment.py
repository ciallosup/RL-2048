"""Multi-seed training + val evaluation for DQN experiments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rl2048.eval.runner import evaluate_policy, summary_to_dict
from rl2048.eval.seeds import dev_seeds, save_seeds, val_seeds
from rl2048.policies.dqn_policy import DQNPolicy
from rl2048.policies.greedy import GreedyMergePolicy
from rl2048.policies.heuristic import HeuristicPolicy
from rl2048.policies.random_policy import RandomPolicy
from rl2048.rl.config import TrainConfig, load_config
from rl2048.rl.trainer import Trainer


def evaluate_checkpoint(
    checkpoint_path: Path,
    *,
    episodes: int,
    seed_set: str,
    max_episode_steps: int,
) -> dict:
    if seed_set == "dev":
        seeds = dev_seeds(episodes)
    else:
        seeds = val_seeds(episodes)
    policy = DQNPolicy.from_checkpoint(checkpoint_path, decode="greedy")
    summary = evaluate_policy(
        policy,
        policy_key="dqn",
        policy_label=f"DQN ({checkpoint_path.name})",
        seeds=seeds,
        max_episode_steps=max_episode_steps,
    )
    return summary_to_dict(summary)


def _eval_seeds(config: TrainConfig):
    if config.eval_seed_set == "val":
        return val_seeds(config.num_eval_episodes)
    return dev_seeds(config.num_eval_episodes)


def _tile_p(summary: dict, key: str) -> float:
    return float(summary.get("tile_reach_probs", {}).get(key, 0.0))


def _aggregate_dqn_runs(runs: list[dict]) -> dict:
    if not runs:
        return {}
    mean_scores = [r["eval"]["score_stats"]["mean"] for r in runs]
    mean_max_tiles = [r["eval"]["max_tile_stats"]["mean"] for r in runs]
    p1024 = [r["eval"]["p_reach_1024"] for r in runs]
    p2048 = [r["eval"]["p_reach_2048"] for r in runs]
    p256 = [_tile_p(r["eval"], "P(>=256)") for r in runs]
    p512 = [_tile_p(r["eval"], "P(>=512)") for r in runs]
    return {
        "n_seeds": len(runs),
        "mean_score": {"mean": float(np.mean(mean_scores)), "std": float(np.std(mean_scores))},
        "mean_max_tile": {"mean": float(np.mean(mean_max_tiles)), "std": float(np.std(mean_max_tiles))},
        "p_reach_1024": {"mean": float(np.mean(p1024)), "std": float(np.std(p1024))},
        "p_reach_2048": {"mean": float(np.mean(p2048)), "std": float(np.std(p2048))},
        "p_ge_256": {"mean": float(np.mean(p256)), "std": float(np.std(p256))},
        "p_ge_512": {"mean": float(np.mean(p512)), "std": float(np.std(p512))},
    }


def _print_comparison(aggregate: dict, baselines: dict[str, dict]) -> None:
    print("\n=== Tile-horizon comparison (same eval seed set) ===")
    print(
        f"{'policy':<14} | {'mean_score':>10} | {'P(>=256)':>8} | {'P(>=512)':>8} | "
        f"{'P(1024)':>8} | {'P(2048)':>8}"
    )
    print("-" * 72)
    for name, summary in baselines.items():
        print(
            f"{name:<14} | {summary['score_stats']['mean']:10.1f} | "
            f"{100 * _tile_p(summary, 'P(>=256)'):7.1f}% | "
            f"{100 * _tile_p(summary, 'P(>=512)'):7.1f}% | "
            f"{100 * summary['p_reach_1024']:7.1f}% | "
            f"{100 * summary['p_reach_2048']:7.1f}%"
        )
    if aggregate:
        print(
            f"{'dqn (seeds)':<14} | "
            f"{aggregate['mean_score']['mean']:7.1f}±{aggregate['mean_score']['std']:.1f} | "
            f"{100 * aggregate['p_ge_256']['mean']:7.1f}% | "
            f"{100 * aggregate['p_ge_512']['mean']:7.1f}% | "
            f"{100 * aggregate['p_reach_1024']['mean']:7.1f}% | "
            f"{100 * aggregate['p_reach_2048']['mean']:7.1f}%"
        )
        p1024 = aggregate["p_reach_1024"]["mean"]
        p2048 = aggregate["p_reach_2048"]["mean"]
        mean_score = aggregate["mean_score"]["mean"]
        print(f"\nPhaseA gate P(1024)>=5%? {'YES' if p1024 >= 0.05 else 'NO'} ({100 * p1024:.2f}%)")
        print(f"PhaseB gate P(2048)>=1%? {'YES' if p2048 >= 0.01 else 'NO'} ({100 * p2048:.2f}%)")
        print(f"Mean score floor >=4500? {'YES' if mean_score >= 4500 else 'NO'} ({mean_score:.1f})")
        heur = baselines.get("heuristic")
        if heur is not None:
            beats = mean_score > heur["score_stats"]["mean"]
            print(
                f"Beat heuristic on mean score? {'YES' if beats else 'NO'} "
                f"(dqn {mean_score:.1f} vs heuristic {heur['score_stats']['mean']:.1f})"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed DQN experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/experiments/latest.json"))
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip random/heuristic/greedy baseline evals (useful for ablations).",
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Evaluate single checkpoint only.")
    args = parser.parse_args()

    base_config = load_config(args.config)

    if args.checkpoint:
        summary = evaluate_checkpoint(
            args.checkpoint,
            episodes=base_config.num_eval_episodes,
            seed_set=base_config.eval_seed_set,
            max_episode_steps=base_config.max_episode_steps,
        )
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checkpoint": str(args.checkpoint),
            "eval": summary,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    runs = []
    for offset in range(args.train_seeds):
        config = TrainConfig.from_dict(base_config.to_dict())
        config.train_seed = args.seed_start + offset
        if args.skip_train:
            continue
        print(
            f"\n=== Train seed {config.train_seed} "
            f"({offset + 1}/{args.train_seeds}, steps={config.total_env_steps}) ===",
            flush=True,
        )
        result = Trainer(config).train()
        print(f"Training finished: {result.checkpoint_path}", flush=True)
        eval_summary = evaluate_checkpoint(
            result.checkpoint_path,
            episodes=config.num_eval_episodes,
            seed_set=config.eval_seed_set,
            max_episode_steps=config.max_episode_steps,
        )
        print(
            f"Seed {config.train_seed} eval: mean_score={eval_summary['score_stats']['mean']:.1f}, "
            f"P(1024)={100 * eval_summary['p_reach_1024']:.2f}%, "
            f"P(2048)={100 * eval_summary['p_reach_2048']:.2f}%, "
            f"P(>=512)={100 * _tile_p(eval_summary, 'P(>=512)'):.1f}%",
            flush=True,
        )
        runs.append(
            {
                "train_seed": config.train_seed,
                "run_dir": str(result.run_dir),
                "checkpoint": str(result.checkpoint_path),
                "env_steps": result.env_steps,
                "eval": eval_summary,
            }
        )

    eval_seed_list = _eval_seeds(base_config)
    save_seeds(
        Path("data/seeds") / f"{base_config.eval_seed_set}_{base_config.num_eval_episodes}.json",
        eval_seed_list,
    )

    baselines: dict[str, dict] = {}
    if not args.skip_baselines:
        for key, cls in (
            ("random", RandomPolicy),
            ("heuristic", HeuristicPolicy),
            ("greedy", GreedyMergePolicy),
        ):
            print(f"\n=== Baseline eval: {key} ({base_config.num_eval_episodes} eps) ===", flush=True)
            summary = evaluate_policy(
                cls(),
                policy_key=key,
                policy_label=key,
                seeds=eval_seed_list,
                max_episode_steps=base_config.max_episode_steps,
            )
            baselines[key] = summary_to_dict(summary)

    aggregate = _aggregate_dqn_runs(runs)
    _print_comparison(aggregate, baselines)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": base_config.to_dict(),
        "train_seeds": args.train_seeds,
        "seed_start": args.seed_start,
        "runs": runs,
        "dqn_aggregate": aggregate,
        "baselines": baselines,
        # keep old key for compatibility
        "random_baseline": baselines.get("random"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nExperiment summary saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
