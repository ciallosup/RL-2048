"""Multi-seed training + val evaluation for DQN experiments."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from rl2048.eval.runner import evaluate_policy, summary_to_dict
from rl2048.eval.seeds import dev_seeds, save_seeds, val_seeds
from rl2048.policies.dqn_policy import DQNPolicy
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
    policy = DQNPolicy.from_checkpoint(checkpoint_path)
    summary = evaluate_policy(
        policy,
        policy_key="dqn",
        policy_label=f"DQN ({checkpoint_path.name})",
        seeds=seeds,
        max_episode_steps=max_episode_steps,
    )
    return summary_to_dict(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-seed DQN experiment.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-seeds", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/experiments/latest.json"))
    parser.add_argument("--skip-train", action="store_true")
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
        result = Trainer(config).train()
        eval_summary = evaluate_checkpoint(
            result.checkpoint_path,
            episodes=config.num_eval_episodes,
            seed_set=config.eval_seed_set,
            max_episode_steps=config.max_episode_steps,
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

    random_seeds = (
        val_seeds(base_config.num_eval_episodes)
        if base_config.eval_seed_set == "val"
        else dev_seeds(base_config.num_eval_episodes)
    )
    save_seeds(
        Path("data/seeds") / f"{base_config.eval_seed_set}_{base_config.num_eval_episodes}.json",
        random_seeds,
    )
    random_summary = evaluate_policy(
        RandomPolicy(),
        policy_key="random",
        policy_label="random",
        seeds=random_seeds,
        max_episode_steps=base_config.max_episode_steps,
    )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": base_config.to_dict(),
        "train_seeds": args.train_seeds,
        "runs": runs,
        "random_baseline": summary_to_dict(random_summary),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Experiment summary saved to {args.output}")


if __name__ == "__main__":
    main()
