"""Baseline evaluation CLI (roadmap section 4 & 6)."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from rl2048.eval.report import (
    check_e0_gate,
    print_baseline_table,
    print_tile_curves,
    save_report,
)
from rl2048.eval.runner import evaluate_policy, summary_to_dict
from rl2048.eval.seeds import dev_seeds, save_seeds, val_seeds
from rl2048.policies.registry import BASELINE_POLICIES, get_policy_class


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate non-neural 2048 baselines.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="Number of evaluation episodes per policy (default: 200 dev set).",
    )
    parser.add_argument(
        "--seed-set",
        choices=("dev", "val"),
        default="dev",
        help="Fixed seed pool (dev=200 base, val=1000 base).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Episode truncation limit (roadmap calibrated ~436+).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/baselines/latest.json"),
        help="JSON report output path.",
    )
    parser.add_argument(
        "--policies",
        nargs="*",
        default=None,
        help="Policy keys to evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Evaluate a DQN checkpoint (adds to policy list).",
    )
    parser.add_argument(
        "--decode",
        choices=("greedy", "1ply", "2ply", "3ply"),
        default="2ply",
        help="RL inference: greedy Q, 1-ply, 2-ply, or adaptive 3-ply (default: 2ply).",
    )
    args = parser.parse_args()

    max_steps = args.max_steps
    if args.checkpoint is not None and args.decode != "greedy" and args.max_steps == 500:
        max_steps = 4000

    if args.seed_set == "dev":
        seeds = dev_seeds(args.episodes)
    else:
        seeds = val_seeds(args.episodes)

    save_seeds(Path("data/seeds") / f"{args.seed_set}_{args.episodes}.json", seeds)

    policy_keys = list(args.policies) if args.policies is not None else list(BASELINE_POLICIES)
    summaries = []
    for key in policy_keys:
        cls = get_policy_class(key)
        summary = evaluate_policy(
            cls(),
            policy_key=cls.key,
            policy_label=cls.label,
            seeds=seeds,
            max_episode_steps=max_steps,
        )
        summaries.append(summary)

    if args.checkpoint:
        from rl2048.policies.dqn_policy import DECODE_LABELS, DQNPolicy

        dqn = DQNPolicy.from_checkpoint(args.checkpoint, decode=args.decode)
        summaries.append(
            evaluate_policy(
                dqn,
                policy_key=f"dqn_{args.decode}",
                policy_label=f"{DECODE_LABELS[args.decode]} ({args.checkpoint.name})",
                seeds=seeds,
                max_episode_steps=max_steps,
            )
        )

    print(f"Evaluated {len(seeds)} episodes per policy ({args.seed_set} seeds)\n")
    print_baseline_table(summaries)
    print_tile_curves(summaries)

    random_summary = next((s for s in summaries if s.policy_key == "random"), None)
    heuristic_summary = next((s for s in summaries if s.policy_key == "heuristic"), None)
    gate = None
    if random_summary is not None and heuristic_summary is not None:
        gate = check_e0_gate(random_summary, heuristic_summary)
        print(f"\nE0 验收: {gate['message']}")
        print(
            f"  random: score={gate['random_mean_score']:.0f}, max_tile={gate['random_mean_max_tile']:.0f}"
        )
        print(
            f"  heuristic: score={gate['heuristic_mean_score']:.0f}, max_tile={gate['heuristic_mean_max_tile']:.0f}"
        )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "episodes": len(seeds),
        "seed_set": args.seed_set,
        "max_episode_steps": max_steps,
        "decode": args.decode if args.checkpoint else None,
        "seeds_file": str(Path("data/seeds") / f"{args.seed_set}_{args.episodes}.json"),
        "policies": [summary_to_dict(s) for s in summaries],
        "e0_gate": gate,
    }
    save_report(args.output, payload)
    print(f"\nReport saved: {args.output}")


if __name__ == "__main__":
    main()
