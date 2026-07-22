"""Estimate natural episode length distribution for max_episode_steps calibration."""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from rl2048.core import ACTION_NAMES, NUM_ACTIONS, valid_action_mask
from rl2048.env import Game2048Env


def random_legal_policy(env: Game2048Env) -> int:
    mask = valid_action_mask(env._game.board)
    legal = np.flatnonzero(mask)
    return int(env.np_random.choice(legal))


def fixed_priority_policy(env: Game2048Env, order: tuple[int, ...]) -> int:
    mask = valid_action_mask(env._game.board)
    for action in order:
        if mask[action]:
            return action
    raise RuntimeError("No legal action")


def run_episodes(policy_fn, episodes: int, seed: int) -> list[int]:
    env = Game2048Env(max_episode_steps=None)
    lengths: list[int] = []
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        terminated = truncated = False
        while not (terminated or truncated):
            action = policy_fn(env)
            _, _, terminated, truncated, _ = env.step(action)
        lengths.append(env._game.step_count)
    return lengths


def summarize(lengths: list[int]) -> dict[str, float]:
    arr = np.asarray(lengths, dtype=np.float64)
    return {
        "count": len(lengths),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "max": float(arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random_lengths = run_episodes(random_legal_policy, args.episodes, args.seed)
    fixed_order = (2, 1, 3, 0)  # left, down, right, up
    fixed_lengths = run_episodes(
        lambda env: fixed_priority_policy(env, fixed_order),
        args.episodes,
        args.seed + 10_000,
    )

    print("Random legal policy:")
    for key, value in summarize(random_lengths).items():
        print(f"  {key}: {value:.1f}")

    print("\nFixed priority (left->down->right->up):")
    for key, value in summarize(fixed_lengths).items():
        print(f"  {key}: {value:.1f}")

    p99 = max(summarize(random_lengths)["p99"], summarize(fixed_lengths)["p99"])
    print(f"\nSuggested starting max_episode_steps: >= {int(np.ceil(p99 * 1.1))}")


if __name__ == "__main__":
    main()
