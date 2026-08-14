from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rl2048.env import Game2048Env
from rl2048.eval.metrics import (
    distribution_summary,
    tile_reach_intervals,
    tile_reach_probs,
    wilson_interval,
)
from rl2048.policies.base import Policy, PolicyContext


@dataclass
class EpisodeResult:
    seed: int
    policy_key: str
    game_score: int
    max_tile: int
    reached_2048: bool
    episode_length: int
    terminated: bool
    truncated: bool


@dataclass
class PolicyEvalSummary:
    policy_key: str
    policy_label: str
    episodes: int
    p_reach_2048: float
    p_reach_2048_ci: tuple[float, float]
    p_reach_1024: float
    tile_reach_probs: dict[str, float]
    score_stats: dict[str, float]
    max_tile_stats: dict[str, float]
    length_stats: dict[str, float]
    truncation_rate: float
    raw_episodes: list[EpisodeResult]


def run_episode(
    env: Game2048Env,
    policy: Policy,
    *,
    seed: int,
    policy_key: str,
    stop_on_2048: bool = False,
) -> EpisodeResult:
    obs, info = env.reset(seed=seed)
    ctx = PolicyContext(env=env, obs=obs, info=info)
    policy.reset(ctx)

    terminated = truncated = False
    while not (terminated or truncated):
        action = policy.select_action(ctx)
        if action is None:
            break
        obs, _, terminated, truncated, info = env.step(action)
        ctx.obs = obs
        ctx.info = info
        if stop_on_2048 and info.get("reached_2048"):
            break

    return EpisodeResult(
        seed=seed,
        policy_key=policy_key,
        game_score=int(info["game_score"]),
        max_tile=int(info["max_tile"]),
        reached_2048=bool(info["reached_2048"]),
        episode_length=int(info["episode_length"]),
        terminated=terminated,
        truncated=truncated,
    )


def evaluate_policy(
    policy: Policy,
    *,
    policy_key: str,
    policy_label: str,
    seeds: list[int],
    max_episode_steps: int | None = 500,
    stop_on_2048: bool = False,
) -> PolicyEvalSummary:
    env = Game2048Env(max_episode_steps=max_episode_steps)
    episodes: list[EpisodeResult] = []
    for seed in seeds:
        episodes.append(
            run_episode(
                env,
                policy,
                seed=seed,
                policy_key=policy_key,
                stop_on_2048=stop_on_2048,
            )
        )

    scores = [e.game_score for e in episodes]
    max_tiles = [e.max_tile for e in episodes]
    lengths = [e.episode_length for e in episodes]
    successes_2048 = sum(e.reached_2048 for e in episodes)
    successes_1024 = sum(e.max_tile >= 1024 for e in episodes)
    ci_2048 = wilson_interval(successes_2048, len(episodes))

    return PolicyEvalSummary(
        policy_key=policy_key,
        policy_label=policy_label,
        episodes=len(episodes),
        p_reach_2048=ci_2048.rate,
        p_reach_2048_ci=(ci_2048.lower, ci_2048.upper),
        p_reach_1024=successes_1024 / len(episodes) if episodes else 0.0,
        tile_reach_probs=tile_reach_probs(max_tiles),
        score_stats=distribution_summary(scores),
        max_tile_stats=distribution_summary(max_tiles),
        length_stats=distribution_summary(lengths),
        truncation_rate=float(np.mean([e.truncated for e in episodes])) if episodes else 0.0,
        raw_episodes=episodes,
    )


def summary_to_dict(summary: PolicyEvalSummary, *, include_raw: bool = False) -> dict[str, Any]:
    data: dict[str, Any] = {
        "policy_key": summary.policy_key,
        "policy_label": summary.policy_label,
        "episodes": summary.episodes,
        "p_reach_2048": summary.p_reach_2048,
        "p_reach_2048_ci": list(summary.p_reach_2048_ci),
        "p_reach_1024": summary.p_reach_1024,
        "tile_reach_probs": summary.tile_reach_probs,
        "score_stats": summary.score_stats,
        "max_tile_stats": summary.max_tile_stats,
        "length_stats": summary.length_stats,
        "truncation_rate": summary.truncation_rate,
        "tile_reach_intervals": {
            k: asdict(v) for k, v in tile_reach_intervals(e.max_tile for e in summary.raw_episodes).items()
        },
    }
    if include_raw:
        data["episodes_raw"] = [asdict(e) for e in summary.raw_episodes]
    return data
