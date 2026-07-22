"""Tests for heuristic features and policy."""

import numpy as np

from rl2048.core import ACTION_LEFT, ACTION_UP
from rl2048.env import Game2048Env
from rl2048.heuristic.features import board_heuristic_score, count_empty
from rl2048.policies.base import PolicyContext
from rl2048.policies.heuristic import HeuristicPolicy
from rl2048.policies.random_policy import RandomPolicy


def test_empty_board_scores_high():
    board = np.zeros((4, 4), dtype=np.int32)
    assert count_empty(board) == 16
    assert board_heuristic_score(board) > 0


def test_heuristic_prefers_merge_when_obvious():
    env = Game2048Env()
    env.reset(seed=0)
    env._game = env._game.from_board(
        [
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        rng=env.np_random,
    )
    ctx = PolicyContext(
        env=env,
        obs=env._game.observation(),
        info=env._build_info(merge_score=0),
    )
    policy = HeuristicPolicy()
    assert policy.select_action(ctx) == ACTION_LEFT


def test_heuristic_beats_random_on_same_seeds():
    """Heuristic mean score should exceed random on shared seeds."""
    from rl2048.eval.runner import evaluate_policy
    from rl2048.eval.seeds import generate_seeds

    seeds = generate_seeds(999, 30)
    h = evaluate_policy(
        HeuristicPolicy(),
        policy_key="heuristic",
        policy_label="heuristic",
        seeds=seeds,
        max_episode_steps=200,
    )
    r = evaluate_policy(
        RandomPolicy(),
        policy_key="random",
        policy_label="random",
        seeds=seeds,
        max_episode_steps=200,
    )
    assert h.score_stats["mean"] > r.score_stats["mean"]
    assert h.max_tile_stats["mean"] >= r.max_tile_stats["mean"]
