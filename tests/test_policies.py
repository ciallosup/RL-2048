"""Tests for baseline policies."""

import numpy as np

from rl2048.core import ACTION_LEFT
from rl2048.env import Game2048Env
from rl2048.policies.base import PolicyContext
from rl2048.policies.greedy import GreedyMergePolicy
from rl2048.policies.random_policy import RandomPolicy


def test_random_policy_picks_legal_action():
    env = Game2048Env()
    obs, info = env.reset(seed=0)
    ctx = PolicyContext(env=env, obs=obs, info=info)
    policy = RandomPolicy()
    policy.reset(ctx)
    action = policy.select_action(ctx)
    assert info["valid_action_mask"][action]


def test_greedy_picks_highest_merge():
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
    obs = env._game.observation()
    info = env._build_info(merge_score=0)
    ctx = PolicyContext(env=env, obs=obs, info=info)
    policy = GreedyMergePolicy()
    assert policy.select_action(ctx) == ACTION_LEFT
