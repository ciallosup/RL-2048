"""Tests for 1-ply expectimax successor expansion and action selection."""

from __future__ import annotations

import numpy as np
import pytest

from rl2048.core import ACTION_DOWN, ACTION_LEFT, TILE_TWO_PROB, move_board
from rl2048.env import Game2048Env
from rl2048.policies.base import PolicyContext
from rl2048.rl.agent import DQNAgent
from rl2048.rl.config import resolve_device
from rl2048.rl.search import expand_action_successors, expectimax_select_action


def test_expand_successors_probs_sum_to_one():
    board = np.array(
        [
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    expanded = expand_action_successors(board, ACTION_LEFT)
    assert expanded is not None
    successors, merge_score = expanded
    assert merge_score == 4
    after, _, changed = move_board(board, ACTION_LEFT)
    assert changed
    n_empty = int(np.count_nonzero(after == 0))
    assert len(successors) == n_empty * 2
    probs = np.array([s.prob for s in successors])
    np.testing.assert_allclose(probs.sum(), 1.0, atol=1e-6)
    np.testing.assert_allclose(probs.max(), (1.0 / n_empty) * TILE_TWO_PROB, atol=1e-6)


def test_illegal_action_returns_none():
    board = np.array(
        [
            [2, 4, 8, 16],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    assert expand_action_successors(board, ACTION_LEFT) is None
    assert expand_action_successors(board, ACTION_DOWN) is not None


def test_expectimax_picks_legal_and_prefers_high_leaf():
    board = np.array(
        [
            [2, 2, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )

    def q_batch_fn(obs_batch: np.ndarray) -> np.ndarray:
        # Prefer states whose top-left tile is 4 (the left-merge outcome).
        q = np.zeros((obs_batch.shape[0], 4), dtype=np.float32)
        topleft = obs_batch[:, 0]
        q[:, :] = topleft[:, None]
        return q

    action = expectimax_select_action(
        board,
        q_batch_fn,
        gamma=0.995,
        reward_mode="log1p",
        include_reward=True,
    )
    assert action == ACTION_LEFT


def test_expectimax_policy_legal_action_random_net():
    torch = pytest.importorskip("torch")
    device = resolve_device("cpu")
    agent = DQNAgent(
        device=device,
        obs_encoding="onehot",
        network_type="dueling_cnn",
        hidden_dims=(32,),
        conv_channels=(8, 8),
        onehot_channels=16,
        gamma=0.995,
    )
    agent.eval_mode()
    env = Game2048Env(max_episode_steps=20)
    obs, info = env.reset(seed=1)
    ctx = PolicyContext(env=env, obs=obs, info=info)

    def q_batch_fn(obs_batch):
        return agent.q_values_batch(obs_batch, obs_scale=16.0)

    action = expectimax_select_action(
        env.board,
        q_batch_fn,
        gamma=0.995,
        reward_mode="log1p",
    )
    assert info["valid_action_mask"][action]
    assert ctx is not None
    assert torch is not None
