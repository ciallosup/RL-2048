"""Tests for expectimax successor expansion, LUT moves, and search depth."""

from __future__ import annotations

import numpy as np
import pytest

from rl2048.core import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    TILE_TWO_PROB,
    exponents_to_observation,
    move_board,
    valid_action_mask,
)
from rl2048.env import Game2048Env
from rl2048.policies.base import PolicyContext
from rl2048.rl.agent import DQNAgent
from rl2048.rl.config import resolve_device
from rl2048.rl.fast_move import boards_to_obs, move_board_fast, valid_action_mask_fast, valid_action_masks_batch
from rl2048.rl.search import (
    expand_action_successors,
    expectimax_action_values,
    expectimax_select_action,
    resolve_search_depth,
)


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


def test_fast_move_matches_core_random_boards():
    rng = np.random.default_rng(0)
    for _ in range(80):
        exponents = rng.integers(0, 12, size=(4, 4), dtype=np.int32)
        board = np.where(exponents == 0, 0, np.left_shift(1, exponents)).astype(np.int32)
        for action in range(4):
            after_a, score_a, ch_a = move_board(board, action)
            after_b, score_b, ch_b = move_board_fast(board, action)
            assert ch_a == ch_b
            assert score_a == score_b
            np.testing.assert_array_equal(after_a, after_b)
        np.testing.assert_array_equal(valid_action_mask(board), valid_action_mask_fast(board))
        np.testing.assert_array_equal(valid_action_mask(board), valid_action_masks_batch(board[None, ...])[0])


def test_boards_to_obs_matches_core():
    board = np.array(
        [[0, 2, 4, 8], [16, 32, 64, 128], [256, 512, 1024, 2048], [0, 0, 2, 4]],
        dtype=np.int32,
    )
    np.testing.assert_allclose(boards_to_obs([board])[0], exponents_to_observation(board).astype(np.float32))


def test_adaptive_depth_only_in_endgame():
    early = np.zeros((4, 4), dtype=np.int32)
    early[0, 0] = 8
    assert resolve_search_depth(early, depth=2, adaptive=True) == 1
    late = np.array(
        [
            [1024, 512, 256, 128],
            [2, 4, 8, 16],
            [4, 8, 16, 32],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    assert resolve_search_depth(late, depth=2, adaptive=True) == 2
    assert resolve_search_depth(late, depth=2, adaptive=False) == 2
    assert resolve_search_depth(early, depth=2, adaptive=False) == 2


def test_depth2_prefers_reply_that_1ply_misses():
    board = np.array(
        [
            [2, 2, 0, 0],
            [4, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int32,
    )
    left_succs, _ = expand_action_successors(board, ACTION_LEFT, with_obs=False)
    down_succs, _ = expand_action_successors(board, ACTION_DOWN, with_obs=False)
    assert left_succs is not None and down_succs is not None

    def board_key(b: np.ndarray) -> tuple[int, ...]:
        return tuple(np.asarray(b, dtype=np.int32).reshape(16).tolist())

    left_set = {board_key(s.board) for s in left_succs}
    down_grand = set()
    for succ in down_succs:
        child_mask = valid_action_mask_fast(succ.board)
        for a2 in np.flatnonzero(child_mask):
            exp2 = expand_action_successors(succ.board, int(a2), with_obs=False)
            assert exp2 is not None
            for s2 in exp2[0]:
                down_grand.add(board_key(s2.board))

    def q_batch_fn(obs_batch: np.ndarray) -> np.ndarray:
        # Reconstruct tiles from exponent observations.
        tiles = np.zeros_like(obs_batch, dtype=np.int32)
        nz = obs_batch > 0
        tiles[nz] = np.left_shift(1, np.rint(obs_batch[nz]).astype(np.int32))
        q = np.zeros((obs_batch.shape[0], 4), dtype=np.float32)
        for i, row in enumerate(tiles):
            key = tuple(row.tolist())
            if key in down_grand:
                q[i] = 100.0
            elif key in left_set:
                q[i] = 50.0
        return q

    kwargs = dict(gamma=1.0, reward_mode="raw", include_reward=False)
    v1 = expectimax_action_values(board, q_batch_fn, depth=1, **kwargs)
    v2 = expectimax_action_values(board, q_batch_fn, depth=2, **kwargs)
    assert v1[ACTION_LEFT] > v1[ACTION_DOWN]
    assert v2[ACTION_DOWN] > v2[ACTION_LEFT]


def test_corner_tiebreak_keeps_max_tile():
    board = np.array(
        [
            [1024, 0, 0, 2],
            [4, 8, 16, 32],
            [8, 16, 32, 64],
            [16, 32, 64, 128],
        ],
        dtype=np.int32,
    )

    def q_batch_fn(obs_batch: np.ndarray) -> np.ndarray:
        return np.ones((obs_batch.shape[0], 4), dtype=np.float32)

    action = expectimax_select_action(
        board,
        q_batch_fn,
        gamma=1.0,
        reward_mode="raw",
        include_reward=False,
        depth=1,
        corner_tiebreak=True,
        corner_margin=10.0,
    )
    after, _, changed = move_board_fast(board, action)
    assert changed
    assert 1024 in (after[0, 0], after[0, 3], after[3, 0], after[3, 3])
    assert action != ACTION_RIGHT
