"""1-ply expectimax over 2048 spawn dynamics, using a Q-network as leaf values."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl2048.core import (
    NUM_ACTIONS,
    TILE_FOUR_PROB,
    TILE_TWO_PROB,
    exponents_to_observation,
    move_board,
    valid_action_mask,
)
from rl2048.rl.rewards import transform_reward

SPAWN_VALUES = ((2, TILE_TWO_PROB), (4, TILE_FOUR_PROB))


@dataclass(frozen=True)
class SpawnSuccessor:
    board: np.ndarray
    obs: np.ndarray
    prob: float


def expand_action_successors(board: np.ndarray, action: int) -> tuple[list[SpawnSuccessor], int] | None:
    """
    After a legal move, enumerate spawn outcomes.

    Returns (successors, merge_score), or None if the action does not change the board.
    Each successor probability is (1 / n_empty) * P(tile).
    """
    after, merge_score, changed = move_board(board, action)
    if not changed:
        return None
    empty = [(int(r), int(c)) for r, c in np.argwhere(after == 0)]
    if not empty:
        return [], int(merge_score)
    n_empty = len(empty)
    successors: list[SpawnSuccessor] = []
    for row, col in empty:
        for value, tile_prob in SPAWN_VALUES:
            nxt = after.copy()
            nxt[row, col] = value
            successors.append(
                SpawnSuccessor(
                    board=nxt,
                    obs=exponents_to_observation(nxt).astype(np.float32),
                    prob=(1.0 / n_empty) * tile_prob,
                )
            )
    return successors, int(merge_score)


def _leaf_values(q_batch: np.ndarray, boards: list[np.ndarray]) -> np.ndarray:
    """V(s') = max legal Q, or 0 if the successor is terminal."""
    masks = np.stack([valid_action_mask(b) for b in boards], axis=0)
    neg_inf = np.finfo(np.float32).min
    masked = np.where(masks, q_batch, neg_inf)
    values = masked.max(axis=1)
    values = np.where(masks.any(axis=1), values, 0.0)
    return values.astype(np.float32)


def expectimax_action_values(
    board: np.ndarray,
    q_batch_fn,
    *,
    gamma: float,
    reward_mode: str,
    include_reward: bool = True,
) -> np.ndarray:
    """
    1-ply backups for all legal actions.

    q_batch_fn: callable mapping obs array (N, 16) -> Q array (N, 4).
    Illegal actions stay -inf so argmax ignores them.
    """
    values = np.full(NUM_ACTIONS, -np.inf, dtype=np.float32)
    mask = valid_action_mask(board)
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return values

    all_obs: list[np.ndarray] = []
    all_boards: list[np.ndarray] = []
    # action, start, end, transformed_r, spawn probabilities
    index_ranges: list[tuple[int, int, int, float, np.ndarray]] = []

    for action in legal:
        expanded = expand_action_successors(board, action)
        assert expanded is not None
        successors, merge_score = expanded
        train_r = transform_reward(float(merge_score), reward_mode) if include_reward else 0.0
        if not successors:
            values[int(action)] = np.float32(train_r)
            continue
        start = len(all_obs)
        probs = np.empty(len(successors), dtype=np.float32)
        for i, succ in enumerate(successors):
            all_obs.append(succ.obs)
            all_boards.append(succ.board)
            probs[i] = succ.prob
        index_ranges.append((int(action), start, len(all_obs), float(train_r), probs))

    if all_obs:
        obs_batch = np.stack(all_obs, axis=0)
        q_batch = np.asarray(q_batch_fn(obs_batch), dtype=np.float32)
        if q_batch.ndim != 2 or q_batch.shape[0] != len(all_obs):
            raise ValueError(f"q_batch_fn returned shape {q_batch.shape}, expected ({len(all_obs)}, 4)")
        leaves = _leaf_values(q_batch, all_boards)
        for action, start, end, train_r, probs in index_ranges:
            expected_v = float(np.dot(probs, leaves[start:end]))
            values[action] = np.float32(train_r + gamma * expected_v)

    return values


def expectimax_select_action(
    board: np.ndarray,
    q_batch_fn,
    *,
    gamma: float,
    reward_mode: str,
    include_reward: bool = True,
) -> int:
    values = expectimax_action_values(
        board,
        q_batch_fn,
        gamma=gamma,
        reward_mode=reward_mode,
        include_reward=include_reward,
    )
    if not np.isfinite(values).any():
        raise RuntimeError("No legal action for expectimax.")
    return int(np.argmax(values))
