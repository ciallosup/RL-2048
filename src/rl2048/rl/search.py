"""Expectimax over 2048 spawn dynamics, using a Q-network as leaf values."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rl2048.core import NUM_ACTIONS, TILE_FOUR_PROB, TILE_TWO_PROB
from rl2048.rl.fast_move import boards_to_obs, move_board_fast, valid_action_mask_fast, valid_action_masks_batch
from rl2048.rl.rewards import transform_reward

SPAWN_VALUES = ((2, TILE_TWO_PROB), (4, TILE_FOUR_PROB))
CORNER_CELLS = ((0, 0), (0, 3), (3, 0), (3, 3))


@dataclass(frozen=True)
class SpawnSuccessor:
    board: np.ndarray
    obs: np.ndarray | None
    prob: float


def expand_action_successors(
    board: np.ndarray,
    action: int,
    *,
    with_obs: bool = True,
) -> tuple[list[SpawnSuccessor], int] | None:
    """
    After a legal move, enumerate spawn outcomes.

    Returns (successors, merge_score), or None if the action does not change the board.
    Each successor probability is (1 / n_empty) * P(tile).
    """
    after, merge_score, changed = move_board_fast(board, action)
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
            obs = None
            if with_obs:
                obs = boards_to_obs([nxt])[0]
            successors.append(
                SpawnSuccessor(
                    board=nxt,
                    obs=obs,
                    prob=(1.0 / n_empty) * tile_prob,
                )
            )
    return successors, int(merge_score)


def _leaf_values(q_batch: np.ndarray, boards: list[np.ndarray]) -> np.ndarray:
    """V(s') = max legal Q, or 0 if the successor is terminal."""
    stacked = np.stack([np.asarray(b, dtype=np.int32).reshape(4, 4) for b in boards], axis=0)
    masks = valid_action_masks_batch(stacked)
    neg_inf = np.finfo(np.float32).min
    masked = np.where(masks, q_batch, neg_inf)
    values = masked.max(axis=1)
    values = np.where(masks.any(axis=1), values, 0.0)
    return values.astype(np.float32)


def resolve_search_depth(board: np.ndarray, depth: int, adaptive: bool) -> int:
    """Pick 1/2/3 ply. Depth 3 only on cramped high-tile boards."""
    depth = max(1, int(depth))
    arr = np.asarray(board)
    empty = int(np.count_nonzero(arr == 0))
    max_tile = int(arr.max())

    if depth >= 3:
        # Keep depth-3 branching tiny: 3-ply with 5–6 empties is thousands of
        # leaves per move and several minutes per 4096 game.
        cramped_2048 = max_tile >= 2048 and empty <= 4
        cramped_1024 = max_tile >= 1024 and empty <= 3
        if cramped_2048 or cramped_1024:
            return 3
        return 2

    if depth == 1 or not adaptive:
        return depth
    if max_tile >= 1024 and empty <= 8:
        return depth
    if empty <= 4:
        return depth
    return 1


def _max_tile_in_corner(board: np.ndarray) -> bool:
    arr = np.asarray(board)
    max_tile = int(arr.max())
    if max_tile <= 0:
        return False
    return any(int(arr[r, c]) == max_tile for r, c in CORNER_CELLS)


def _expectimax_depth1(
    board: np.ndarray,
    q_batch_fn,
    *,
    gamma: float,
    reward_mode: str,
    include_reward: bool,
) -> np.ndarray:
    values = np.full(NUM_ACTIONS, -np.inf, dtype=np.float32)
    mask = valid_action_mask_fast(board)
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return values

    all_boards: list[np.ndarray] = []
    index_ranges: list[tuple[int, int, int, float, np.ndarray]] = []

    for action in legal:
        expanded = expand_action_successors(board, int(action), with_obs=False)
        assert expanded is not None
        successors, merge_score = expanded
        train_r = transform_reward(float(merge_score), reward_mode) if include_reward else 0.0
        if not successors:
            values[int(action)] = np.float32(train_r)
            continue
        start = len(all_boards)
        probs = np.empty(len(successors), dtype=np.float32)
        for i, succ in enumerate(successors):
            all_boards.append(succ.board)
            probs[i] = succ.prob
        index_ranges.append((int(action), start, len(all_boards), float(train_r), probs))

    if all_boards:
        obs_batch = boards_to_obs(all_boards)
        q_batch = np.asarray(q_batch_fn(obs_batch), dtype=np.float32)
        if q_batch.ndim != 2 or q_batch.shape[0] != len(all_boards):
            raise ValueError(f"q_batch_fn returned shape {q_batch.shape}, expected ({len(all_boards)}, 4)")
        leaves = _leaf_values(q_batch, all_boards)
        for action, start, end, train_r, probs in index_ranges:
            expected_v = float(np.dot(probs, leaves[start:end]))
            values[action] = np.float32(train_r + gamma * expected_v)

    return values


def _expectimax_depth2(
    board: np.ndarray,
    q_batch_fn,
    *,
    gamma: float,
    reward_mode: str,
    include_reward: bool,
) -> np.ndarray:
    """Two (max, chance) layers; leaves are V = masked max Q."""
    values = np.full(NUM_ACTIONS, -np.inf, dtype=np.float32)
    mask = valid_action_mask_fast(board)
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return values

    # root action -> (r1, list of (p1, child_spec))
    # child_spec is list of action-2 nodes:
    #   ("dead",) | ("term", r2) | ("exp", r2, start, end, probs)
    root_nodes: list[tuple[int, float, list[tuple[float, list]]]] = []
    leaf_boards: list[np.ndarray] = []

    for action in legal:
        expanded = expand_action_successors(board, int(action), with_obs=False)
        assert expanded is not None
        successors, merge_score = expanded
        train_r = transform_reward(float(merge_score), reward_mode) if include_reward else 0.0
        if not successors:
            values[int(action)] = np.float32(train_r)
            continue
        children: list[tuple[float, list]] = []
        for succ in successors:
            child = succ.board
            child_mask = valid_action_mask_fast(child)
            child_legal = np.flatnonzero(child_mask)
            a2_nodes: list = []
            if child_legal.size == 0:
                a2_nodes.append(("dead",))
            else:
                for a2 in child_legal:
                    exp2 = expand_action_successors(child, int(a2), with_obs=False)
                    assert exp2 is not None
                    succs2, score2 = exp2
                    r2 = transform_reward(float(score2), reward_mode) if include_reward else 0.0
                    if not succs2:
                        a2_nodes.append(("term", float(r2)))
                        continue
                    start = len(leaf_boards)
                    probs2 = np.empty(len(succs2), dtype=np.float32)
                    for i, s2 in enumerate(succs2):
                        leaf_boards.append(s2.board)
                        probs2[i] = s2.prob
                    a2_nodes.append(("exp", float(r2), start, len(leaf_boards), probs2))
            children.append((float(succ.prob), a2_nodes))
        root_nodes.append((int(action), float(train_r), children))

    if leaf_boards:
        obs_batch = boards_to_obs(leaf_boards)
        q_batch = np.asarray(q_batch_fn(obs_batch), dtype=np.float32)
        if q_batch.ndim != 2 or q_batch.shape[0] != len(leaf_boards):
            raise ValueError(
                f"q_batch_fn returned shape {q_batch.shape}, expected ({len(leaf_boards)}, 4)"
            )
        leaves = _leaf_values(q_batch, leaf_boards)
    else:
        leaves = np.zeros(0, dtype=np.float32)

    for action, train_r, children in root_nodes:
        expected_v = 0.0
        for p1, a2_nodes in children:
            best = -np.inf
            for node in a2_nodes:
                kind = node[0]
                if kind == "dead":
                    best = max(best, 0.0)
                elif kind == "term":
                    best = max(best, float(node[1]))
                else:
                    _, r2, start, end, probs2 = node
                    ev = float(np.dot(probs2, leaves[start:end]))
                    best = max(best, float(r2) + gamma * ev)
            if not np.isfinite(best):
                best = 0.0
            expected_v += p1 * best
        values[action] = np.float32(train_r + gamma * expected_v)

    return values


def _build_state(
    board: np.ndarray,
    remaining: int,
    leaf_boards: list[np.ndarray],
    *,
    reward_mode: str,
    include_reward: bool,
):
    """Tree spec: ('L', idx) leaf, ('D',) dead, ('M', actions)."""
    if remaining <= 0:
        leaf_boards.append(np.asarray(board, dtype=np.int32).reshape(4, 4).copy())
        return ("L", len(leaf_boards) - 1)
    mask = valid_action_mask_fast(board)
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return ("D",)
    acts: list = []
    for action in legal:
        expanded = expand_action_successors(board, int(action), with_obs=False)
        assert expanded is not None
        successors, merge_score = expanded
        train_r = transform_reward(float(merge_score), reward_mode) if include_reward else 0.0
        if not successors:
            acts.append((int(action), float(train_r), None))
            continue
        kids = [
            (
                float(succ.prob),
                _build_state(
                    succ.board,
                    remaining - 1,
                    leaf_boards,
                    reward_mode=reward_mode,
                    include_reward=include_reward,
                ),
            )
            for succ in successors
        ]
        acts.append((int(action), float(train_r), kids))
    return ("M", acts)


def _eval_spec(spec, leaves: np.ndarray, gamma: float) -> float:
    kind = spec[0]
    if kind == "L":
        return float(leaves[spec[1]])
    if kind == "D":
        return 0.0
    best = -np.inf
    for _action, train_r, kids in spec[1]:
        if kids is None:
            val = float(train_r)
        else:
            expected = 0.0
            for prob, child in kids:
                expected += prob * _eval_spec(child, leaves, gamma)
            val = float(train_r) + gamma * expected
        if val > best:
            best = val
    return 0.0 if not np.isfinite(best) else float(best)


def _expectimax_general(
    board: np.ndarray,
    q_batch_fn,
    *,
    gamma: float,
    reward_mode: str,
    include_reward: bool,
    depth: int,
) -> np.ndarray:
    """Depth >= 3: extra (max, chance) layers, then V = masked max Q."""
    values = np.full(NUM_ACTIONS, -np.inf, dtype=np.float32)
    mask = valid_action_mask_fast(board)
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return values

    leaf_boards: list[np.ndarray] = []
    root_acts: list[tuple[int, float, list]] = []
    for action in legal:
        expanded = expand_action_successors(board, int(action), with_obs=False)
        assert expanded is not None
        successors, merge_score = expanded
        train_r = transform_reward(float(merge_score), reward_mode) if include_reward else 0.0
        if not successors:
            values[int(action)] = np.float32(train_r)
            continue
        kids = [
            (
                float(succ.prob),
                _build_state(
                    succ.board,
                    depth - 1,
                    leaf_boards,
                    reward_mode=reward_mode,
                    include_reward=include_reward,
                ),
            )
            for succ in successors
        ]
        root_acts.append((int(action), float(train_r), kids))

    if leaf_boards:
        obs_batch = boards_to_obs(leaf_boards)
        q_batch = np.asarray(q_batch_fn(obs_batch), dtype=np.float32)
        if q_batch.ndim != 2 or q_batch.shape[0] != len(leaf_boards):
            raise ValueError(
                f"q_batch_fn returned shape {q_batch.shape}, expected ({len(leaf_boards)}, 4)"
            )
        leaves = _leaf_values(q_batch, leaf_boards)
    else:
        leaves = np.zeros(0, dtype=np.float32)

    for action, train_r, kids in root_acts:
        expected_v = 0.0
        for prob, child in kids:
            expected_v += prob * _eval_spec(child, leaves, gamma)
        values[action] = np.float32(train_r + gamma * expected_v)
    return values


def expectimax_action_values(
    board: np.ndarray,
    q_batch_fn,
    *,
    gamma: float,
    reward_mode: str,
    include_reward: bool = True,
    depth: int = 1,
    adaptive: bool = False,
) -> np.ndarray:
    """
    Expectimax backups for all legal actions.

    depth=1: one move + spawn, V = masked max Q (original C0).
    depth=2: an extra reply + spawn before the Q leaves.
    depth=3: one more ply; resolve_search_depth only uses it on cramped 1024/2048 boards.

    q_batch_fn: callable mapping obs array (N, 16) -> Q array (N, 4).
    Illegal actions stay -inf so argmax ignores them.
    """
    used_depth = resolve_search_depth(board, depth, adaptive)
    kwargs = dict(gamma=gamma, reward_mode=reward_mode, include_reward=include_reward)
    if used_depth <= 1:
        return _expectimax_depth1(board, q_batch_fn, **kwargs)
    if used_depth == 2:
        return _expectimax_depth2(board, q_batch_fn, **kwargs)
    return _expectimax_general(board, q_batch_fn, depth=used_depth, **kwargs)


def expectimax_select_action(
    board: np.ndarray,
    q_batch_fn,
    *,
    gamma: float,
    reward_mode: str,
    include_reward: bool = True,
    depth: int = 1,
    adaptive: bool = False,
    corner_tiebreak: bool = False,
    corner_margin: float = 2.0,
) -> int:
    values = expectimax_action_values(
        board,
        q_batch_fn,
        gamma=gamma,
        reward_mode=reward_mode,
        include_reward=include_reward,
        depth=depth,
        adaptive=adaptive,
    )
    if not np.isfinite(values).any():
        raise RuntimeError("No legal action for expectimax.")
    if not corner_tiebreak:
        return int(np.argmax(values))

    margin = float(corner_margin)
    if int(np.asarray(board).max()) >= 2048:
        margin = max(margin, 4.0)

    finite = np.isfinite(values)
    best = float(np.max(values[finite]))
    candidates = [
        action
        for action in range(NUM_ACTIONS)
        if finite[action] and float(values[action]) >= best - margin
    ]
    if len(candidates) == 1:
        return int(candidates[0])

    ranked: list[tuple[bool, float, int]] = []
    for action in candidates:
        after, _, changed = move_board_fast(board, action)
        if not changed:
            continue
        ranked.append((_max_tile_in_corner(after), float(values[action]), int(action)))
    if not ranked:
        return int(np.argmax(values))
    ranked.sort(key=lambda item: (item[0], item[1], -item[2]))
    return int(ranked[-1][2])
