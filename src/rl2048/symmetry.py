"""D4 symmetry transforms for board/action pairs."""

from __future__ import annotations

import numpy as np

from rl2048.core import ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP, NUM_ACTIONS

# Transform index: rotation quarter-turns CCW (0..3), then optional horizontal flip.
_TRANSFORM_SPECS: list[tuple[int, bool]] = [
    (0, False),
    (1, False),
    (2, False),
    (3, False),
    (0, True),
    (1, True),
    (2, True),
    (3, True),
]


def _rotate_board(board: np.ndarray, quarters: int) -> np.ndarray:
    return np.rot90(board, k=quarters)


def _flip_board_horizontal(board: np.ndarray) -> np.ndarray:
    return np.fliplr(board)


def transform_board(board: np.ndarray, transform_id: int) -> np.ndarray:
    if not 0 <= transform_id < len(_TRANSFORM_SPECS):
        raise ValueError(f"transform_id must be in [0, {len(_TRANSFORM_SPECS)}), got {transform_id}")
    rotation, flip = _TRANSFORM_SPECS[transform_id]
    result = _rotate_board(board, rotation)
    if flip:
        result = _flip_board_horizontal(result)
    return result


def _rotate_action(action: int, quarters: int) -> int:
    """Rotate action labels with the board (CCW)."""
    cycle = (ACTION_UP, ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT)
    idx = cycle.index(action)
    return cycle[(idx + quarters) % 4]


def _flip_action_horizontal(action: int) -> int:
    mapping = {
        ACTION_UP: ACTION_UP,
        ACTION_DOWN: ACTION_DOWN,
        ACTION_LEFT: ACTION_RIGHT,
        ACTION_RIGHT: ACTION_LEFT,
    }
    return mapping[action]


def transform_action(action: int, transform_id: int) -> int:
    if not 0 <= action < NUM_ACTIONS:
        raise ValueError(f"Invalid action: {action}")
    rotation, flip = _TRANSFORM_SPECS[transform_id]
    transformed = _rotate_action(action, rotation)
    if flip:
        transformed = _flip_action_horizontal(transformed)
    return transformed


def transform_mask(mask: np.ndarray, transform_id: int) -> np.ndarray:
    """Permute a length-4 (or batched) action validity mask under a D4 transform."""
    arr = np.asarray(mask, dtype=np.bool_)
    if arr.ndim == 1:
        out = np.zeros(NUM_ACTIONS, dtype=np.bool_)
        for action in range(NUM_ACTIONS):
            if arr[action]:
                out[transform_action(action, transform_id)] = True
        return out
    if arr.ndim == 2:
        return np.stack([transform_mask(row, transform_id) for row in arr], axis=0)
    raise ValueError(f"mask must be 1D or 2D, got shape {arr.shape}")


def all_transform_ids() -> range:
    return range(len(_TRANSFORM_SPECS))


def _build_flat_perms() -> np.ndarray:
    """perms[t, i] = source flat index for destination i under transform t."""
    base = np.arange(16, dtype=np.int64).reshape(4, 4)
    perms = np.empty((len(_TRANSFORM_SPECS), 16), dtype=np.int64)
    for tid in range(len(_TRANSFORM_SPECS)):
        perms[tid] = transform_board(base, tid).reshape(-1)
    return perms


def _build_action_map() -> np.ndarray:
    """action_map[t, a] = transformed action."""
    action_map = np.empty((len(_TRANSFORM_SPECS), NUM_ACTIONS), dtype=np.int64)
    for tid in range(len(_TRANSFORM_SPECS)):
        for action in range(NUM_ACTIONS):
            action_map[tid, action] = transform_action(action, tid)
    return action_map


FLAT_PERMS = _build_flat_perms()
ACTION_MAP = _build_action_map()


def transform_flat_obs_batch(states: np.ndarray, transform_ids: np.ndarray) -> np.ndarray:
    """Vectorized D4 transform for flat observations. states: (B, 16)."""
    flat = np.asarray(states, dtype=np.float32).reshape(-1, 16)
    tids = np.asarray(transform_ids, dtype=np.int64).reshape(-1)
    perms = FLAT_PERMS[tids]
    batch_idx = np.arange(flat.shape[0])[:, None]
    return flat[batch_idx, perms]


def transform_actions_batch(actions: np.ndarray, transform_ids: np.ndarray) -> np.ndarray:
    acts = np.asarray(actions, dtype=np.int64).reshape(-1)
    tids = np.asarray(transform_ids, dtype=np.int64).reshape(-1)
    return ACTION_MAP[tids, acts]


def transform_masks_batch(masks: np.ndarray, transform_ids: np.ndarray) -> np.ndarray:
    arr = np.asarray(masks, dtype=np.bool_)
    tids = np.asarray(transform_ids, dtype=np.int64).reshape(-1)
    out = np.zeros_like(arr)
    batch_idx = np.arange(arr.shape[0])
    for action in range(NUM_ACTIONS):
        out[batch_idx, ACTION_MAP[tids, action]] = arr[batch_idx, action]
    return out
