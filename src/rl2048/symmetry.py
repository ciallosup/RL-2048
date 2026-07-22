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


def all_transform_ids() -> range:
    return range(len(_TRANSFORM_SPECS))
