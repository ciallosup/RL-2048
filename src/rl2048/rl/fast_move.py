"""Row-lookup 2048 moves for search. Must match ``core.move_board``."""

from __future__ import annotations

import numpy as np

from rl2048.core import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    NUM_ACTIONS,
    _merge_line,
)

_ROW_AFTER = np.zeros(65536, dtype=np.uint16)
_ROW_SCORE = np.zeros(65536, dtype=np.int32)


def _pack_exp_row(exps: np.ndarray | list[int]) -> int:
    packed = 0
    for i, exp in enumerate(exps):
        packed |= (int(exp) & 15) << (4 * i)
    return packed


def _unpack_exp_row(packed: int) -> np.ndarray:
    return np.array([(packed >> (4 * i)) & 15 for i in range(4)], dtype=np.int32)


def _tiles_from_exp_row(exps: np.ndarray) -> np.ndarray:
    tiles = np.zeros(4, dtype=np.int32)
    for i, exp in enumerate(exps):
        if exp:
            tiles[i] = 1 << int(exp)
    return tiles


def _exp_from_tiles_row(tiles: np.ndarray) -> np.ndarray:
    exps = np.zeros(4, dtype=np.int32)
    for i, val in enumerate(tiles):
        if val:
            exps[i] = int(val).bit_length() - 1
    return exps


def _build_row_lut() -> None:
    for packed in range(65536):
        tiles = _tiles_from_exp_row(_unpack_exp_row(packed))
        merged, score = _merge_line(tiles)
        _ROW_AFTER[packed] = _pack_exp_row(_exp_from_tiles_row(merged))
        _ROW_SCORE[packed] = int(score)


_build_row_lut()


def reverse_nibbles(packed: np.ndarray | int) -> np.ndarray | int:
    """Reverse four 4-bit cells in a 16-bit row."""
    p = packed
    return ((p & 0xF) << 12) | ((p & 0xF0) << 4) | ((p & 0xF00) >> 4) | ((p & 0xF000) >> 12)


def _pack_exp_planes(exp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pack (N,4,4) exponents into row and column uint16 arrays of shape (N, 4)."""

    def pack(a, b, c, d):
        return (a | (b << 4) | (c << 8) | (d << 12)).astype(np.int32)

    rows = np.stack(
        [pack(exp[:, r, 0], exp[:, r, 1], exp[:, r, 2], exp[:, r, 3]) for r in range(4)],
        axis=1,
    )
    cols = np.stack(
        [pack(exp[:, 0, c], exp[:, 1, c], exp[:, 2, c], exp[:, 3, c]) for c in range(4)],
        axis=1,
    )
    return rows, cols


def tiles_to_exp_batch(boards: np.ndarray) -> np.ndarray:
    arr = np.asarray(boards, dtype=np.int32).reshape(-1, 4, 4)
    exp = np.zeros_like(arr)
    nz = arr > 0
    if np.any(nz):
        exp[nz] = np.rint(np.log2(arr[nz].astype(np.float64))).astype(np.int32)
    return exp


def valid_action_masks_batch(boards: np.ndarray) -> np.ndarray:
    """Vectorized legal-action masks. boards: (N,4,4) or list -> (N,4) bool."""
    arr = np.asarray(boards, dtype=np.int32).reshape(-1, 4, 4)
    exp = tiles_to_exp_batch(arr)
    rows, cols = _pack_exp_planes(exp)
    left_legal = np.any(_ROW_AFTER[rows] != rows, axis=1)
    right_legal = np.any(reverse_nibbles(_ROW_AFTER[reverse_nibbles(rows)]) != rows, axis=1)
    up_legal = np.any(_ROW_AFTER[cols] != cols, axis=1)
    down_legal = np.any(reverse_nibbles(_ROW_AFTER[reverse_nibbles(cols)]) != cols, axis=1)
    # ACTION_UP, DOWN, LEFT, RIGHT
    return np.stack([up_legal, down_legal, left_legal, right_legal], axis=1)


def _pack_tile_row(row: np.ndarray) -> int:
    packed = 0
    for i, val in enumerate(row):
        if val:
            packed |= ((int(val).bit_length() - 1) & 15) << (4 * i)
    return packed


def _unpack_tile_row(packed: int) -> np.ndarray:
    row = np.zeros(4, dtype=np.int32)
    for i in range(4):
        exp = (packed >> (4 * i)) & 15
        if exp:
            row[i] = 1 << exp
    return row


def _move_left_tiles(board: np.ndarray) -> tuple[np.ndarray, int, bool]:
    out = np.empty((4, 4), dtype=np.int32)
    total = 0
    changed = False
    for row in range(4):
        packed = _pack_tile_row(board[row])
        new_packed = int(_ROW_AFTER[packed])
        total += int(_ROW_SCORE[packed])
        if new_packed != packed:
            changed = True
        out[row] = _unpack_tile_row(new_packed)
    return out, total, changed


def move_board_fast(board: np.ndarray, action: int) -> tuple[np.ndarray, int, bool]:
    """Same contract as ``core.move_board``: (after, merge_score, changed)."""
    b = np.asarray(board, dtype=np.int32).reshape(4, 4)
    if action == ACTION_LEFT:
        return _move_left_tiles(b)
    if action == ACTION_RIGHT:
        after, score, changed = _move_left_tiles(np.fliplr(b))
        return np.fliplr(after), score, changed
    if action == ACTION_UP:
        after, score, changed = _move_left_tiles(b.T)
        return after.T.copy(), score, changed
    if action == ACTION_DOWN:
        after, score, changed = _move_left_tiles(np.fliplr(b.T))
        return np.fliplr(after).T.copy(), score, changed
    raise ValueError(f"Invalid action: {action}")


def valid_action_mask_fast(board: np.ndarray) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for action in range(NUM_ACTIONS):
        _, _, changed = move_board_fast(board, action)
        mask[action] = changed
    return mask


def boards_to_obs(boards: list[np.ndarray]) -> np.ndarray:
    """Batch tile boards -> flat exponent observations (N, 16)."""
    arr = np.stack([np.asarray(b, dtype=np.int32).reshape(16) for b in boards], axis=0)
    obs = np.zeros((arr.shape[0], 16), dtype=np.float32)
    nz = arr > 0
    if np.any(nz):
        obs[nz] = np.log2(arr[nz].astype(np.float32))
    return obs
