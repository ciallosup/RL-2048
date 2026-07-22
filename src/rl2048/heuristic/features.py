"""Board heuristic features for non-learning baseline (roadmap section 4)."""

from __future__ import annotations

import math

import numpy as np

# Classic 2048 expectimax-style weights (empty / monotonicity / smoothness).
WEIGHT_EMPTY = 2.7
WEIGHT_MONOTONICITY = 1.0
WEIGHT_SMOOTHNESS = 0.1
WEIGHT_CORNER = 3.0


def _tile_log(value: int) -> float:
    return math.log2(value) if value > 0 else 0.0


def count_empty(board: np.ndarray) -> int:
    return int(np.count_nonzero(board == 0))


def monotonicity(board: np.ndarray) -> float:
    """Best monotonicity score over four directions on rows and columns."""
    totals = [0.0, 0.0, 0.0, 0.0]
    for i in range(4):
        _accumulate_line(board[i], totals)
        _accumulate_line(board[:, i], totals)
    return max(totals)


def _accumulate_line(line: np.ndarray, totals: list[float]) -> None:
    current = 0
    while current < 4:
        nxt = current + 1
        while nxt < 4 and line[nxt] == 0:
            nxt += 1
        if nxt >= 4:
            break
        cur_val = int(line[current]) if line[current] else 0
        nxt_val = int(line[nxt])
        if cur_val > nxt_val:
            totals[0] += nxt_val - cur_val
        elif nxt_val > cur_val:
            totals[2] += cur_val - nxt_val
        current = nxt


def smoothness(board: np.ndarray) -> float:
    score = 0.0
    for row in range(4):
        for col in range(3):
            if board[row, col] and board[row, col + 1]:
                score -= abs(_tile_log(int(board[row, col])) - _tile_log(int(board[row, col + 1])))
    for col in range(4):
        for row in range(3):
            if board[row, col] and board[row + 1, col]:
                score -= abs(_tile_log(int(board[row, col])) - _tile_log(int(board[row + 1, col])))
    return score


def corner_max_bonus(board: np.ndarray) -> float:
    """Small binary bonus when the largest tile sits in a corner."""
    max_tile = int(board.max(initial=0))
    if max_tile == 0:
        return 0.0
    if max_tile in (board[0, 0], board[0, 3], board[3, 0], board[3, 3]):
        return 1.0
    return 0.0


def board_heuristic_score(board: np.ndarray, *, merge_score: int = 0) -> float:
    return (
        WEIGHT_EMPTY * count_empty(board)
        + WEIGHT_MONOTONICITY * monotonicity(board)
        + WEIGHT_SMOOTHNESS * smoothness(board)
        + WEIGHT_CORNER * corner_max_bonus(board)
        + 0.05 * merge_score
    )
