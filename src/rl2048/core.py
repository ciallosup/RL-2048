"""Core 2048 board logic (rules frozen per experiment roadmap section 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

# 0=up, 1=down, 2=left, 3=right
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3

ACTION_NAMES = ("up", "down", "left", "right")
NUM_ACTIONS = 4

BOARD_SIZE = 4
TILE_TWO_PROB = 0.9
TILE_FOUR_PROB = 0.1
TARGET_TILE = 2048


def tile_to_exponent(value: int) -> int:
    """Map board tile value to exponential encoding (empty -> 0, 2 -> 1, 4 -> 2, ...)."""
    if value == 0:
        return 0
    exp = 0
    v = value
    while v > 1:
        v //= 2
        exp += 1
    return exp


def board_to_exponents(board: np.ndarray) -> np.ndarray:
    """Return 4x4 exponential encoding."""
    return np.vectorize(tile_to_exponent, otypes=[np.int32])(board)


def exponents_to_observation(board: np.ndarray) -> np.ndarray:
    """Flat 16-dim exponential observation."""
    return board_to_exponents(board).reshape(-1).astype(np.int32)


def _merge_line(line: np.ndarray) -> tuple[np.ndarray, int]:
    """Slide and merge one row left; each tile merges at most once per step."""
    non_zero = line[line != 0]
    if non_zero.size == 0:
        return np.zeros(BOARD_SIZE, dtype=np.int32), 0

    merged: list[int] = []
    merge_score = 0
    i = 0
    while i < non_zero.size:
        if i + 1 < non_zero.size and non_zero[i] == non_zero[i + 1]:
            new_val = int(non_zero[i] * 2)
            merged.append(new_val)
            merge_score += new_val
            i += 2
        else:
            merged.append(int(non_zero[i]))
            i += 1

    result = np.zeros(BOARD_SIZE, dtype=np.int32)
    result[: len(merged)] = merged
    return result, merge_score


def _transform_board(board: np.ndarray, action: int) -> np.ndarray:
    """Orient board so the chosen action becomes a left merge."""
    if action == ACTION_LEFT:
        return board.copy()
    if action == ACTION_RIGHT:
        return np.fliplr(board)
    if action == ACTION_UP:
        return board.T
    if action == ACTION_DOWN:
        return np.fliplr(board.T)
    raise ValueError(f"Invalid action: {action}")


def _inverse_transform_board(board: np.ndarray, action: int) -> np.ndarray:
    """Restore board orientation after left-merge."""
    if action == ACTION_LEFT:
        return board
    if action == ACTION_RIGHT:
        return np.fliplr(board)
    if action == ACTION_UP:
        return board.T
    if action == ACTION_DOWN:
        return np.fliplr(board).T
    raise ValueError(f"Invalid action: {action}")


def move_board(board: np.ndarray, action: int) -> tuple[np.ndarray, int, bool]:
    """
    Apply one move.

    Returns (new_board, merge_score, changed).
    """
    oriented = _transform_board(board, action)
    new_rows = []
    total_score = 0
    for row in oriented:
        merged_row, row_score = _merge_line(row)
        new_rows.append(merged_row)
        total_score += row_score
    result = _inverse_transform_board(np.stack(new_rows), action)
    changed = not np.array_equal(result, board)
    return result, total_score, changed


def valid_action_mask(board: np.ndarray) -> np.ndarray:
    """Boolean mask of actions that change the board."""
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    for action in range(NUM_ACTIONS):
        _, _, changed = move_board(board, action)
        mask[action] = changed
    return mask


def max_tile_value(board: np.ndarray) -> int:
    return int(board.max(initial=0))


@dataclass
class StepResult:
    board: np.ndarray
    merge_score: int
    changed: bool
    spawned_value: int
    spawn_position: tuple[int, int] | None


class Board2048:
    """
    Mutable 2048 game state.

    Rules:
    - Spawn only after a valid (board-changing) move.
    - New tile: uniform empty cell; 2 with p=0.9, 4 with p=0.1.
    - Reaching 2048 does not end the episode.
    """

    def __init__(self, rng: np.random.Generator | None = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
        self.game_score = 0
        self.step_count = 0
        self.reached_2048 = False
        self._spawn_tile()
        self._spawn_tile()

    def copy(self) -> Board2048:
        clone = Board2048.__new__(Board2048)
        clone.rng = self.rng
        clone.board = self.board.copy()
        clone.game_score = self.game_score
        clone.step_count = self.step_count
        clone.reached_2048 = self.reached_2048
        return clone

    def observation(self) -> np.ndarray:
        return exponents_to_observation(self.board)

    def info(self) -> dict:
        return {
            "valid_action_mask": valid_action_mask(self.board),
            "merge_score": 0,
            "game_score": self.game_score,
            "max_tile": max_tile_value(self.board),
            "episode_length": self.step_count,
            "reached_2048": self.reached_2048,
        }

    def _spawn_tile(self) -> tuple[int, tuple[int, int] | None]:
        empty = list(zip(*np.where(self.board == 0)))
        if not empty:
            return 0, None
        row, col = empty[self.rng.integers(len(empty))]
        value = 2 if self.rng.random() < TILE_TWO_PROB else 4
        self.board[row, col] = value
        return value, (int(row), int(col))

    def step(self, action: int) -> StepResult:
        if not 0 <= action < NUM_ACTIONS:
            raise ValueError(f"Invalid action: {action}")

        new_board, merge_score, changed = move_board(self.board, action)
        spawned_value = 0
        spawn_position: tuple[int, int] | None = None

        if changed:
            self.board = new_board
            self.game_score += merge_score
            if max_tile_value(self.board) >= TARGET_TILE:
                self.reached_2048 = True
            spawned_value, spawn_position = self._spawn_tile()

        self.step_count += 1

        return StepResult(
            board=self.board.copy(),
            merge_score=merge_score,
            changed=changed,
            spawned_value=spawned_value,
            spawn_position=spawn_position,
        )

    def is_terminated(self) -> bool:
        return not valid_action_mask(self.board).any()

    @classmethod
    def from_board(
        cls,
        board: Iterable[Iterable[int]],
        *,
        rng: np.random.Generator | None = None,
        game_score: int = 0,
        step_count: int = 0,
        reached_2048: bool | None = None,
    ) -> Board2048:
        """Construct a board from a fixed layout (for tests). Skips initial spawns."""
        game = cls.__new__(Board2048)
        game.rng = rng if rng is not None else np.random.default_rng()
        game.board = np.asarray(board, dtype=np.int32).reshape(BOARD_SIZE, BOARD_SIZE)
        game.game_score = game_score
        game.step_count = step_count
        if reached_2048 is None:
            game.reached_2048 = max_tile_value(game.board) >= TARGET_TILE
        else:
            game.reached_2048 = reached_2048
        return game
