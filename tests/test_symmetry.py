"""Symmetry equivariance tests (roadmap section 3.2)."""

import numpy as np
import pytest

from rl2048.core import Board2048, move_board, valid_action_mask
from rl2048.symmetry import all_transform_ids, transform_action, transform_board


class TestSymmetryEquivariance:
    @pytest.mark.parametrize("transform_id", list(all_transform_ids()))
    @pytest.mark.parametrize(
        "board",
        [
            [
                [2, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 2, 0],
                [0, 0, 0, 2],
            ],
            [
                [2, 2, 4, 8],
                [0, 0, 2, 2],
                [4, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            [
                [2, 4, 8, 16],
                [4, 8, 16, 32],
                [8, 16, 32, 64],
                [16, 32, 64, 128],
            ],
        ],
    )
    @pytest.mark.parametrize("action", range(4))
    def test_move_equivariance(self, transform_id, board, action):
        board_arr = np.asarray(board, dtype=np.int32)
        t_board = transform_board(board_arr, transform_id)
        t_action = transform_action(action, transform_id)

        moved, score, _ = move_board(board_arr, action)
        t_moved, t_score, _ = move_board(t_board, t_action)

        expected = transform_board(moved, transform_id)
        assert t_score == score
        assert np.array_equal(t_moved, expected)

    @pytest.mark.parametrize("transform_id", list(all_transform_ids()))
    def test_valid_mask_equivariance(self, transform_id):
        game = Board2048.from_board(
            [
                [2, 0, 4, 0],
                [0, 2, 0, 4],
                [4, 0, 2, 0],
                [0, 4, 0, 2],
            ]
        )
        board = game.board
        mask = valid_action_mask(board)
        t_board = transform_board(board, transform_id)
        t_mask = valid_action_mask(t_board)

        expected = np.array(
            [transform_action(a, transform_id) for a in range(4) if mask[a]],
            dtype=int,
        )
        actual = np.array([a for a in range(4) if t_mask[a]], dtype=int)
        assert set(actual.tolist()) == set(expected.tolist())

    def test_rotation_action_mapping_example(self):
        # rotation=1: board rot90 CCW once -> up becomes left
        assert transform_action(0, 1) == 2  # up -> left
        assert transform_action(2, 1) == 1  # left -> down
        assert transform_action(1, 1) == 3  # down -> right
        assert transform_action(3, 1) == 0  # right -> up
        # rotation=3: rot90 CCW three times (= clockwise 90°) -> up becomes right
        assert transform_action(0, 3) == 3  # up -> right
