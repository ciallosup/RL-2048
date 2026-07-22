"""Unit tests for core slide/merge/spawn rules (roadmap section 3.2)."""

import numpy as np
import pytest

from rl2048.core import (
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_UP,
    TILE_FOUR_PROB,
    TILE_TWO_PROB,
    Board2048,
    board_to_exponents,
    exponents_to_observation,
    move_board,
    tile_to_exponent,
    valid_action_mask,
)


class TestEncoding:
    def test_tile_to_exponent(self):
        assert tile_to_exponent(0) == 0
        assert tile_to_exponent(2) == 1
        assert tile_to_exponent(4) == 2
        assert tile_to_exponent(2048) == 11

    def test_observation_shape(self):
        board = np.array(
            [[2, 0, 0, 0], [0, 4, 0, 0], [0, 0, 8, 0], [0, 0, 0, 16]],
            dtype=np.int32,
        )
        obs = exponents_to_observation(board)
        assert obs.shape == (16,)
        assert obs.tolist() == [1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, 4]


class TestSlideRules:
    def test_no_merge_shift_left(self):
        board = Board2048.from_board(
            [
                [0, 2, 0, 4],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ).board
        result, score, changed = move_board(board, ACTION_LEFT)
        assert changed
        assert score == 0
        assert result[0].tolist() == [2, 4, 0, 0]

    def test_single_merge(self):
        board = Board2048.from_board(
            [
                [2, 2, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ).board
        result, score, changed = move_board(board, ACTION_LEFT)
        assert changed
        assert score == 4
        assert result[0].tolist() == [4, 0, 0, 0]

    def test_quad_merge_once_per_tile(self):
        board = Board2048.from_board(
            [
                [2, 2, 2, 2],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ).board
        result, score, changed = move_board(board, ACTION_LEFT)
        assert changed
        assert score == 8
        assert result[0].tolist() == [4, 4, 0, 0]

    def test_gap_between_equal_tiles(self):
        board = Board2048.from_board(
            [
                [2, 0, 2, 2],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ).board
        result, score, changed = move_board(board, ACTION_LEFT)
        assert changed
        assert score == 4
        assert result[0].tolist() == [4, 2, 0, 0]

    def test_boundary_move_up(self):
        board = Board2048.from_board(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 2, 2],
                [0, 0, 2, 4],
            ]
        ).board
        result, score, changed = move_board(board, ACTION_UP)
        assert changed
        assert score == 4
        assert result[:, 2].tolist() == [4, 0, 0, 0]
        assert result[:, 3].tolist() == [2, 4, 0, 0]

    def test_no_change_invalid_direction(self):
        board = Board2048.from_board(
            [
                [2, 4, 8, 16],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        ).board
        result, score, changed = move_board(board, ACTION_LEFT)
        assert not changed
        assert score == 0
        assert np.array_equal(result, board)


class TestInvalidActions:
    def test_invalid_action_no_spawn(self):
        game = Board2048.from_board(
            [
                [2, 4, 8, 16],
                [32, 64, 128, 256],
                [2, 4, 8, 16],
                [32, 64, 128, 256],
            ],
            rng=np.random.default_rng(0),
        )
        board_before = game.board.copy()
        result = game.step(ACTION_LEFT)
        assert not result.changed
        assert result.merge_score == 0
        assert result.spawned_value == 0
        assert result.spawn_position is None
        assert np.array_equal(game.board, board_before)

    def test_invalid_action_still_counts_step(self):
        game = Board2048.from_board(
            [
                [2, 4, 8, 16],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        assert game.step_count == 0
        game.step(ACTION_LEFT)
        assert game.step_count == 1


class TestSpawnDistribution:
    def test_spawn_position_uniform(self):
        rng = np.random.default_rng(123)
        counts = np.zeros((4, 4), dtype=np.int64)
        trials = 30_000
        for _ in range(trials):
            game = Board2048.from_board([[0, 0, 0, 0]] * 4, rng=rng)
            _, pos = game._spawn_tile()
            assert pos is not None
            counts[pos] += 1

        expected = trials / 16
        for cell_count in counts.flatten():
            assert abs(cell_count - expected) / expected < 0.12

    def test_spawn_value_ratio(self):
        rng = np.random.default_rng(456)
        twos = 0
        fours = 0
        trials = 50_000
        for _ in range(trials):
            game = Board2048.from_board(
                [[0, 0, 0, 0]] * 4,
                rng=rng,
            )
            value, _ = game._spawn_tile()
            if value == 2:
                twos += 1
            elif value == 4:
                fours += 1
        ratio = twos / (twos + fours)
        assert abs(ratio - TILE_TWO_PROB) < 0.02
        assert abs(fours / (twos + fours) - TILE_FOUR_PROB) < 0.02


class TestTermination:
    def test_full_board_mergeable_not_terminated(self):
        game = Board2048.from_board(
            [
                [2, 2, 4, 8],
                [4, 8, 16, 32],
                [8, 16, 32, 64],
                [16, 32, 64, 128],
            ]
        )
        assert not game.is_terminated()
        assert valid_action_mask(game.board).any()

    def test_full_board_no_moves_terminated(self):
        game = Board2048.from_board(
            [
                [2, 4, 2, 4],
                [4, 2, 4, 8],
                [2, 4, 2, 4],
                [4, 2, 4, 8],
            ]
        )
        assert game.is_terminated()
        assert not valid_action_mask(game.board).any()


class TestSeedReproducibility:
    def test_same_seed_same_trajectory(self):
        def run(seed: int) -> list[tuple[list[int], float]]:
            rng = np.random.default_rng(seed)
            game = Board2048(rng=rng)
            trace = [(game.observation().tolist(), float(game.game_score))]
            actions = [ACTION_LEFT, ACTION_UP, ACTION_RIGHT, ACTION_DOWN, ACTION_LEFT]
            for action in actions:
                game.step(action)
                trace.append((game.observation().tolist(), float(game.game_score)))
            return trace

        a = run(42)
        b = run(42)
        assert a == b
        assert run(42) != run(43)


class TestReached2048:
    def test_reaching_2048_does_not_auto_terminate(self):
        game = Board2048.from_board(
            [
                [1024, 1024, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            rng=np.random.default_rng(0),
        )
        game.step(ACTION_LEFT)
        assert game.reached_2048
        assert not game.is_terminated()

    def test_reached_flag_sticky(self):
        game = Board2048.from_board(
            [
                [2048, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        assert game.reached_2048
