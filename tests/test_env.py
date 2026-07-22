"""Tests for Gymnasium environment wrapper."""

import numpy as np
import pytest

from rl2048.core import ACTION_LEFT, ACTION_UP
from rl2048.env import Game2048Env


class TestGame2048Env:
    def test_reset_step_shapes(self):
        env = Game2048Env()
        obs, info = env.reset(seed=0)
        assert obs.shape == (16,)
        assert set(info.keys()) >= {
            "valid_action_mask",
            "merge_score",
            "game_score",
            "max_tile",
            "episode_length",
            "reached_2048",
        }
        assert info["valid_action_mask"].shape == (4,)

    def test_truncation_without_termination(self):
        env = Game2048Env(max_episode_steps=3)
        env.reset(seed=1)
        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            mask = env._game.info()["valid_action_mask"]
            action = int(np.argmax(mask))
            _, _, terminated, truncated, _ = env.step(action)
            steps += 1
        assert truncated
        assert not terminated
        assert steps == 3

    def test_invalid_action_zero_reward(self):
        env = Game2048Env()
        env.reset(seed=0)
        # Build a locked row on the left edge if possible; fallback: use LEFT on stable board
        env._game = env._game.from_board(
            [
                [2, 4, 8, 16],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ]
        )
        obs, reward, terminated, truncated, info = env.step(ACTION_LEFT)
        assert reward == 0.0
        assert info["merge_score"] == 0
        assert info["action_changed_board"] is False
        assert not terminated
        assert not truncated

    def test_valid_move_positive_or_zero_reward(self):
        env = Game2048Env()
        env.reset(seed=0)
        mask = env._game.info()["valid_action_mask"]
        action = int(np.argmax(mask))
        _, reward, _, _, info = env.step(action)
        assert reward == float(info["merge_score"])
        assert reward >= 0.0

    def test_info_tracks_scores(self):
        env = Game2048Env()
        env.reset(seed=5)
        env._game = env._game.from_board(
            [
                [2, 2, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            rng=env._game.rng,
        )
        _, reward, _, _, info = env.step(ACTION_LEFT)
        assert reward == 4.0
        assert info["game_score"] == 4
        assert info["max_tile"] == 4
