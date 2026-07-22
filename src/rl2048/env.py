"""Gymnasium-compatible 2048 environment."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl2048.core import (
    ACTION_NAMES,
    Board2048,
    NUM_ACTIONS,
    exponents_to_observation,
    valid_action_mask,
)


class Game2048Env(gym.Env):
    """
    Standard 2048 RL environment (roadmap sections 2–3).

    - Reward: merge score for the step (0 on invalid/no-merge moves).
    - terminated: no legal move changes the board.
    - truncated: step_count >= max_episode_steps (default None -> no truncation).
    - Reaching 2048 does not terminate; tracked in info['reached_2048'].
    """

    metadata = {"render_modes": []}

    def __init__(self, max_episode_steps: int | None = None) -> None:
        super().__init__()
        self.max_episode_steps = max_episode_steps
        self.observation_space = spaces.Box(
            low=0,
            high=17,  # 2^17 = 131072 upper exponent bound for 4x4
            shape=(16,),
            dtype=np.int32,
        )
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self._game: Board2048 | None = None
        self._last_merge_score = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._game = Board2048(rng=self.np_random)
        self._last_merge_score = 0
        return self._game.observation(), self._build_info(merge_score=0)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self._game is None:
            raise RuntimeError("Call reset() before step().")

        result = self._game.step(int(action))
        self._last_merge_score = result.merge_score

        terminated = self._game.is_terminated()
        truncated = (
            self.max_episode_steps is not None
            and self._game.step_count >= self.max_episode_steps
        )

        reward = float(result.merge_score)
        info = self._build_info(
            merge_score=result.merge_score,
            changed=result.changed,
            spawned_value=result.spawned_value,
            spawn_position=result.spawn_position,
        )
        return self._game.observation(), reward, terminated, truncated, info

    def _build_info(
        self,
        *,
        merge_score: int,
        changed: bool | None = None,
        spawned_value: int = 0,
        spawn_position: tuple[int, int] | None = None,
    ) -> dict[str, Any]:
        assert self._game is not None
        info = self._game.info()
        info["merge_score"] = merge_score
        info["valid_action_mask"] = valid_action_mask(self._game.board)
        if changed is not None:
            info["action_changed_board"] = changed
        if spawned_value:
            info["spawned_value"] = spawned_value
        if spawn_position is not None:
            info["spawn_position"] = spawn_position
        return info

    @property
    def board(self) -> np.ndarray:
        if self._game is None:
            raise RuntimeError("Call reset() before accessing board.")
        return self._game.board.copy()

    def render(self) -> None:
        if self._game is None:
            return
        board = self._game.board
        lines = ["+----" * 4 + "+"]
        for row in board:
            cells = "|".join(f"{v:4d}" if v else "    " for v in row)
            lines.append(f"|{cells}|")
            lines.append("+----" * 4 + "+")
        print("\n".join(lines))

    @staticmethod
    def action_name(action: int) -> str:
        return ACTION_NAMES[action]
