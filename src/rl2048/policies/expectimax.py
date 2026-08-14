"""1-ply expectimax policy wrapping a DQN checkpoint."""

from __future__ import annotations

import os
from pathlib import Path

from rl2048.policies.base import PolicyContext
from rl2048.rl.checkpoint import load_checkpoint
from rl2048.rl.config import resolve_device
from rl2048.rl.search import expectimax_select_action


class ExpectimaxDQNPolicy:
    key = "dqn_expectimax"
    label = "DQN 1-ply expectimax"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        include_reward: bool = True,
    ) -> None:
        path = checkpoint_path or os.environ.get("RL2048_CHECKPOINT")
        if not path:
            raise ValueError(
                "ExpectimaxDQNPolicy requires checkpoint_path or RL2048_CHECKPOINT."
            )
        device = resolve_device("auto")
        self.agent, self.config, self.meta = load_checkpoint(Path(path), device=device)
        self.agent.eval_mode()
        self.obs_scale = self.config.obs_scale
        self.gamma = float(self.config.gamma)
        self.reward_mode = str(self.config.reward_mode)
        self.include_reward = include_reward

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        include_reward: bool = True,
    ) -> ExpectimaxDQNPolicy:
        return cls(checkpoint_path=checkpoint_path, include_reward=include_reward)

    def reset(self, ctx: PolicyContext) -> None:
        return None

    def select_action(self, ctx: PolicyContext) -> int:
        board = ctx.env.board

        def q_batch_fn(obs_batch):
            return self.agent.q_values_batch(obs_batch, obs_scale=self.obs_scale)

        return expectimax_select_action(
            board,
            q_batch_fn,
            gamma=self.gamma,
            reward_mode=self.reward_mode,
            include_reward=self.include_reward,
        )
