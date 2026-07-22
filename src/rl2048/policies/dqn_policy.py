"""DQN policy loaded from checkpoint for eval/viz."""

from __future__ import annotations

import os
from pathlib import Path

from rl2048.policies.base import PolicyContext
from rl2048.rl.checkpoint import load_checkpoint
from rl2048.rl.config import resolve_device
from rl2048.rl.masking import numpy_masked_argmax


class DQNPolicy:
    key = "dqn"
    label = "DQN (checkpoint)"

    def __init__(self, checkpoint_path: str | Path | None = None) -> None:
        path = checkpoint_path or os.environ.get("RL2048_CHECKPOINT")
        if not path:
            raise ValueError(
                "DQNPolicy requires checkpoint_path or RL2048_CHECKPOINT environment variable."
            )
        device = resolve_device("auto")
        self.agent, self.config, self.meta = load_checkpoint(Path(path), device=device)
        self.agent.eval_mode()
        self.obs_scale = self.config.obs_scale

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path) -> DQNPolicy:
        return cls(checkpoint_path=checkpoint_path)

    def reset(self, ctx: PolicyContext) -> None:
        return None

    def select_action(self, ctx: PolicyContext) -> int:
        mask = ctx.info["valid_action_mask"]
        assert ctx.obs is not None
        q_values = self.agent.q_values(ctx.obs, obs_scale=self.obs_scale)
        return numpy_masked_argmax(q_values, mask)
