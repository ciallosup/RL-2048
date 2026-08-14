"""DQN policy loaded from checkpoint for eval/viz.

Default inference is 2-ply expectimax. Training/eval of the raw Q-network
should pass ``decode="greedy"``.
"""

from __future__ import annotations

import os
from pathlib import Path

from rl2048.policies.base import PolicyContext
from rl2048.rl.checkpoint import load_checkpoint
from rl2048.rl.config import resolve_device
from rl2048.rl.masking import numpy_masked_argmax
from rl2048.rl.search import expectimax_select_action

DECODE_GREEDY = "greedy"
DECODE_1PLY = "1ply"
DECODE_2PLY = "2ply"
DECODE_MODES = (DECODE_GREEDY, DECODE_1PLY, DECODE_2PLY)
DECODE_LABELS = {
    DECODE_GREEDY: "DQN 贪心 Q",
    DECODE_1PLY: "DQN 1-ply",
    DECODE_2PLY: "DQN 2-ply",
}


def normalize_decode(decode: str) -> str:
    key = str(decode).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "greedy": DECODE_GREEDY,
        "q": DECODE_GREEDY,
        "0": DECODE_GREEDY,
        "1ply": DECODE_1PLY,
        "1": DECODE_1PLY,
        "2ply": DECODE_2PLY,
        "2": DECODE_2PLY,
    }
    if key not in aliases:
        raise ValueError(f"Unknown decode mode {decode!r}; use greedy, 1ply, or 2ply.")
    return aliases[key]


class DQNPolicy:
    key = "dqn"
    label = DECODE_LABELS[DECODE_2PLY]

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        decode: str = DECODE_2PLY,
        include_reward: bool = True,
        adaptive: bool = False,
        corner_tiebreak: bool | None = None,
        corner_margin: float = 2.0,
    ) -> None:
        path = checkpoint_path or os.environ.get("RL2048_CHECKPOINT")
        if not path:
            raise ValueError(
                "DQNPolicy requires checkpoint_path or RL2048_CHECKPOINT environment variable."
            )
        device = resolve_device("auto")
        self.checkpoint_path = str(Path(path))
        self.agent, self.config, self.meta = load_checkpoint(Path(path), device=device)
        self.agent.eval_mode()
        self.obs_scale = self.config.obs_scale
        self.gamma = float(self.config.gamma)
        self.reward_mode = str(self.config.reward_mode)
        self.include_reward = include_reward
        self.adaptive = bool(adaptive)
        self.corner_margin = float(corner_margin)
        self._corner_tiebreak_override = corner_tiebreak
        self.set_decode(decode)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        decode: str = DECODE_2PLY,
        include_reward: bool = True,
        adaptive: bool = False,
        corner_tiebreak: bool | None = None,
        corner_margin: float = 2.0,
    ) -> DQNPolicy:
        return cls(
            checkpoint_path=checkpoint_path,
            decode=decode,
            include_reward=include_reward,
            adaptive=adaptive,
            corner_tiebreak=corner_tiebreak,
            corner_margin=corner_margin,
        )

    def set_decode(self, decode: str) -> None:
        self.decode = normalize_decode(decode)
        self.label = DECODE_LABELS[self.decode]

    @property
    def corner_tiebreak(self) -> bool:
        if self._corner_tiebreak_override is not None:
            return bool(self._corner_tiebreak_override)
        return self.decode == DECODE_2PLY

    def reset(self, ctx: PolicyContext) -> None:
        return None

    def select_action(self, ctx: PolicyContext) -> int:
        if self.decode == DECODE_GREEDY:
            mask = ctx.info["valid_action_mask"]
            assert ctx.obs is not None
            q_values = self.agent.q_values(ctx.obs, obs_scale=self.obs_scale)
            return numpy_masked_argmax(q_values, mask)

        def q_batch_fn(obs_batch):
            return self.agent.q_values_batch(obs_batch, obs_scale=self.obs_scale)

        return expectimax_select_action(
            ctx.env.board,
            q_batch_fn,
            gamma=self.gamma,
            reward_mode=self.reward_mode,
            include_reward=self.include_reward,
            depth=1 if self.decode == DECODE_1PLY else 2,
            adaptive=self.adaptive,
            corner_tiebreak=self.corner_tiebreak,
            corner_margin=self.corner_margin,
        )
