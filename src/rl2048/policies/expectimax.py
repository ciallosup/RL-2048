"""Expectimax policy wrapping a DQN checkpoint (thin decode alias)."""

from __future__ import annotations

from pathlib import Path

from rl2048.policies.dqn_policy import DECODE_1PLY, DECODE_2PLY, DECODE_3PLY, DQNPolicy


class ExpectimaxDQNPolicy(DQNPolicy):
    key = "dqn_expectimax"
    label = "DQN expectimax"

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        *,
        include_reward: bool = True,
        depth: int = 2,
        adaptive: bool = False,
        corner_tiebreak: bool = True,
        corner_margin: float = 2.0,
    ) -> None:
        requested = max(1, int(depth))
        if requested <= 1:
            decode = DECODE_1PLY
        elif requested == 2:
            decode = DECODE_2PLY
        else:
            decode = DECODE_3PLY
        super().__init__(
            checkpoint_path=checkpoint_path,
            decode=decode,
            include_reward=include_reward,
            adaptive=adaptive,
            corner_tiebreak=corner_tiebreak,
            corner_margin=corner_margin,
        )
        self.depth = requested
        self.label = (
            f"DQN expectimax d{self.depth}"
            f"{' adapt' if self.adaptive else ''}"
            f"{' corner' if self.corner_tiebreak else ''}"
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        *,
        include_reward: bool = True,
        depth: int = 2,
        adaptive: bool = False,
        corner_tiebreak: bool = True,
        corner_margin: float = 2.0,
    ) -> ExpectimaxDQNPolicy:
        return cls(
            checkpoint_path=checkpoint_path,
            include_reward=include_reward,
            depth=depth,
            adaptive=adaptive,
            corner_tiebreak=corner_tiebreak,
            corner_margin=corner_margin,
        )
