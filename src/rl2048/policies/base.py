"""Policy interface for 2048 agents (random, greedy, RL, ...)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from rl2048.env import Game2048Env


@dataclass
class PolicyContext:
    """Mutable per-episode state shared between env and policy."""

    env: Game2048Env
    obs: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)
    done: bool = False


class Policy(Protocol):
    """Select the next action given the current environment state."""

    key: str
    label: str

    def reset(self, ctx: PolicyContext) -> None: ...

    def select_action(self, ctx: PolicyContext) -> int | None:
        """
        Return an action index, or None if the policy cannot decide yet
        (e.g. manual control waiting for keyboard input).
        """
        ...
