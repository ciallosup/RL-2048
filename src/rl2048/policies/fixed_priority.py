from __future__ import annotations

import numpy as np

from rl2048.core import ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP
from rl2048.policies.base import PolicyContext


class FixedPriorityPolicy:
    key = "fixed"
    label = "固定优先级 (左→下→右→上)"

    _order = (ACTION_LEFT, ACTION_DOWN, ACTION_RIGHT, ACTION_UP)

    def reset(self, ctx: PolicyContext) -> None:
        return None

    def select_action(self, ctx: PolicyContext) -> int:
        mask = ctx.info["valid_action_mask"]
        for action in self._order:
            if mask[action]:
                return action
        raise RuntimeError("No legal action available")
