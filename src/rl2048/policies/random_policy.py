from __future__ import annotations

import numpy as np

from rl2048.policies.base import PolicyContext


class RandomPolicy:
    key = "random"
    label = "随机 (合法动作均匀采样)"

    def reset(self, ctx: PolicyContext) -> None:
        return None

    def select_action(self, ctx: PolicyContext) -> int:
        mask = ctx.info["valid_action_mask"]
        legal = np.flatnonzero(mask)
        return int(ctx.env.np_random.choice(legal))
