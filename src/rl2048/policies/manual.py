from __future__ import annotations

from rl2048.policies.base import PolicyContext


class ManualPolicy:
    key = "manual"
    label = "人工 (方向键 / WASD)"

    def __init__(self) -> None:
        self._pending_action: int | None = None

    def reset(self, ctx: PolicyContext) -> None:
        self._pending_action = None

    def queue_action(self, action: int) -> None:
        self._pending_action = action

    def select_action(self, ctx: PolicyContext) -> int | None:
        if self._pending_action is None:
            return None
        action = self._pending_action
        self._pending_action = None
        mask = ctx.info["valid_action_mask"]
        if not mask[action]:
            return None
        return action
