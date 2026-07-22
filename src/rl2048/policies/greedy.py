from __future__ import annotations

import numpy as np

from rl2048.core import move_board
from rl2048.policies.base import PolicyContext


class GreedyMergePolicy:
    """Oracle one-step greedy: pick the legal action with highest immediate merge score."""

    key = "greedy"
    label = "贪心 (一步合并分最高)"

    def reset(self, ctx: PolicyContext) -> None:
        return None

    def select_action(self, ctx: PolicyContext) -> int:
        board = ctx.env.board
        mask = ctx.info["valid_action_mask"]
        best_action = int(np.flatnonzero(mask)[0])
        best_score = -1
        for action in np.flatnonzero(mask):
            _, merge_score, _ = move_board(board, int(action))
            if merge_score > best_score:
                best_score = merge_score
                best_action = int(action)
        return best_action
