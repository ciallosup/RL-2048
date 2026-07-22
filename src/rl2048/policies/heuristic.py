from __future__ import annotations

import numpy as np

from rl2048.core import move_board
from rl2048.heuristic.features import board_heuristic_score
from rl2048.policies.base import PolicyContext


class HeuristicPolicy:
    """
    Simple weighted heuristic baseline.

    Scores each legal move by evaluating the board *after* the slide/merge
    (before random spawn), using empty cells, monotonicity, smoothness,
    and corner-max preference.
    """

    key = "heuristic"
    label = "启发式 (空格/单调/平滑/角落)"

    def reset(self, ctx: PolicyContext) -> None:
        return None

    def select_action(self, ctx: PolicyContext) -> int:
        board = ctx.env.board
        mask = ctx.info["valid_action_mask"]
        legal = np.flatnonzero(mask)

        best_action = int(legal[0])
        best_score = float("-inf")
        best_merge = -1

        for action in legal:
            new_board, merge_score, _ = move_board(board, int(action))
            score = board_heuristic_score(new_board, merge_score=merge_score)
            if score > best_score or (score == best_score and merge_score > best_merge):
                best_score = score
                best_merge = merge_score
                best_action = int(action)

        return best_action
