"""Reward transforms for training (eval still uses raw game score)."""

from __future__ import annotations

import math


def transform_reward(merge_score: float, mode: str = "raw") -> float:
    if mode == "raw":
        return float(merge_score)
    if mode == "log1p":
        return math.log1p(max(float(merge_score), 0.0))
    raise ValueError(f"Unknown reward_mode: {mode}")
