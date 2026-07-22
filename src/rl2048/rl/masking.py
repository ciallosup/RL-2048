"""Masked action selection for behavior policy and TD targets."""

from __future__ import annotations

import numpy as np
import torch


def masked_argmax(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Deterministic masked argmax; ties broken by smallest action index.

    q_values: (batch, num_actions)
    mask: (batch, num_actions) bool
    """
    neg_inf = torch.finfo(q_values.dtype).min
    masked = q_values.masked_fill(~mask, neg_inf)
    return masked.argmax(dim=1)


def masked_random_actions(
    mask: torch.Tensor,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample uniformly among legal actions per row."""
    batch_size, num_actions = mask.shape
    actions = torch.empty(batch_size, dtype=torch.long, device=mask.device)
    probs = mask.float()
    for i in range(batch_size):
        legal = torch.nonzero(probs[i], as_tuple=False).squeeze(1)
        idx = torch.randint(len(legal), (1,), generator=generator, device=mask.device)
        actions[i] = legal[idx]
    return actions


def select_epsilon_greedy_actions(
    q_values: torch.Tensor,
    mask: torch.Tensor,
    epsilon: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (actions, is_random) with masked epsilon-greedy."""
    batch_size = q_values.shape[0]
    greedy = masked_argmax(q_values, mask)
    if epsilon <= 0.0:
        return greedy, torch.zeros(batch_size, dtype=torch.bool, device=q_values.device)
    random_flags = torch.rand(batch_size, generator=generator, device=q_values.device) < epsilon
    if not random_flags.any():
        return greedy, random_flags
    random_actions = masked_random_actions(mask, generator=generator)
    actions = torch.where(random_flags, random_actions, greedy)
    return actions, random_flags


def numpy_masked_argmax(q_values: np.ndarray, mask: np.ndarray) -> int:
    neg_inf = np.finfo(np.float32).min
    masked = np.where(mask, q_values, neg_inf)
    max_q = masked.max()
    candidates = np.flatnonzero(masked == max_q)
    return int(candidates[0])
