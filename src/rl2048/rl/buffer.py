"""Uniform replay buffer with action masks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool
    valid_mask: np.ndarray
    next_valid_mask: np.ndarray


@dataclass
class Batch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    valid_masks: np.ndarray
    next_valid_masks: np.ndarray
    indices: np.ndarray
    ages: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int = 16, num_actions: int = 4) -> None:
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.num_actions = num_actions
        self._pos = 0
        self._size = 0
        self._total_added = 0

        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.int64)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.terminated = np.zeros((capacity,), dtype=np.bool_)
        self.truncated = np.zeros((capacity,), dtype=np.bool_)
        self.valid_masks = np.zeros((capacity, num_actions), dtype=np.bool_)
        self.next_valid_masks = np.zeros((capacity, num_actions), dtype=np.bool_)

    def __len__(self) -> int:
        return self._size

    @property
    def total_added(self) -> int:
        return self._total_added

    def push(self, transition: Transition) -> None:
        idx = self._pos
        self.states[idx] = transition.state
        self.actions[idx] = transition.action
        self.rewards[idx] = transition.reward
        self.next_states[idx] = transition.next_state
        self.terminated[idx] = transition.terminated
        self.truncated[idx] = transition.truncated
        self.valid_masks[idx] = transition.valid_mask
        self.next_valid_masks[idx] = transition.next_valid_mask

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._total_added += 1

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        indices = rng.integers(0, self._size, size=batch_size)
        ages = self._total_added - indices
        return Batch(
            states=self.states[indices],
            actions=self.actions[indices],
            rewards=self.rewards[indices],
            next_states=self.next_states[indices],
            terminated=self.terminated[indices],
            truncated=self.truncated[indices],
            valid_masks=self.valid_masks[indices],
            next_valid_masks=self.next_valid_masks[indices],
            indices=indices,
            ages=ages.astype(np.int64),
        )
