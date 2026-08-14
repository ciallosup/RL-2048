"""Uniform and prioritized replay buffers with n-step assembly."""

from __future__ import annotations

from collections import deque
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
    # Multiplier for bootstrapped next-state value: gamma^k for k-step return.
    bootstrap_discount: float = 1.0


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
    bootstrap_discounts: np.ndarray
    indices: np.ndarray
    ages: np.ndarray
    weights: np.ndarray | None = None


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
        self.bootstrap_discounts = np.ones((capacity,), dtype=np.float32)

    def __len__(self) -> int:
        return self._size

    @property
    def total_added(self) -> int:
        return self._total_added

    def _write(self, idx: int, transition: Transition) -> None:
        self.states[idx] = transition.state
        self.actions[idx] = transition.action
        self.rewards[idx] = transition.reward
        self.next_states[idx] = transition.next_state
        self.terminated[idx] = transition.terminated
        self.truncated[idx] = transition.truncated
        self.valid_masks[idx] = transition.valid_mask
        self.next_valid_masks[idx] = transition.next_valid_mask
        self.bootstrap_discounts[idx] = transition.bootstrap_discount

    def push(self, transition: Transition) -> int:
        idx = self._pos
        self._write(idx, transition)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)
        self._total_added += 1
        return idx

    def _make_batch(self, indices: np.ndarray, weights: np.ndarray | None = None) -> Batch:
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
            bootstrap_discounts=self.bootstrap_discounts[indices],
            indices=indices,
            ages=ages.astype(np.int64),
            weights=weights,
        )

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        indices = rng.integers(0, self._size, size=batch_size)
        return self._make_batch(indices)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        return None


class SumTree:
    """Binary sum tree for proportional prioritization."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data_idx: int) -> None:
        tree_idx = data_idx + self.capacity - 1
        self.update(tree_idx, priority)

    def update(self, tree_idx: int, priority: float) -> None:
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get(self, value: float) -> tuple[int, float, int]:
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                leaf = parent
                break
            if value <= self.tree[left]:
                parent = left
            else:
                value -= self.tree[left]
                parent = right
        data_idx = leaf - (self.capacity - 1)
        return leaf, float(self.tree[leaf]), data_idx


class PrioritizedReplayBuffer(ReplayBuffer):
    """Proportional PER with importance-sampling weights."""

    def __init__(
        self,
        capacity: int,
        obs_dim: int = 16,
        num_actions: int = 4,
        *,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 1_000_000,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(capacity, obs_dim=obs_dim, num_actions=num_actions)
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = max(beta_frames, 1)
        self.eps = eps
        self.tree = SumTree(capacity)
        self.max_priority = 1.0
        self._frame = 0

    def push(self, transition: Transition) -> int:
        idx = super().push(transition)
        priority = self.max_priority**self.alpha
        tree_idx = idx + self.capacity - 1
        self.tree.update(tree_idx, priority)
        return idx

    def _beta(self) -> float:
        fraction = min(self._frame / self.beta_frames, 1.0)
        return self.beta_start + fraction * (1.0 - self.beta_start)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        if self._size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        self._frame += 1
        indices = np.empty(batch_size, dtype=np.int64)
        priorities = np.empty(batch_size, dtype=np.float64)
        total = self.tree.total
        if total <= 0:
            indices = rng.integers(0, self._size, size=batch_size)
            weights = np.ones(batch_size, dtype=np.float32)
            return self._make_batch(indices, weights)

        segment = total / batch_size
        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)
            value = rng.uniform(low, high)
            _, priority, data_idx = self.tree.get(value)
            # Guard against empty slots beyond current size.
            data_idx = int(np.clip(data_idx, 0, self._size - 1))
            indices[i] = data_idx
            priorities[i] = max(priority, self.eps)

        probs = priorities / total
        beta = self._beta()
        weights = (self._size * probs) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)
        return self._make_batch(indices, weights)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        for idx, priority in zip(indices, priorities, strict=False):
            p = float(abs(priority) + self.eps)
            self.max_priority = max(self.max_priority, p)
            tree_idx = int(idx) + self.capacity - 1
            self.tree.update(tree_idx, p**self.alpha)


@dataclass
class _PendingStep:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    terminated: bool
    truncated: bool
    valid_mask: np.ndarray
    next_valid_mask: np.ndarray


class NStepAssembler:
    """
    Assemble n-step transitions for the replay buffer.

    - terminated cuts the return and zeroes bootstrap.
    - truncated cuts the window early but keeps bootstrap from the truncated next state.
    """

    def __init__(self, n_step: int, gamma: float) -> None:
        if n_step < 1:
            raise ValueError("n_step must be >= 1")
        self.n_step = n_step
        self.gamma = gamma
        self._pending: deque[_PendingStep] = deque()

    def push(self, transition: Transition) -> list[Transition]:
        if self.n_step == 1:
            return [
                Transition(
                    state=transition.state,
                    action=transition.action,
                    reward=transition.reward,
                    next_state=transition.next_state,
                    terminated=transition.terminated,
                    truncated=transition.truncated,
                    valid_mask=transition.valid_mask,
                    next_valid_mask=transition.next_valid_mask,
                    bootstrap_discount=self.gamma,
                )
            ]

        self._pending.append(
            _PendingStep(
                state=transition.state,
                action=transition.action,
                reward=float(transition.reward),
                next_state=transition.next_state,
                terminated=bool(transition.terminated),
                truncated=bool(transition.truncated),
                valid_mask=transition.valid_mask,
                next_valid_mask=transition.next_valid_mask,
            )
        )
        emitted: list[Transition] = []
        done = transition.terminated or transition.truncated
        while self._pending and (len(self._pending) >= self.n_step or done):
            if not done and len(self._pending) < self.n_step:
                break
            emitted.append(self._emit_first())
            if done and not self._pending:
                break
            if done and self._pending:
                continue
        return emitted

    def flush(self) -> list[Transition]:
        emitted: list[Transition] = []
        while self._pending:
            emitted.append(self._emit_first())
        return emitted

    def _emit_first(self) -> Transition:
        first = self._pending[0]
        n_return = 0.0
        discount = 1.0
        next_state = first.next_state
        next_mask = first.next_valid_mask
        terminated = False
        truncated = False
        steps = 0

        for step in list(self._pending):
            n_return += discount * step.reward
            next_state = step.next_state
            next_mask = step.next_valid_mask
            steps += 1
            if step.terminated:
                terminated = True
                truncated = False
                break
            if step.truncated:
                truncated = True
                terminated = False
                break
            discount *= self.gamma
            if steps >= self.n_step:
                break

        bootstrap_discount = 0.0 if terminated else (self.gamma**steps)
        self._pending.popleft()
        return Transition(
            state=first.state,
            action=first.action,
            reward=n_return,
            next_state=next_state,
            terminated=terminated,
            truncated=truncated,
            valid_mask=first.valid_mask,
            next_valid_mask=next_mask,
            bootstrap_discount=bootstrap_discount,
        )
