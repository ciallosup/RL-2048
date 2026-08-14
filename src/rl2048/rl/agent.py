"""Masked DQN / Double DQN agent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from rl2048.rl.buffer import Batch
from rl2048.rl.encoding import encode_onehot, encode_onehot_torch, encode_scaled_flat
from rl2048.rl.masking import masked_argmax, select_epsilon_greedy_actions
from rl2048.rl.network import build_q_network, clone_network, hard_update
from rl2048.symmetry import (
    transform_actions_batch,
    transform_flat_obs_batch,
    transform_masks_batch,
)


@dataclass
class TrainMetrics:
    loss: float
    grad_norm: float
    td_error_mean: float
    mean_q: float
    max_q: float
    target_mean: float
    vanilla_target_mean: float | None = None
    td_errors: np.ndarray | None = None
    indices: np.ndarray | None = None


class DQNAgent:
    def __init__(
        self,
        *,
        device: torch.device,
        hidden_dims: tuple[int, ...] = (256, 256),
        gamma: float = 0.99,
        lr: float = 1e-4,
        use_double_dqn: bool = True,
        huber_delta: float = 1.0,
        grad_clip_norm: float = 10.0,
        obs_encoding: str = "scaled",
        network_type: str = "mlp",
        onehot_channels: int = 16,
        conv_channels: tuple[int, ...] = (128, 128, 128),
        symmetry_aug: bool = False,
        obs_dim: int = 16,
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.use_double_dqn = use_double_dqn
        self.grad_clip_norm = grad_clip_norm
        self.obs_encoding = obs_encoding
        self.network_type = network_type
        self.onehot_channels = onehot_channels
        self.symmetry_aug = symmetry_aug
        self.obs_dim = obs_dim

        self.online = build_q_network(
            network_type=network_type,
            hidden_dims=hidden_dims,
            onehot_channels=onehot_channels,
            conv_channels=conv_channels,
        ).to(device)
        self.target = clone_network(self.online).to(device)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

    def encode_obs(self, obs: np.ndarray, obs_scale: float = 16.0) -> np.ndarray:
        if self.obs_encoding == "onehot":
            return encode_onehot(obs, num_channels=self.onehot_channels)
        if self.obs_encoding == "scaled":
            return encode_scaled_flat(obs, obs_scale)
        raise ValueError(f"Unknown obs_encoding: {self.obs_encoding}")

    def scale_obs(self, obs: np.ndarray, obs_scale: float) -> np.ndarray:
        """Backward-compatible alias for scaled flat encoding."""
        return encode_scaled_flat(obs, obs_scale)

    def _obs_to_tensor(self, obs: np.ndarray, obs_scale: float) -> torch.Tensor:
        if self.obs_encoding == "onehot":
            return encode_onehot_torch(obs, num_channels=self.onehot_channels, device=self.device)
        encoded = self.encode_obs(obs, obs_scale)
        return torch.as_tensor(encoded, dtype=torch.float32, device=self.device)

    def select_action(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        epsilon: float,
        *,
        obs_scale: float = 16.0,
        generator: torch.Generator | None = None,
    ) -> tuple[int, bool]:
        state = self._obs_to_tensor(obs, obs_scale)
        if state.ndim == 3:
            state = state.unsqueeze(0)
        elif state.ndim == 1:
            state = state.unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online(state)
            actions, random_flags = select_epsilon_greedy_actions(
                q_values, mask_t, epsilon, generator=generator
            )
        return int(actions.item()), bool(random_flags.item())

    def select_actions(
        self,
        obs_batch: np.ndarray,
        mask_batch: np.ndarray,
        epsilon: float,
        *,
        obs_scale: float = 16.0,
        generator: torch.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched masked epsilon-greedy; returns (actions[B], is_random[B])."""
        state = self._obs_to_tensor(obs_batch, obs_scale)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        elif state.ndim == 3 and obs_batch.ndim == 1:
            state = state.unsqueeze(0)
        mask_t = torch.as_tensor(mask_batch, dtype=torch.bool, device=self.device)
        if mask_t.ndim == 1:
            mask_t = mask_t.unsqueeze(0)
        with torch.no_grad():
            q_values = self.online(state)
            actions, random_flags = select_epsilon_greedy_actions(
                q_values, mask_t, epsilon, generator=generator
            )
        return actions.detach().cpu().numpy(), random_flags.detach().cpu().numpy()

    def _augment_batch(self, batch: Batch, rng: np.random.Generator) -> Batch:
        if not self.symmetry_aug:
            return batch
        batch_size = batch.states.shape[0]
        transform_ids = rng.integers(0, 8, size=batch_size)
        return Batch(
            states=transform_flat_obs_batch(batch.states, transform_ids),
            actions=transform_actions_batch(batch.actions, transform_ids),
            rewards=batch.rewards,
            next_states=transform_flat_obs_batch(batch.next_states, transform_ids),
            terminated=batch.terminated,
            truncated=batch.truncated,
            valid_masks=transform_masks_batch(batch.valid_masks, transform_ids),
            next_valid_masks=transform_masks_batch(batch.next_valid_masks, transform_ids),
            bootstrap_discounts=batch.bootstrap_discounts,
            indices=batch.indices,
            ages=batch.ages,
            weights=batch.weights,
        )

    def compute_loss(
        self,
        batch: Batch,
        obs_scale: float = 16.0,
        *,
        rng: np.random.Generator | None = None,
    ) -> tuple[torch.Tensor, TrainMetrics]:
        if self.symmetry_aug:
            if rng is None:
                rng = np.random.default_rng()
            batch = self._augment_batch(batch, rng)

        states = self._obs_to_tensor(batch.states, obs_scale)
        next_states = self._obs_to_tensor(batch.next_states, obs_scale)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        terminated = torch.as_tensor(batch.terminated, dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(batch.next_valid_masks, dtype=torch.bool, device=self.device)
        discounts = torch.as_tensor(batch.bootstrap_discounts, dtype=torch.float32, device=self.device)

        q_values = self.online(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_online = self.online(next_states)
            next_q_target = self.target(next_states)
            if self.use_double_dqn:
                next_actions = masked_argmax(next_q_online, next_masks)
            else:
                next_actions = masked_argmax(next_q_target, next_masks)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            # truncated still bootstraps; only natural termination zeroes future value
            bootstrap = 1.0 - terminated
            targets = rewards + discounts * bootstrap * next_q
            vanilla_targets = None
            if self.use_double_dqn:
                vanilla_actions = masked_argmax(next_q_online, next_masks)
                vanilla_next_q = next_q_online.gather(1, vanilla_actions.unsqueeze(1)).squeeze(1)
                vanilla_targets = rewards + discounts * bootstrap * vanilla_next_q

        per_sample_loss = self.loss_fn(q_sa, targets)
        td_error = (q_sa - targets).detach()
        if batch.weights is not None:
            weights = torch.as_tensor(batch.weights, dtype=torch.float32, device=self.device)
            loss = (per_sample_loss * weights).mean()
        else:
            loss = per_sample_loss.mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        metrics = TrainMetrics(
            loss=float(loss.item()),
            grad_norm=float(grad_norm),
            td_error_mean=float(td_error.abs().mean().item()),
            mean_q=float(q_values.mean().item()),
            max_q=float(q_values.max().item()),
            target_mean=float(targets.mean().item()),
            vanilla_target_mean=float(vanilla_targets.mean().item()) if vanilla_targets is not None else None,
            td_errors=td_error.abs().cpu().numpy(),
            indices=np.asarray(batch.indices),
        )
        return loss, metrics

    def update_target(self) -> None:
        hard_update(self.target, self.online)

    def q_values(self, obs: np.ndarray, obs_scale: float = 16.0) -> np.ndarray:
        state = self._obs_to_tensor(obs, obs_scale)
        if state.ndim == 3:
            state = state.unsqueeze(0)
        elif state.ndim == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            q = self.online(state).squeeze(0).cpu().numpy()
        return q

    def q_values_batch(self, obs: np.ndarray, obs_scale: float = 16.0) -> np.ndarray:
        """Batched Q; obs (N, 16) -> (N, 4). Single (16,) -> (4,)."""
        arr = np.asarray(obs)
        state = self._obs_to_tensor(arr, obs_scale)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        elif state.ndim == 3 and arr.ndim == 1:
            state = state.unsqueeze(0)
        with torch.no_grad():
            q = self.online(state)
        out = q.detach().cpu().numpy()
        if arr.ndim == 1:
            return out[0]
        return out

    def eval_mode(self) -> None:
        self.online.eval()

    def train_mode(self) -> None:
        self.online.train()
