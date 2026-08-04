"""Masked DQN / Double DQN agent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from rl2048.rl.buffer import Batch
from rl2048.rl.masking import masked_argmax, select_epsilon_greedy_actions
from rl2048.rl.network import QNetwork, clone_network, hard_update


@dataclass
class TrainMetrics:
    loss: float
    grad_norm: float
    td_error_mean: float
    mean_q: float
    max_q: float
    target_mean: float
    vanilla_target_mean: float | None = None


class DQNAgent:
    def __init__(
        self,
        *,
        obs_dim: int = 16,
        hidden_dims: tuple[int, ...] = (256, 256),
        device: torch.device,
        gamma: float = 0.99,
        lr: float = 1e-4,
        use_double_dqn: bool = True,
        huber_delta: float = 1.0,
        grad_clip_norm: float = 10.0,
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.use_double_dqn = use_double_dqn
        self.grad_clip_norm = grad_clip_norm
        self.huber_delta = huber_delta
        self.online = QNetwork(obs_dim, hidden_dims).to(device)
        self.target = clone_network(self.online).to(device)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        # SmoothL1Loss beta == Huber delta; wire config through (default was always 1.0).
        self.loss_fn = nn.SmoothL1Loss(reduction="none", beta=huber_delta)

    def scale_obs(self, obs: np.ndarray, obs_scale: float) -> np.ndarray:
        return (obs.astype(np.float32) / obs_scale) if obs_scale else obs.astype(np.float32)

    def select_action(
        self,
        obs: np.ndarray,
        mask: np.ndarray,
        epsilon: float,
        *,
        obs_scale: float = 16.0,
        generator: torch.Generator | None = None,
    ) -> tuple[int, bool]:
        state = torch.as_tensor(
            self.scale_obs(obs, obs_scale),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online(state)
            actions, random_flags = select_epsilon_greedy_actions(
                q_values, mask_t, epsilon, generator=generator
            )
        return int(actions.item()), bool(random_flags.item())

    def compute_loss(self, batch: Batch, obs_scale: float = 16.0) -> tuple[torch.Tensor, TrainMetrics]:
        # Expect raw exponent obs in the buffer; scale once here (same as select_action).
        states = torch.as_tensor(self.scale_obs(batch.states, obs_scale), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            self.scale_obs(batch.next_states, obs_scale), dtype=torch.float32, device=self.device
        )
        terminated = torch.as_tensor(batch.terminated, dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(batch.next_valid_masks, dtype=torch.bool, device=self.device)

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
            targets = rewards + self.gamma * bootstrap * next_q
            vanilla_targets = None
            if self.use_double_dqn:
                vanilla_actions = masked_argmax(next_q_online, next_masks)
                vanilla_next_q = next_q_online.gather(1, vanilla_actions.unsqueeze(1)).squeeze(1)
                vanilla_targets = rewards + self.gamma * bootstrap * vanilla_next_q

        per_sample_loss = self.loss_fn(q_sa, targets)
        loss = per_sample_loss.mean()
        td_error = (q_sa - targets).detach()

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
        )
        return loss, metrics

    def update_target(self) -> None:
        hard_update(self.target, self.online)

    def q_values(self, obs: np.ndarray, obs_scale: float = 16.0) -> np.ndarray:
        state = torch.as_tensor(
            self.scale_obs(obs, obs_scale),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            q = self.online(state).squeeze(0).cpu().numpy()
        return q

    def eval_mode(self) -> None:
        self.online.eval()

    def train_mode(self) -> None:
        self.online.train()
