"""Q-network MLP for 2048."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from rl2048.core import NUM_ACTIONS


class QNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int = 16,
        hidden_dims: tuple[int, ...] = (256, 256),
        num_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        layers.append(nn.Linear(in_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def clone_network(network: QNetwork) -> QNetwork:
    clone = copy.deepcopy(network)
    clone.load_state_dict(network.state_dict())
    return clone
