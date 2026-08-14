"""Q-networks for 2048 (MLP, CNN, Dueling CNN)."""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from rl2048.core import NUM_ACTIONS
from rl2048.rl.encoding import DEFAULT_ONEHOT_CHANNELS


class QNetwork(nn.Module):
    """MLP over scaled flat exponents."""

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


def _build_conv_stack(
    num_channels: int,
    conv_channels: tuple[int, ...],
) -> tuple[nn.Sequential, int]:
    convs: list[nn.Module] = []
    in_c = num_channels
    for out_c in conv_channels:
        convs.extend(
            [
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, stride=1),
                nn.ReLU(),
            ]
        )
        in_c = out_c
    return nn.Sequential(*convs), in_c


class CNNQNetwork(nn.Module):
    """No-pooling CNN over one-hot C×4×4 board planes."""

    def __init__(
        self,
        num_channels: int = DEFAULT_ONEHOT_CHANNELS,
        conv_channels: tuple[int, ...] = (128, 128, 128),
        hidden_dims: tuple[int, ...] = (256,),
        num_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        self.conv, out_c = _build_conv_stack(num_channels, conv_channels)
        flat_dim = out_c * 4 * 4
        head: list[nn.Module] = []
        in_dim = flat_dim
        for hidden in hidden_dims:
            head.extend([nn.Linear(in_dim, hidden), nn.ReLU()])
            in_dim = hidden
        head.append(nn.Linear(in_dim, num_actions))
        self.head = nn.Sequential(*head)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.conv(obs)
        x = x.flatten(start_dim=1)
        return self.head(x)


class DuelingCNNQNetwork(nn.Module):
    """Dueling CNN: Q = V(s) + A(s,a) - mean_a A(s,a)."""

    def __init__(
        self,
        num_channels: int = DEFAULT_ONEHOT_CHANNELS,
        conv_channels: tuple[int, ...] = (256, 256, 256),
        hidden_dims: tuple[int, ...] = (512,),
        num_actions: int = NUM_ACTIONS,
    ) -> None:
        super().__init__()
        self.conv, out_c = _build_conv_stack(num_channels, conv_channels)
        flat_dim = out_c * 4 * 4
        hidden = hidden_dims[0] if hidden_dims else 512

        self.value_stream = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.conv(obs).flatten(start_dim=1)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


def build_q_network(
    *,
    network_type: str = "mlp",
    hidden_dims: tuple[int, ...] = (256, 256),
    onehot_channels: int = DEFAULT_ONEHOT_CHANNELS,
    conv_channels: tuple[int, ...] = (128, 128, 128),
) -> nn.Module:
    if network_type == "mlp":
        return QNetwork(obs_dim=16, hidden_dims=hidden_dims)
    if network_type == "cnn":
        return CNNQNetwork(
            num_channels=onehot_channels,
            conv_channels=conv_channels,
            hidden_dims=hidden_dims if hidden_dims else (256,),
        )
    if network_type == "dueling_cnn":
        return DuelingCNNQNetwork(
            num_channels=onehot_channels,
            conv_channels=conv_channels if conv_channels else (256, 256, 256),
            hidden_dims=hidden_dims if hidden_dims else (512,),
        )
    raise ValueError(f"Unknown network_type: {network_type}")


def hard_update(target: nn.Module, source: nn.Module) -> None:
    target.load_state_dict(source.state_dict())


def clone_network(network: nn.Module) -> nn.Module:
    clone = copy.deepcopy(network)
    clone.load_state_dict(network.state_dict())
    return clone
