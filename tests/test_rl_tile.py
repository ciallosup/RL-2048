"""Tests for tile-horizon upgrades: log1p reward, dueling CNN, PER."""

from __future__ import annotations

import numpy as np
import torch

from rl2048.rl.agent import DQNAgent
from rl2048.rl.buffer import PrioritizedReplayBuffer, ReplayBuffer, Transition
from rl2048.rl.config import TrainConfig, resolve_device
from rl2048.rl.network import DuelingCNNQNetwork, build_q_network
from rl2048.rl.rewards import transform_reward


def test_transform_reward_log1p():
    assert transform_reward(0.0, "raw") == 0.0
    assert transform_reward(0.0, "log1p") == 0.0
    assert abs(transform_reward(np.e - 1, "log1p") - 1.0) < 1e-6


def test_dueling_cnn_forward():
    net = build_q_network(
        network_type="dueling_cnn",
        conv_channels=(32, 32),
        hidden_dims=(64,),
        onehot_channels=16,
    )
    assert isinstance(net, DuelingCNNQNetwork)
    x = torch.zeros(2, 16, 4, 4)
    q = net(x)
    assert q.shape == (2, 4)


def test_dueling_agent_loss():
    device = resolve_device("cpu")
    agent = DQNAgent(
        device=device,
        obs_encoding="onehot",
        network_type="dueling_cnn",
        conv_channels=(32, 32),
        hidden_dims=(64,),
    )
    buffer = ReplayBuffer(capacity=32)
    for _ in range(16):
        buffer.push(
            Transition(
                state=np.random.randint(0, 8, size=16).astype(np.float32),
                action=0,
                reward=1.0,
                next_state=np.random.randint(0, 8, size=16).astype(np.float32),
                terminated=False,
                truncated=False,
                valid_mask=np.ones(4, dtype=bool),
                next_valid_mask=np.ones(4, dtype=bool),
                bootstrap_discount=0.995,
            )
        )
    loss, metrics = agent.compute_loss(buffer.sample(8, np.random.default_rng(0)))
    assert loss.ndim == 0
    assert metrics.td_errors is not None
    assert metrics.indices is not None


def test_per_sample_and_update():
    buf = PrioritizedReplayBuffer(capacity=64, alpha=0.6, beta_start=0.4, beta_frames=100)
    for i in range(32):
        buf.push(
            Transition(
                state=np.full(16, i, dtype=np.float32),
                action=i % 4,
                reward=float(i),
                next_state=np.zeros(16, dtype=np.float32),
                terminated=False,
                truncated=False,
                valid_mask=np.ones(4, dtype=bool),
                next_valid_mask=np.ones(4, dtype=bool),
                bootstrap_discount=0.99,
            )
        )
    batch = buf.sample(8, np.random.default_rng(0))
    assert batch.weights is not None
    assert batch.weights.shape == (8,)
    assert np.all(batch.weights > 0)
    buf.update_priorities(batch.indices, np.linspace(0.1, 2.0, num=8))


def test_config_new_fields_roundtrip():
    cfg = TrainConfig(
        reward_mode="log1p",
        network_type="dueling_cnn",
        use_per=True,
        max_episode_steps=1200,
        n_step=5,
        gamma=0.995,
    )
    restored = TrainConfig.from_dict(cfg.to_dict())
    assert restored.reward_mode == "log1p"
    assert restored.network_type == "dueling_cnn"
    assert restored.use_per is True
    assert restored.max_episode_steps == 1200
