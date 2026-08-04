"""Tests for RL core modules."""

import numpy as np
import torch

from rl2048.rl.agent import DQNAgent
from rl2048.rl.buffer import ReplayBuffer, Transition
from rl2048.rl.config import TrainConfig, resolve_device
from rl2048.rl.masking import masked_argmax, numpy_masked_argmax, select_epsilon_greedy_actions


def test_masked_argmax_tie_break_smallest_index():
    q = torch.tensor([[1.0, 1.0, 0.5, 0.1]])
    mask = torch.tensor([[True, True, True, False]])
    assert masked_argmax(q, mask).item() == 0


def test_numpy_masked_argmax():
    q = np.array([0.1, 0.3, 0.3, -1.0], dtype=np.float32)
    mask = np.array([True, True, True, False])
    assert numpy_masked_argmax(q, mask) == 1


def test_epsilon_greedy_respects_mask():
    q = torch.tensor([[10.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[True, False, True, False]])
    actions, flags = select_epsilon_greedy_actions(q, mask, epsilon=0.0)
    assert actions.item() == 0
    assert not flags.item()


def test_replay_buffer_sample_shapes():
    buffer = ReplayBuffer(capacity=100)
    for i in range(10):
        buffer.push(
            Transition(
                state=np.ones(16, dtype=np.float32) * i,
                action=i % 4,
                reward=float(i),
                next_state=np.zeros(16, dtype=np.float32),
                terminated=i == 9,
                truncated=False,
                valid_mask=np.array([True, True, False, False]),
                next_valid_mask=np.array([True, False, True, False]),
            )
        )
    batch = buffer.sample(4, np.random.default_rng(0))
    assert batch.states.shape == (4, 16)
    assert batch.actions.shape == (4,)
    assert batch.next_valid_masks.shape == (4, 4)


def test_dqn_loss_runs():
    device = resolve_device("cpu")
    agent = DQNAgent(device=device, use_double_dqn=True)
    buffer = ReplayBuffer(capacity=32)
    for _ in range(16):
        buffer.push(
            Transition(
                state=np.random.rand(16).astype(np.float32),
                action=0,
                reward=1.0,
                next_state=np.random.rand(16).astype(np.float32),
                terminated=False,
                truncated=False,
                valid_mask=np.array([True, True, True, True]),
                next_valid_mask=np.array([True, True, True, True]),
            )
        )
    batch = buffer.sample(8, np.random.default_rng(0))
    loss, metrics = agent.compute_loss(batch)
    assert loss.ndim == 0
    assert metrics.loss >= 0.0


def test_truncated_bootstraps_in_target():
    device = resolve_device("cpu")
    agent = DQNAgent(device=device, gamma=0.99, use_double_dqn=True)
    buffer = ReplayBuffer(capacity=8)
    buffer.push(
        Transition(
            state=np.zeros(16, dtype=np.float32),
            action=1,
            reward=4.0,
            next_state=np.ones(16, dtype=np.float32),
            terminated=False,
            truncated=True,
            valid_mask=np.array([True, True, True, True]),
            next_valid_mask=np.array([True, True, False, False]),
        )
    )
    batch = buffer.sample(1, np.random.default_rng(0))
    _, metrics = agent.compute_loss(batch)
    assert metrics.target_mean != 4.0 or metrics.mean_q != 0.0


def test_vanilla_vs_double_switch():
    device = resolve_device("cpu")
    double_agent = DQNAgent(device=device, use_double_dqn=True)
    vanilla_agent = DQNAgent(device=device, use_double_dqn=False)
    buffer = ReplayBuffer(capacity=16)
    for _ in range(8):
        buffer.push(
            Transition(
                state=np.random.rand(16).astype(np.float32),
                action=2,
                reward=2.0,
                next_state=np.random.rand(16).astype(np.float32),
                terminated=False,
                truncated=False,
                valid_mask=np.array([True, True, True, True]),
                next_valid_mask=np.array([True, False, True, True]),
            )
        )
    batch = buffer.sample(4, np.random.default_rng(1))
    _, double_metrics = double_agent.compute_loss(batch)
    _, vanilla_metrics = vanilla_agent.compute_loss(batch)
    assert double_metrics.vanilla_target_mean is not None
    assert vanilla_metrics.vanilla_target_mean is None


def test_train_config_batch_size_resolution():
    cfg = TrainConfig(batch_size=None)
    assert cfg.resolved_batch_size(torch.device("cpu")) == 64


def test_obs_scale_matches_select_and_loss():
    """Buffer holds raw exponents; select_action and compute_loss must scale once the same way."""
    device = resolve_device("cpu")
    agent = DQNAgent(device=device, use_double_dqn=True)
    obs_scale = 16.0
    raw = np.full(16, 16.0, dtype=np.float32)
    expected = agent.scale_obs(raw, obs_scale)

    captured: list[torch.Tensor] = []
    original_forward = agent.online.forward

    def spy_forward(x: torch.Tensor) -> torch.Tensor:
        captured.append(x.detach().cpu().clone())
        return original_forward(x)

    agent.online.forward = spy_forward  # type: ignore[method-assign]
    mask = np.array([True, True, True, True])
    agent.select_action(raw.astype(np.int32), mask, epsilon=0.0, obs_scale=obs_scale)
    select_input = captured[-1].squeeze(0).numpy()
    np.testing.assert_allclose(select_input, expected, rtol=0.0, atol=0.0)

    buffer = ReplayBuffer(capacity=4)
    buffer.push(
        Transition(
            state=raw.copy(),
            action=0,
            reward=1.0,
            next_state=raw.copy(),
            terminated=False,
            truncated=False,
            valid_mask=mask,
            next_valid_mask=mask,
        )
    )
    captured.clear()
    batch = buffer.sample(1, np.random.default_rng(0))
    agent.compute_loss(batch, obs_scale=obs_scale)
    loss_input = captured[0].squeeze(0).numpy()
    np.testing.assert_allclose(loss_input, expected, rtol=0.0, atol=0.0)


def test_huber_delta_wired_into_loss():
    device = resolve_device("cpu")
    agent = DQNAgent(device=device, huber_delta=2.5)
    assert agent.loss_fn.beta == 2.5
