"""Tests for encoding, CNN, D4 aug, and n-step returns."""

from __future__ import annotations

import numpy as np
import torch

from rl2048.rl.agent import DQNAgent
from rl2048.rl.buffer import NStepAssembler, ReplayBuffer, Transition
from rl2048.rl.config import TrainConfig, resolve_device
from rl2048.rl.encoding import encode_onehot, transform_flat_obs
from rl2048.rl.network import CNNQNetwork, build_q_network
from rl2048.symmetry import transform_action, transform_mask


def test_encode_onehot_shape_and_channels():
    obs = np.array([0, 1, 2, 3, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 15], dtype=np.float32)
    encoded = encode_onehot(obs, num_channels=16)
    assert encoded.shape == (16, 4, 4)
    assert encoded[:, 0, 0].sum() == 1.0
    assert encoded[0, 0, 0] == 1.0
    assert encoded[1, 0, 1] == 1.0
    assert encoded[15, 3, 3] == 1.0


def test_cnn_forward_and_loss():
    device = resolve_device("cpu")
    agent = DQNAgent(
        device=device,
        obs_encoding="onehot",
        network_type="cnn",
        hidden_dims=(128,),
        conv_channels=(32, 32),
        onehot_channels=16,
    )
    obs = np.arange(16, dtype=np.float32) % 8
    action, _ = agent.select_action(obs, np.ones(4, dtype=bool), epsilon=0.0)
    assert 0 <= action < 4
    q = agent.q_values(obs)
    assert q.shape == (4,)

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
                valid_mask=np.array([True, True, True, True]),
                next_valid_mask=np.array([True, True, True, True]),
                bootstrap_discount=0.99,
            )
        )
    loss, metrics = agent.compute_loss(buffer.sample(8, np.random.default_rng(0)))
    assert loss.ndim == 0
    assert metrics.loss >= 0.0


def test_build_q_network_types():
    mlp = build_q_network(network_type="mlp", hidden_dims=(64, 64))
    cnn = build_q_network(network_type="cnn", hidden_dims=(64,), conv_channels=(16, 16))
    assert isinstance(cnn, CNNQNetwork)
    x = torch.zeros(2, 16)
    assert mlp(x).shape == (2, 4)
    x4 = torch.zeros(2, 16, 4, 4)
    assert cnn(x4).shape == (2, 4)


def test_transform_mask_matches_action_map():
    mask = np.array([True, False, True, False])
    for tid in range(8):
        out = transform_mask(mask, tid)
        expected = np.zeros(4, dtype=bool)
        for a in range(4):
            if mask[a]:
                expected[transform_action(a, tid)] = True
        np.testing.assert_array_equal(out, expected)


def test_symmetry_aug_preserves_batch_shapes():
    device = resolve_device("cpu")
    agent = DQNAgent(device=device, symmetry_aug=True, use_double_dqn=True)
    buffer = ReplayBuffer(capacity=16)
    for i in range(8):
        buffer.push(
            Transition(
                state=np.arange(16, dtype=np.float32) + i,
                action=i % 4,
                reward=1.0,
                next_state=np.arange(16, dtype=np.float32)[::-1],
                terminated=False,
                truncated=False,
                valid_mask=np.array([True, True, False, True]),
                next_valid_mask=np.array([True, False, True, True]),
                bootstrap_discount=0.99,
            )
        )
    batch = buffer.sample(4, np.random.default_rng(0))
    aug = agent._augment_batch(batch, np.random.default_rng(1))
    assert aug.states.shape == batch.states.shape
    assert aug.actions.shape == batch.actions.shape
    assert aug.next_valid_masks.shape == batch.next_valid_masks.shape
    loss, _ = agent.compute_loss(batch, rng=np.random.default_rng(2))
    assert loss.ndim == 0


def test_vectorized_d4_matches_scalar():
    from rl2048.symmetry import (
        transform_action,
        transform_actions_batch,
        transform_flat_obs_batch,
        transform_mask,
        transform_masks_batch,
    )
    from rl2048.rl.encoding import transform_flat_obs

    states = np.arange(16, dtype=np.float32)[None, :].repeat(8, axis=0)
    states = states + np.arange(8, dtype=np.float32)[:, None]
    tids = np.arange(8, dtype=np.int64)
    actions = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    masks = np.array([[True, False, True, True]] * 8)

    got_s = transform_flat_obs_batch(states, tids)
    got_a = transform_actions_batch(actions, tids)
    got_m = transform_masks_batch(masks, tids)
    for i, tid in enumerate(tids):
        np.testing.assert_array_equal(got_s[i], transform_flat_obs(states[i], int(tid)))
        assert got_a[i] == transform_action(int(actions[i]), int(tid))
        np.testing.assert_array_equal(got_m[i], transform_mask(masks[i], int(tid)))


def test_encode_onehot_torch_matches_numpy():
    from rl2048.rl.encoding import encode_onehot, encode_onehot_torch

    obs = np.arange(16, dtype=np.float32) % 8
    np_out = encode_onehot(obs)
    torch_out = encode_onehot_torch(obs, device="cpu").numpy()
    np.testing.assert_allclose(np_out, torch_out)


def test_n_step_assembler_full_window():
    asm = NStepAssembler(n_step=3, gamma=0.5)
    emitted = []
    for i in range(3):
        emitted.extend(
            asm.push(
                Transition(
                    state=np.full(16, i, dtype=np.float32),
                    action=0,
                    reward=2.0,
                    next_state=np.full(16, i + 1, dtype=np.float32),
                    terminated=False,
                    truncated=False,
                    valid_mask=np.ones(4, dtype=bool),
                    next_valid_mask=np.ones(4, dtype=bool),
                )
            )
        )
    assert len(emitted) == 1
    # 2 + 0.5*2 + 0.25*2 = 3.5
    assert abs(emitted[0].reward - 3.5) < 1e-6
    assert abs(emitted[0].bootstrap_discount - (0.5**3)) < 1e-6
    assert not emitted[0].terminated


def test_n_step_terminated_cuts_bootstrap():
    asm = NStepAssembler(n_step=3, gamma=0.9)
    out = []
    out.extend(
        asm.push(
            Transition(
                state=np.zeros(16, dtype=np.float32),
                action=1,
                reward=4.0,
                next_state=np.ones(16, dtype=np.float32),
                terminated=True,
                truncated=False,
                valid_mask=np.ones(4, dtype=bool),
                next_valid_mask=np.ones(4, dtype=bool),
            )
        )
    )
    assert len(out) == 1
    assert out[0].reward == 4.0
    assert out[0].terminated is True
    assert out[0].bootstrap_discount == 0.0


def test_n_step_truncated_still_bootstraps():
    asm = NStepAssembler(n_step=3, gamma=0.9)
    out = asm.push(
        Transition(
            state=np.zeros(16, dtype=np.float32),
            action=1,
            reward=4.0,
            next_state=np.ones(16, dtype=np.float32),
            terminated=False,
            truncated=True,
            valid_mask=np.ones(4, dtype=bool),
            next_valid_mask=np.ones(4, dtype=bool),
        )
    )
    assert len(out) == 1
    assert out[0].truncated is True
    assert out[0].terminated is False
    assert abs(out[0].bootstrap_discount - 0.9) < 1e-6


def test_config_roundtrip_new_fields():
    cfg = TrainConfig(
        obs_encoding="onehot",
        network_type="cnn",
        symmetry_aug=True,
        n_step=3,
        conv_channels=(64, 64),
        hidden_dims=(128,),
    )
    restored = TrainConfig.from_dict(cfg.to_dict())
    assert restored.obs_encoding == "onehot"
    assert restored.network_type == "cnn"
    assert restored.symmetry_aug is True
    assert restored.n_step == 3
    assert restored.conv_channels == (64, 64)
