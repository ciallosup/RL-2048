"""Checkpoint and trainer integration tests."""

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from rl2048.env import Game2048Env
from rl2048.policies.base import PolicyContext
from rl2048.policies.dqn_policy import DQNPolicy
from rl2048.rl.checkpoint import load_checkpoint, save_checkpoint
from rl2048.rl.config import TrainConfig, load_config, resolve_device
from rl2048.rl.trainer import Trainer


def test_load_yaml_config(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("run_name: test\ntotal_env_steps: 123\n", encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg.run_name == "test"
    assert cfg.total_env_steps == 123


def test_checkpoint_roundtrip(tmp_path):
    device = resolve_device("cpu")
    config = TrainConfig(total_env_steps=100, run_name="ckpt_test")
    from rl2048.rl.agent import DQNAgent

    agent = DQNAgent(device=device, use_double_dqn=True)
    ckpt = tmp_path / "checkpoint.pt"
    save_checkpoint(ckpt, agent=agent, config=config, env_steps=42, episode_idx=3)
    loaded, loaded_cfg, meta = load_checkpoint(ckpt, device=device)
    assert loaded_cfg.run_name == "ckpt_test"
    assert meta["env_steps"] == 42
    obs = np.zeros(16, dtype=np.int32)
    q1 = agent.q_values(obs)
    q2 = loaded.q_values(obs)
    np.testing.assert_allclose(q1, q2, rtol=1e-5, atol=1e-5)


def test_dqn_policy_greedy_action(tmp_path):
    device = resolve_device("cpu")
    config = TrainConfig()
    from rl2048.rl.agent import DQNAgent

    agent = DQNAgent(device=device)
    ckpt = tmp_path / "checkpoint.pt"
    save_checkpoint(ckpt, agent=agent, config=config, env_steps=1, episode_idx=0)
    policy = DQNPolicy.from_checkpoint(ckpt)
    env = Game2048Env()
    obs, info = env.reset(seed=0)
    ctx = PolicyContext(env=env, obs=obs, info=info)
    action = policy.select_action(ctx)
    assert 0 <= action < 4
    assert info["valid_action_mask"][action]


@pytest.mark.slow
def test_trainer_short_run(tmp_path):
    config = TrainConfig(
        run_name="short",
        total_env_steps=500,
        learning_starts=50,
        train_freq=2,
        log_freq=200,
        checkpoint_freq=0,
        output_dir=str(tmp_path),
        num_envs=1,
    )
    result = Trainer(config).train()
    assert result.env_steps == 500
    assert result.checkpoint_path.exists()


@pytest.mark.slow
def test_trainer_multi_env_short_run(tmp_path):
    config = TrainConfig(
        run_name="short_vec",
        total_env_steps=256,
        learning_starts=32,
        train_freq=4,
        log_freq=128,
        checkpoint_freq=0,
        output_dir=str(tmp_path),
        num_envs=4,
        device="cpu",
        max_episode_steps=40,
    )
    result = Trainer(config).train()
    assert result.env_steps == 256
    assert result.checkpoint_path.exists()
    assert result.episodes >= 1
