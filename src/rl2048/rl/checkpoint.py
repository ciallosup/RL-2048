"""Checkpoint save/load."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

from rl2048.rl.agent import DQNAgent
from rl2048.rl.config import TrainConfig


def save_checkpoint(
    path: Path,
    *,
    agent: DQNAgent,
    config: TrainConfig,
    env_steps: int,
    episode_idx: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "env_steps": env_steps,
        "episode_idx": episode_idx,
        "config": config.to_dict(),
        "online_state_dict": agent.online.state_dict(),
        "target_state_dict": agent.target.state_dict(),
        "optimizer_state_dict": agent.optimizer.state_dict(),
    }
    torch.save(payload, path)
    config_path = path.parent / "config.yaml"
    if not config_path.exists():
        config_path.write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")


def load_checkpoint(path: Path, *, device: torch.device | None = None) -> tuple[DQNAgent, TrainConfig, dict[str, Any]]:
    map_location = device or torch.device("cpu")
    payload = torch.load(path, map_location=map_location, weights_only=False)
    config = TrainConfig.from_dict(payload["config"])
    device = device or resolve_device_from_config(config)
    agent = DQNAgent(
        hidden_dims=config.hidden_dims,
        device=device,
        gamma=config.gamma,
        lr=config.lr,
        use_double_dqn=config.use_double_dqn,
        huber_delta=config.huber_delta,
        grad_clip_norm=config.grad_clip_norm,
    )
    agent.online.load_state_dict(payload["online_state_dict"])
    agent.target.load_state_dict(payload["target_state_dict"])
    if "optimizer_state_dict" in payload:
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
    meta = {
        "env_steps": payload.get("env_steps", 0),
        "episode_idx": payload.get("episode_idx", 0),
    }
    return agent, config, meta


def resolve_device_from_config(config: TrainConfig) -> torch.device:
    from rl2048.rl.config import resolve_device

    return resolve_device(config.device)


def latest_checkpoint(run_dir: Path) -> Path | None:
    if not run_dir.exists():
        return None
    candidates = sorted(run_dir.glob("checkpoint_*.pt"))
    return candidates[-1] if candidates else None
