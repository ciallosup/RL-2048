"""Training configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import yaml


@dataclass
class TrainConfig:
    run_name: str = "dqn_baseline"
    train_seed: int = 0
    total_env_steps: int = 500_000
    max_episode_steps: int = 500
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int | None = None
    replay_capacity: int = 100_000
    learning_starts: int = 10_000
    train_freq: int = 4
    target_update_freq: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.8
    grad_clip_norm: float = 10.0
    use_double_dqn: bool = True
    huber_delta: float = 1.0
    obs_scale: float = 16.0
    hidden_dims: tuple[int, ...] = (256, 256)
    log_freq: int = 1000
    eval_freq: int = 0
    checkpoint_freq: int = 50_000
    output_dir: str = "results/runs"
    device: str = "auto"
    num_eval_episodes: int = 200
    eval_seed_set: str = "dev"

    def resolved_batch_size(self, device: torch.device) -> int:
        if self.batch_size is not None:
            return self.batch_size
        return 128 if device.type == "cuda" else 64

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hidden_dims"] = list(self.hidden_dims)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        payload = dict(data)
        if "hidden_dims" in payload:
            payload["hidden_dims"] = tuple(payload["hidden_dims"])
        return cls(**payload)


def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_config(path: Path | str) -> TrainConfig:
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return TrainConfig.from_dict(data or {})
