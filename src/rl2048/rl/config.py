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
    # If set, overrides fraction-based schedule (absolute env steps to reach epsilon_end).
    epsilon_decay_steps: int | None = None
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
    # Optimization stack
    obs_encoding: str = "scaled"  # scaled | onehot
    network_type: str = "mlp"  # mlp | cnn | dueling_cnn
    onehot_channels: int = 16
    conv_channels: tuple[int, ...] = (128, 128, 128)
    symmetry_aug: bool = False
    n_step: int = 1
    reward_mode: str = "raw"  # raw | log1p
    use_per: bool = False
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 1_000_000
    num_envs: int = 1  # >1 uses multi-process AsyncVectorEnv (multi-core)
    init_checkpoint: str | None = None
    # Rollout policy while collecting replay. greedy = epsilon-greedy Q.
    # 2ply uses expectimax so fine-tune actually visits 2048/4096 boards.
    collect_decode: str = "greedy"  # greedy | 1ply | 2ply | 3ply
    collect_corner_tiebreak: bool = True
    # Cap random actions once the max tile is this high (search endgames).
    collect_endgame_epsilon: float = 0.02
    collect_endgame_tile: int = 1024
    # If true, 2-ply collection uses a frozen copy of init_checkpoint (not the online net).
    collect_frozen_teacher: bool = False
    bc_coef: float = 0.0  # cross-entropy toward the collected action
    td_coef: float = 1.0
    freeze_target: bool = False  # keep target at init weights

    def resolved_batch_size(self, device: torch.device) -> int:
        if self.batch_size is not None:
            return self.batch_size
        return 128 if device.type == "cuda" else 64

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["hidden_dims"] = list(self.hidden_dims)
        data["conv_channels"] = list(self.conv_channels)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        payload = dict(data)
        if "hidden_dims" in payload:
            payload["hidden_dims"] = tuple(payload["hidden_dims"])
        if "conv_channels" in payload:
            payload["conv_channels"] = tuple(payload["conv_channels"])
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        payload = {k: v for k, v in payload.items() if k in known}
        return cls(**payload)


def resolve_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_config(path: Path | str) -> TrainConfig:
    with Path(path).open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return TrainConfig.from_dict(data or {})
