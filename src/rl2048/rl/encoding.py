"""Observation encodings for DQN inputs."""

from __future__ import annotations

import numpy as np

# Exponent channels: empty=0, 2=1, ..., 2^15=15. Matches env Box high≈17 with clip.
DEFAULT_ONEHOT_CHANNELS = 16


def encode_scaled_flat(obs: np.ndarray, obs_scale: float) -> np.ndarray:
    """16-dim scaled exponents (legacy MLP path)."""
    arr = obs.astype(np.float32)
    if obs_scale:
        arr = arr / obs_scale
    return arr


def encode_onehot(obs: np.ndarray, num_channels: int = DEFAULT_ONEHOT_CHANNELS) -> np.ndarray:
    """
    Encode flat or batched exponent observations as one-hot planes.

    obs: (16,) or (B, 16) with integer-like exponents.
    returns: (C, 4, 4) or (B, C, 4, 4) float32.
    """
    single = obs.ndim == 1
    flat = obs.reshape(1, -1) if single else obs.reshape(obs.shape[0], -1)
    if flat.shape[-1] != 16:
        raise ValueError(f"Expected 16-dim observation, got shape {obs.shape}")

    batch = flat.shape[0]
    exponents = np.clip(flat.astype(np.int64), 0, num_channels - 1)
    eye = np.eye(num_channels, dtype=np.float32)
    # (B, 16, C) -> (B, C, 16) -> (B, C, 4, 4)
    out = eye[exponents].transpose(0, 2, 1).reshape(batch, num_channels, 4, 4)
    return out[0] if single else out


def encode_onehot_torch(
    obs: np.ndarray | "torch.Tensor",
    num_channels: int = DEFAULT_ONEHOT_CHANNELS,
    *,
    device: "torch.device | str | None" = None,
) -> "torch.Tensor":
    """GPU-friendly one-hot encoding to (B, C, 4, 4) float32 tensor."""
    import torch
    import torch.nn.functional as F

    t = torch.as_tensor(obs, device=device, dtype=torch.long)
    single = t.ndim == 1
    if single:
        t = t.unsqueeze(0)
    if t.shape[-1] != 16:
        raise ValueError(f"Expected 16-dim observation, got shape {tuple(t.shape)}")
    t = t.clamp(0, num_channels - 1)
    oh = F.one_hot(t, num_classes=num_channels).to(dtype=torch.float32)
    oh = oh.permute(0, 2, 1).reshape(-1, num_channels, 4, 4)
    return oh[0] if single else oh


def transform_flat_obs(obs: np.ndarray, transform_id: int) -> np.ndarray:
    """Apply D4 transform to a flat 16-dim board observation."""
    from rl2048.symmetry import transform_board

    board = np.asarray(obs, dtype=np.float32).reshape(4, 4)
    return transform_board(board, transform_id).reshape(-1).astype(np.float32)
