"""Multi-process vectorized 2048 environments for multi-core rollout."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from gymnasium.vector import AsyncVectorEnv, AutoresetMode, SyncVectorEnv

from rl2048.env import Game2048Env


def make_env_fn(max_episode_steps: int | None) -> Callable[[], Game2048Env]:
    def _thunk() -> Game2048Env:
        return Game2048Env(max_episode_steps=max_episode_steps)

    return _thunk


def build_vector_env(
    num_envs: int,
    *,
    max_episode_steps: int | None,
    async_mode: bool = True,
) -> AsyncVectorEnv | SyncVectorEnv:
    """Build vector env. Async uses spawn workers (true multi-core)."""
    if num_envs < 1:
        raise ValueError("num_envs must be >= 1")
    env_fns = [make_env_fn(max_episode_steps) for _ in range(num_envs)]
    # SAME_STEP: on done, returned obs is reset; true terminal board is in final_obs.
    autoreset = AutoresetMode.SAME_STEP
    if num_envs == 1 or not async_mode:
        return SyncVectorEnv(env_fns, autoreset_mode=autoreset)
    return AsyncVectorEnv(env_fns, context="spawn", autoreset_mode=autoreset)


def stack_valid_masks(infos: dict[str, Any], num_envs: int) -> np.ndarray:
    """Extract (N, 4) valid action masks from vector-env info dict."""
    raw = infos.get("valid_action_mask")
    if raw is None:
        return np.ones((num_envs, 4), dtype=np.bool_)
    arr = np.asarray(raw)
    if arr.ndim == 2 and arr.shape[0] == num_envs:
        return arr.astype(np.bool_)
    return np.stack([np.asarray(x, dtype=np.bool_) for x in raw], axis=0)


def true_next_observations(
    next_obs: np.ndarray,
    terminations: np.ndarray,
    truncations: np.ndarray,
    infos: dict[str, Any],
) -> np.ndarray:
    """
    Recover true next_state for transitions when vector env auto-resets (SAME_STEP).

    After done, returned obs is the *new* episode obs; terminal board is in infos['final_obs'].
    """
    out = np.asarray(next_obs, dtype=np.float32).copy()
    done = np.asarray(terminations) | np.asarray(truncations)
    if not np.any(done):
        return out
    finals = infos.get("final_obs", infos.get("final_observation"))
    if finals is None:
        return out
    for i, is_done in enumerate(done):
        if not is_done:
            continue
        final = finals[i]
        if final is None:
            continue
        out[i] = np.asarray(final, dtype=np.float32)
    return out


def true_next_masks(
    next_masks: np.ndarray,
    terminations: np.ndarray,
    truncations: np.ndarray,
    infos: dict[str, Any],
) -> np.ndarray:
    """Recover valid masks at terminal/truncated states from final_info."""
    out = np.asarray(next_masks, dtype=np.bool_).copy()
    done = np.asarray(terminations) | np.asarray(truncations)
    if not np.any(done):
        return out
    final_info = infos.get("final_info")
    if final_info is None:
        return out
    # Gymnasium 1.x: final_info is a batched dict of arrays + _final_info mask.
    if isinstance(final_info, dict) and "valid_action_mask" in final_info:
        masks = np.asarray(final_info["valid_action_mask"])
        for i, is_done in enumerate(done):
            if is_done:
                out[i] = masks[i]
        return out
    # Fallback: list of per-env dicts
    for i, is_done in enumerate(done):
        if not is_done:
            continue
        info_i = final_info[i]
        if not isinstance(info_i, dict):
            continue
        mask = info_i.get("valid_action_mask")
        if mask is not None:
            out[i] = np.asarray(mask, dtype=np.bool_)
    return out


def episode_stat(infos: dict[str, Any], env_i: int, key: str, default: Any = 0) -> Any:
    """Read episode metric for a done env from final_info (batched) or infos."""
    final_info = infos.get("final_info")
    if isinstance(final_info, dict) and key in final_info:
        return final_info[key][env_i]
    if isinstance(final_info, (list, tuple)) and env_i < len(final_info):
        info_i = final_info[env_i]
        if isinstance(info_i, dict) and key in info_i:
            return info_i[key]
    raw = infos.get(key)
    if raw is not None:
        return raw[env_i]
    return default
