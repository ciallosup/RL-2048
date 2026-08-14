"""DQN training loop."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from rl2048.rl.agent import DQNAgent
from rl2048.rl.buffer import NStepAssembler, PrioritizedReplayBuffer, ReplayBuffer, Transition
from rl2048.rl.checkpoint import save_checkpoint
from rl2048.rl.config import TrainConfig, resolve_device
from rl2048.rl.logger import MetricsLogger
from rl2048.rl.rewards import transform_reward
from rl2048.rl.vec_env import (
    build_vector_env,
    episode_stat,
    stack_valid_masks,
    true_next_masks,
    true_next_observations,
)


@dataclass
class TrainResult:
    run_dir: Path
    env_steps: int
    episodes: int
    checkpoint_path: Path


class Trainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.rng = np.random.default_rng(config.train_seed)
        self.num_envs = max(1, int(config.num_envs))

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.run_dir = Path(config.output_dir) / f"{config.run_name}_seed{config.train_seed}_{timestamp}"
        self.logger = MetricsLogger(self.run_dir)
        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False),
            encoding="utf-8",
        )

        # Build envs before CUDA agent so Async spawn workers stay CPU-only.
        self.envs = build_vector_env(
            self.num_envs,
            max_episode_steps=config.max_episode_steps,
            async_mode=self.num_envs > 1,
        )
        self.agent = DQNAgent(
            hidden_dims=config.hidden_dims,
            device=self.device,
            gamma=config.gamma,
            lr=config.lr,
            use_double_dqn=config.use_double_dqn,
            huber_delta=config.huber_delta,
            grad_clip_norm=config.grad_clip_norm,
            obs_encoding=config.obs_encoding,
            network_type=config.network_type,
            onehot_channels=config.onehot_channels,
            conv_channels=config.conv_channels,
            symmetry_aug=config.symmetry_aug,
        )
        if config.use_per:
            self.buffer: ReplayBuffer = PrioritizedReplayBuffer(
                config.replay_capacity,
                alpha=config.per_alpha,
                beta_start=config.per_beta_start,
                beta_frames=config.per_beta_frames,
            )
        else:
            self.buffer = ReplayBuffer(config.replay_capacity)
        self.n_steps = [
            NStepAssembler(n_step=config.n_step, gamma=config.gamma) for _ in range(self.num_envs)
        ]
        self.batch_size = config.resolved_batch_size(self.device)
        print(
            f"Training: device={self.device}, batch_size={self.batch_size}, "
            f"num_envs={self.num_envs}, encoding={config.obs_encoding}, net={config.network_type}, "
            f"n_step={config.n_step}, symmetry_aug={config.symmetry_aug}, "
            f"reward={config.reward_mode}, per={config.use_per}, "
            f"max_steps={config.max_episode_steps}, run_dir={self.run_dir}",
            flush=True,
        )
        self.torch_generator = torch.Generator(device=self.device)
        self.torch_generator.manual_seed(config.train_seed + 17)

        self.env_steps = 0
        self.episode_idx = 0
        self.updates = 0
        self.epsilon = config.epsilon_start

    def _episode_reset_seed(self, env_i: int = 0) -> int:
        return int(self.config.train_seed * 10_000 + self.episode_idx * self.num_envs + env_i)

    def _current_epsilon(self) -> float:
        if self.config.epsilon_decay_steps is not None:
            decay_steps = int(self.config.epsilon_decay_steps)
        else:
            decay_steps = int(self.config.total_env_steps * self.config.epsilon_decay_fraction)
        if self.env_steps >= decay_steps:
            return self.config.epsilon_end
        progress = self.env_steps / max(decay_steps, 1)
        return self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)

    def _store_transition(self, env_i: int, transition: Transition) -> None:
        for assembled in self.n_steps[env_i].push(transition):
            self.buffer.push(assembled)

    def _flush_n_step(self, env_i: int) -> None:
        for assembled in self.n_steps[env_i].flush():
            self.buffer.push(assembled)

    def _maybe_train_step(self) -> None:
        if (
            self.env_steps >= self.config.learning_starts
            and self.env_steps % self.config.train_freq == 0
            and len(self.buffer) >= self.batch_size
        ):
            batch = self.buffer.sample(self.batch_size, self.rng)
            _, metrics = self.agent.compute_loss(
                batch,
                obs_scale=self.config.obs_scale,
                rng=self.rng,
            )
            if metrics.td_errors is not None and metrics.indices is not None:
                self.buffer.update_priorities(metrics.indices, metrics.td_errors)
            self.updates += 1
            self.logger.state.updates = self.updates

            if self.env_steps % self.config.log_freq == 0:
                utd_ratio = self.updates / max(self.env_steps, 1)
                extra = {
                    "epsilon": self.epsilon,
                    "loss": metrics.loss,
                    "grad_norm": metrics.grad_norm,
                    "td_error_mean": metrics.td_error_mean,
                    "mean_q": metrics.mean_q,
                    "max_q": metrics.max_q,
                    "target_mean": metrics.target_mean,
                    "vanilla_target_mean": metrics.vanilla_target_mean,
                    "buffer_size": len(self.buffer),
                    "reward_mean": float(np.mean(self.buffer.rewards[: len(self.buffer)])),
                    "utd_ratio": utd_ratio,
                    "num_envs": self.num_envs,
                }
                self.logger.log_step(self.logger.snapshot(extra))

    def _after_env_step(self) -> None:
        """Advance counters that are tied to a single env transition."""
        self.env_steps += 1
        self.logger.state.env_steps = self.env_steps
        self._maybe_train_step()
        if self.env_steps % self.config.target_update_freq == 0:
            self.agent.update_target()
        if self.config.checkpoint_freq and self.env_steps % self.config.checkpoint_freq == 0:
            ckpt = self.run_dir / f"checkpoint_{self.env_steps}.pt"
            save_checkpoint(
                ckpt,
                agent=self.agent,
                config=self.config,
                env_steps=self.env_steps,
                episode_idx=self.episode_idx,
            )
            self._last_checkpoint = ckpt

    def train(self) -> TrainResult:
        reset_seeds = [self._episode_reset_seed(i) for i in range(self.num_envs)]
        obs, infos = self.envs.reset(seed=reset_seeds)
        obs = np.asarray(obs, dtype=np.float32)
        masks = stack_valid_masks(infos, self.num_envs)
        self._last_checkpoint = self.run_dir / "checkpoint_final.pt"

        try:
            while self.env_steps < self.config.total_env_steps:
                self.epsilon = self._current_epsilon()
                actions, random_flags = self.agent.select_actions(
                    obs,
                    masks,
                    self.epsilon,
                    obs_scale=self.config.obs_scale,
                    generator=self.torch_generator,
                )
                for is_random in random_flags:
                    self.logger.note_action(is_random=bool(is_random))

                next_obs, rewards, terminations, truncations, next_infos = self.envs.step(actions)
                next_obs = np.asarray(next_obs, dtype=np.float32)
                next_masks = stack_valid_masks(next_infos, self.num_envs)
                true_next = true_next_observations(next_obs, terminations, truncations, next_infos)
                true_next_mask = true_next_masks(next_masks, terminations, truncations, next_infos)

                for i in range(self.num_envs):
                    if self.env_steps >= self.config.total_env_steps:
                        break
                    train_reward = transform_reward(float(rewards[i]), self.config.reward_mode)
                    transition = Transition(
                        state=obs[i].astype(np.float32, copy=True),
                        action=int(actions[i]),
                        reward=train_reward,
                        next_state=true_next[i].astype(np.float32, copy=True),
                        terminated=bool(terminations[i]),
                        truncated=bool(truncations[i]),
                        valid_mask=masks[i].astype(np.bool_, copy=True),
                        next_valid_mask=true_next_mask[i].astype(np.bool_, copy=True),
                    )
                    self._store_transition(i, transition)
                    self._after_env_step()

                    if terminations[i] or truncations[i]:
                        self._flush_n_step(i)
                        self.logger.note_episode_end(
                            length=int(episode_stat(next_infos, i, "episode_length")),
                            score=int(episode_stat(next_infos, i, "game_score")),
                            max_tile=int(episode_stat(next_infos, i, "max_tile")),
                            reached_2048=bool(episode_stat(next_infos, i, "reached_2048", False)),
                            truncated=bool(truncations[i]),
                        )
                        self.episode_idx += 1

                obs = next_obs
                masks = next_masks
        finally:
            self.envs.close()

        for i in range(self.num_envs):
            self._flush_n_step(i)
        final_checkpoint = self.run_dir / "checkpoint_final.pt"
        save_checkpoint(
            final_checkpoint,
            agent=self.agent,
            config=self.config,
            env_steps=self.env_steps,
            episode_idx=self.episode_idx,
        )
        self.logger.save_state()
        return TrainResult(
            run_dir=self.run_dir,
            env_steps=self.env_steps,
            episodes=self.episode_idx,
            checkpoint_path=final_checkpoint,
        )
