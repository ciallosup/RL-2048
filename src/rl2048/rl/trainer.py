"""DQN training loop."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml

from rl2048.env import Game2048Env
from rl2048.rl.agent import DQNAgent
from rl2048.rl.buffer import ReplayBuffer, Transition
from rl2048.rl.checkpoint import save_checkpoint
from rl2048.rl.config import TrainConfig, resolve_device
from rl2048.rl.logger import MetricsLogger


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

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.run_dir = Path(config.output_dir) / f"{config.run_name}_seed{config.train_seed}_{timestamp}"
        self.logger = MetricsLogger(self.run_dir)
        (self.run_dir / "config.yaml").write_text(
            yaml.safe_dump(config.to_dict(), sort_keys=False),
            encoding="utf-8",
        )

        self.env = Game2048Env(max_episode_steps=config.max_episode_steps)
        self.agent = DQNAgent(
            hidden_dims=config.hidden_dims,
            device=self.device,
            gamma=config.gamma,
            lr=config.lr,
            use_double_dqn=config.use_double_dqn,
            huber_delta=config.huber_delta,
            grad_clip_norm=config.grad_clip_norm,
        )
        self.buffer = ReplayBuffer(config.replay_capacity)
        self.batch_size = config.resolved_batch_size(self.device)
        print(
            f"Training: device={self.device}, batch_size={self.batch_size}, run_dir={self.run_dir}",
            flush=True,
        )
        self.torch_generator = torch.Generator(device=self.device)
        self.torch_generator.manual_seed(config.train_seed + 17)

        self.env_steps = 0
        self.episode_idx = 0
        self.updates = 0
        self.epsilon = config.epsilon_start

    def _episode_reset_seed(self) -> int:
        return int(self.config.train_seed * 10_000 + self.episode_idx)

    def _current_epsilon(self) -> float:
        decay_steps = int(self.config.total_env_steps * self.config.epsilon_decay_fraction)
        if self.env_steps >= decay_steps:
            return self.config.epsilon_end
        progress = self.env_steps / max(decay_steps, 1)
        return self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)

    def train(self) -> TrainResult:
        obs, info = self.env.reset(seed=self._episode_reset_seed())
        last_checkpoint = self.run_dir / "checkpoint_final.pt"

        while self.env_steps < self.config.total_env_steps:
            self.epsilon = self._current_epsilon()
            mask = info["valid_action_mask"]
            action, is_random = self.agent.select_action(
                obs,
                mask,
                self.epsilon,
                obs_scale=self.config.obs_scale,
                generator=self.torch_generator,
            )
            self.logger.note_action(is_random=is_random)

            next_obs, reward, terminated, truncated, next_info = self.env.step(action)
            done = terminated or truncated

            # Store raw exponent obs; compute_loss / select_action scale once.
            transition = Transition(
                state=obs.astype(np.float32, copy=False),
                action=action,
                reward=reward,
                next_state=next_obs.astype(np.float32, copy=False),
                terminated=terminated,
                truncated=truncated,
                valid_mask=mask.astype(np.bool_),
                next_valid_mask=next_info["valid_action_mask"].astype(np.bool_),
            )
            self.buffer.push(transition)
            self.env_steps += 1
            self.logger.state.env_steps = self.env_steps

            if (
                self.env_steps >= self.config.learning_starts
                and self.env_steps % self.config.train_freq == 0
                and len(self.buffer) >= self.batch_size
            ):
                batch = self.buffer.sample(self.batch_size, self.rng)
                _, metrics = self.agent.compute_loss(batch, obs_scale=self.config.obs_scale)
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
                    }
                    self.logger.log_step(self.logger.snapshot(extra))

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
                last_checkpoint = ckpt

            obs, info = next_obs, next_info
            if done:
                self.logger.note_episode_end(
                    length=int(info["episode_length"]),
                    score=int(info["game_score"]),
                    max_tile=int(info["max_tile"]),
                    reached_2048=bool(info["reached_2048"]),
                    truncated=truncated,
                )
                self.episode_idx += 1
                obs, info = self.env.reset(seed=self._episode_reset_seed())

        # Always write a stable final path (periodic saves may overwrite last_checkpoint).
        final_path = self.run_dir / "checkpoint_final.pt"
        save_checkpoint(
            final_path,
            agent=self.agent,
            config=self.config,
            env_steps=self.env_steps,
            episode_idx=self.episode_idx,
        )
        last_checkpoint = final_path
        self.logger.save_state()
        return TrainResult(
            run_dir=self.run_dir,
            env_steps=self.env_steps,
            episodes=self.episode_idx,
            checkpoint_path=last_checkpoint,
        )
