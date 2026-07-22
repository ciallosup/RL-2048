"""Training metrics logger."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LoggerState:
    env_steps: int = 0
    episodes: int = 0
    updates: int = 0
    random_action_count: int = 0
    action_count: int = 0
    truncation_count: int = 0
    episode_lengths: list[int] = field(default_factory=list)
    episode_scores: list[int] = field(default_factory=list)
    episode_max_tiles: list[int] = field(default_factory=list)
    reached_2048_count: int = 0


class MetricsLogger:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.state = LoggerState()

    def log_step(self, payload: dict[str, Any]) -> None:
        record = {"env_steps": self.state.env_steps, **payload}
        with self.metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def note_action(self, *, is_random: bool) -> None:
        self.state.action_count += 1
        if is_random:
            self.state.random_action_count += 1

    def note_episode_end(
        self,
        *,
        length: int,
        score: int,
        max_tile: int,
        reached_2048: bool,
        truncated: bool,
    ) -> None:
        self.state.episodes += 1
        self.state.episode_lengths.append(length)
        self.state.episode_scores.append(score)
        self.state.episode_max_tiles.append(max_tile)
        if reached_2048:
            self.state.reached_2048_count += 1
        if truncated:
            self.state.truncation_count += 1

    def snapshot(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        episodes = max(self.state.episodes, 1)
        recent_scores = self.state.episode_scores[-100:]
        recent_lengths = self.state.episode_lengths[-100:]
        payload: dict[str, Any] = {
            "episodes": self.state.episodes,
            "updates": self.state.updates,
            "random_action_ratio": self.state.random_action_count / max(self.state.action_count, 1),
            "truncation_rate": self.state.truncation_count / episodes,
            "mean_episode_score_recent": sum(recent_scores) / max(len(recent_scores), 1),
            "mean_episode_length_recent": sum(recent_lengths) / max(len(recent_lengths), 1),
            "reached_2048_rate": self.state.reached_2048_count / episodes,
        }
        if extra:
            payload.update(extra)
        return payload

    def save_state(self) -> None:
        path = self.run_dir / "logger_state.json"
        path.write_text(json.dumps(asdict(self.state), indent=2), encoding="utf-8")
