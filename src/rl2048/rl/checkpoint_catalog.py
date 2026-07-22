"""Discover and label trained DQN checkpoints for viz/eval."""

from __future__ import annotations

from pathlib import Path

import torch


def discover_checkpoints(root: Path | str = "results/runs", *, limit: int = 30) -> list[dict]:
    """Return recent checkpoints sorted by mtime (newest first)."""
    root_path = Path(root)
    if not root_path.exists():
        return []

    entries: list[dict] = []
    for path in root_path.rglob("checkpoint_*.pt"):
        if path.name == "checkpoint_final.pt" or path.stem.startswith("checkpoint_"):
            meta = _read_meta(path)
            entries.append(
                {
                    "path": str(path.resolve()),
                    "label": _format_label(path, meta),
                    "train_seed": meta.get("train_seed"),
                    "env_steps": meta.get("env_steps"),
                    "run_name": meta.get("run_name"),
                    "mtime": path.stat().st_mtime,
                }
            )

    entries.sort(key=lambda item: item["mtime"], reverse=True)
    return entries[:limit]


def _read_meta(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    config = payload.get("config", {})
    return {
        "train_seed": config.get("train_seed"),
        "env_steps": payload.get("env_steps"),
        "run_name": config.get("run_name"),
    }


def _format_label(path: Path, meta: dict) -> str:
    step = meta.get("env_steps")
    if step is None and path.stem.startswith("checkpoint_"):
        suffix = path.stem.removeprefix("checkpoint_")
        step = suffix if suffix.isdigit() else None
    seed = meta.get("train_seed")
    run = meta.get("run_name") or path.parent.name
    parts = ["DQN"]
    if seed is not None:
        parts.append(f"seed={seed}")
    if step is not None:
        parts.append(f"{step}步")
    parts.append(f"({run})")
    return " ".join(parts)
