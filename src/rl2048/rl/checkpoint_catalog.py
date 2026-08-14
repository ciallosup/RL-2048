"""Discover and label trained DQN checkpoints for viz/eval."""

from __future__ import annotations

from pathlib import Path

import torch


def discover_checkpoints(
    root: Path | str = "results/runs",
    *,
    extra_roots: tuple[Path | str, ...] = ("checkpoints",),
    limit: int = 30,
) -> list[dict]:
    """Return recent checkpoints sorted by mtime (newest first)."""
    roots = [Path(root), *[Path(p) for p in extra_roots]]
    entries: list[dict] = []
    seen: set[str] = set()
    for root_path in roots:
        if not root_path.exists():
            continue
        patterns = ("checkpoint_*.pt", "*.pt") if root_path.name == "checkpoints" else ("checkpoint_*.pt",)
        for pattern in patterns:
            for path in root_path.rglob(pattern):
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                if path.suffix != ".pt":
                    continue
                seen.add(resolved)
                meta = _read_meta(path)
                entries.append(
                    {
                        "path": resolved,
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
    if path.parent.name == "checkpoints":
        run = path.stem
    parts = ["DQN"]
    if seed is not None:
        parts.append(f"seed={seed}")
    if step is not None:
        parts.append(f"{step}步")
    parts.append(f"({run})")
    return " ".join(parts)
