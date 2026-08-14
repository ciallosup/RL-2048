#!/usr/bin/env python3
"""Summarize num_envs × PER ablation experiment JSONs + mid-run metrics."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

EXP_ROOT = Path("/root/autodl-tmp/RL-2048/results/experiments")
RUN_ROOT = Path("/root/autodl-tmp/RL-2048/results/runs")
CELLS = ("n1_p0", "n1_p1", "n8_p0", "n8_p1")


def _tile_p(eval_d: dict, key: str) -> float:
    return float(eval_d.get("tile_reach_probs", {}).get(key, 0.0))


def row_from_json(cell: str) -> str:
    path = EXP_ROOT / f"abl_{cell}_latest.json"
    if not path.exists():
        return f"{cell:<8} | {'MISSING':>10} | {'-':>8} | {'-':>8} | {'-':>8} | {'-':>8}"
    d = json.loads(path.read_text(encoding="utf-8"))
    cfg = d.get("config") or {}
    tag = f"n={cfg.get('num_envs', '?')} per={cfg.get('use_per', '?')}"
    a = d.get("dqn_aggregate") or {}
    if a:
        score = a["mean_score"]["mean"]
        p512 = 100 * a["p_ge_512"]["mean"]
        p1024 = 100 * a["p_reach_1024"]["mean"]
        p2048 = 100 * a["p_reach_2048"]["mean"]
        n = a.get("n_seeds", "?")
        return (
            f"{cell:<8} | {score:10.1f} | {p512:7.1f}% | {p1024:7.2f}% | {p2048:7.2f}% | "
            f"{tag} seeds={n}"
        )
    if d.get("runs"):
        e = d["runs"][0]["eval"]
        return (
            f"{cell:<8} | {e['score_stats']['mean']:10.1f} | "
            f"{100 * _tile_p(e, 'P(>=512)'):7.1f}% | "
            f"{100 * e['p_reach_1024']:7.2f}% | "
            f"{100 * e['p_reach_2048']:7.2f}% | {tag}"
        )
    return f"{cell:<8} | {'EMPTY':>10} | {'-':>8} | {'-':>8} | {'-':>8} | {tag}"


def latest_run_dir(cell: str) -> Path | None:
    root = RUN_ROOT / f"abl_{cell}"
    if not root.exists():
        return None
    runs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)
    return runs[-1] if runs else None


def midrun_line(cell: str) -> str:
    run = latest_run_dir(cell)
    if run is None:
        return f"{cell:<8} | no run dir yet"
    metrics = run / "metrics.jsonl"
    cfg_path = run / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
    if not metrics.exists():
        return f"{cell:<8} | {run.name} | metrics missing"
    rows = [json.loads(line) for line in metrics.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        return f"{cell:<8} | {run.name} | empty metrics"
    last = rows[-1]
    at_5m = next((r for r in rows if r.get("env_steps") == 5_000_000), None)
    pick = at_5m or last
    return (
        f"{cell:<8} | steps={pick.get('env_steps')} | "
        f"score≈{pick.get('mean_episode_score_recent')} | "
        f"eps={pick.get('epsilon'):.3f} | "
        f"n_envs={cfg.get('num_envs')} per={cfg.get('use_per')} | "
        f"{run.name}"
    )


def main() -> None:
    lines = [
        "=== Ablation eval (val 1000) ===",
        f"{'cell':<8} | {'mean_score':>10} | {'P(>=512)':>8} | {'P(1024)':>8} | {'P(2048)':>8} | note",
        "-" * 88,
    ]
    for cell in CELLS:
        lines.append(row_from_json(cell))
    lines += [
        "",
        "=== Latest mid-run train scores ===",
        "-" * 88,
    ]
    for cell in CELLS:
        lines.append(midrun_line(cell))
    text = "\n".join(lines) + "\n"
    out = EXP_ROOT / "ablation_nenv_per.txt"
    EXP_ROOT.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
