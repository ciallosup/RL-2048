#!/usr/bin/env python3
"""Summarize beat-heuristic / tile1024 / tile2048 experiment JSONs."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path("/root/autodl-tmp/RL-2048/results/experiments")
FILES = {
    "opt_beat_heuristic": ROOT / "opt_beat_heuristic_latest.json",
    "opt_tile1024_smoke": ROOT / "opt_tile1024_smoke_latest.json",
    "opt_tile1024": ROOT / "opt_tile1024_latest.json",
    "opt_tile2048": ROOT / "opt_tile2048_latest.json",
    "opt_tile2048_v2": ROOT / "opt_tile2048_v2_latest.json",
}


def row(name: str, path: Path) -> str:
    if not path.exists():
        return f"{name:<22} | {'MISSING':>10} | {'-':>8} | {'-':>8} | {'-':>8} | {'-':>8}"
    d = json.loads(path.read_text())
    a = d.get("dqn_aggregate") or {}
    if not a and d.get("runs"):
        # single-run smoke style
        e = d["runs"][0]["eval"]
        score = e["score_stats"]["mean"]
        p512 = 100 * e["tile_reach_probs"].get("P(>=512)", 0.0)
        p1024 = 100 * e["p_reach_1024"]
        p2048 = 100 * e["p_reach_2048"]
        return f"{name:<22} | {score:10.1f} | {p512:7.1f}% | {p1024:7.2f}% | {p2048:7.2f}%"
    score = a["mean_score"]["mean"]
    p512 = 100 * a["p_ge_512"]["mean"]
    p1024 = 100 * a["p_reach_1024"]["mean"]
    p2048 = 100 * a["p_reach_2048"]["mean"]
    return (
        f"{name:<22} | {score:10.1f} | {p512:7.1f}% | {p1024:7.2f}% | {p2048:7.2f}% "
        f"(±{100*a['p_reach_1024']['std']:.2f}/±{100*a['p_reach_2048']['std']:.2f})"
    )


def main() -> None:
    lines = [
        f"{'run':<22} | {'mean_score':>10} | {'P(>=512)':>8} | {'P(1024)':>8} | {'P(2048)':>8}",
        "-" * 78,
    ]
    for name, path in FILES.items():
        lines.append(row(name, path))
    text = "\n".join(lines) + "\n"
    out = ROOT / "tile_horizon_comparison.txt"
    out.write_text(text, encoding="utf-8")
    print(text)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
