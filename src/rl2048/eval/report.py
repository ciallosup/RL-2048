from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from rl2048.eval.runner import PolicyEvalSummary


def format_percent(value: float) -> str:
    return f"{100.0 * value:5.1f}%"


def print_baseline_table(summaries: Iterable[PolicyEvalSummary]) -> None:
    summaries = list(summaries)
    headers = ["策略", "P(2048)", "95% CI", "P(1024)", "均分", "分位P95", "均最大块", "均步数", "截断率"]
    rows: list[list[str]] = []
    for s in summaries:
        rows.append(
            [
                s.policy_label,
                format_percent(s.p_reach_2048),
                f"[{format_percent(s.p_reach_2048_ci[0])}, {format_percent(s.p_reach_2048_ci[1])}]",
                format_percent(s.p_reach_1024),
                f"{s.score_stats['mean']:.0f}",
                f"{s.score_stats['p95']:.0f}",
                f"{s.max_tile_stats['mean']:.0f}",
                f"{s.length_stats['mean']:.0f}",
                format_percent(s.truncation_rate),
            ]
        )

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def print_tile_curves(summaries: Iterable[PolicyEvalSummary]) -> None:
    print("\n最大方块达到概率曲线 P(max >= 2^k):")
    summaries = list(summaries)
    keys = list(summaries[0].tile_reach_probs.keys()) if summaries else []
    header = ["策略"] + keys
    print(" | ".join(header))
    for s in summaries:
        vals = [format_percent(s.tile_reach_probs[k]) for k in keys]
        print(" | ".join([s.policy_key] + vals))


def check_e0_gate(random_summary: PolicyEvalSummary, heuristic_summary: PolicyEvalSummary) -> dict:
    """
    Roadmap E0 gate: heuristic clearly better than random.

    On small dev sets P(2048) is often 0 for both policies; use mean score and
    mean max tile as practical checks when P(2048) is sparse.
    """
    score_better = heuristic_summary.score_stats["mean"] > random_summary.score_stats["mean"]
    max_tile_better = heuristic_summary.max_tile_stats["mean"] > random_summary.max_tile_stats["mean"]
    p2048_better = heuristic_summary.p_reach_2048 >= random_summary.p_reach_2048
    h_low, _ = heuristic_summary.p_reach_2048_ci
    _, r_high = random_summary.p_reach_2048_ci
    ci_separated = h_low > r_high

    if heuristic_summary.p_reach_2048 == 0 and random_summary.p_reach_2048 == 0:
        passed = score_better and max_tile_better
    else:
        passed = p2048_better and score_better

    return {
        "passed": passed,
        "heuristic_p2048": heuristic_summary.p_reach_2048,
        "random_p2048": random_summary.p_reach_2048,
        "heuristic_mean_score": heuristic_summary.score_stats["mean"],
        "random_mean_score": random_summary.score_stats["mean"],
        "heuristic_mean_max_tile": heuristic_summary.max_tile_stats["mean"],
        "random_mean_max_tile": random_summary.max_tile_stats["mean"],
        "ci_separated": ci_separated,
        "message": "E0 通过：启发式优于随机" if passed else "E0 未通过：需检查环境或启发式权重",
    }


def save_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
