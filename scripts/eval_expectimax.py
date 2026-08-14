#!/usr/bin/env python3
"""C0/C0b: greedy DQN vs expectimax on Phase A checkpoints.

Smoke (200 val seeds) first; if search helps, continue to val 1000.
Supports resume: --expectimax-only --seeds 2 --max-steps 4000
Default search is 2-ply + corner tiebreak.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from rl2048.env import Game2048Env
from rl2048.eval.metrics import distribution_summary, tile_reach_probs, wilson_interval
from rl2048.eval.runner import evaluate_policy, run_episode, summary_to_dict
from rl2048.eval.seeds import val_seeds
from rl2048.policies.dqn_policy import DQNPolicy
from rl2048.policies.expectimax import ExpectimaxDQNPolicy

PHASE_A_ROOT = Path("/root/autodl-tmp/RL-2048/results/runs/opt_tile1024")
OUT_DIR = Path("/root/autodl-tmp/RL-2048/results/experiments")


def _tile_p(summary: dict, key: str) -> float:
    return float(summary.get("tile_reach_probs", {}).get(key, 0.0))


def latest_phase_a_ckpts() -> list[tuple[int, Path]]:
    runs = sorted(p for p in PHASE_A_ROOT.iterdir() if p.is_dir())
    found: list[tuple[int, Path]] = []
    for run in runs:
        try:
            seed = int(run.name.split("seed")[1].split("_")[0])
        except (IndexError, ValueError):
            continue
        ckpt = run / "checkpoint_final.pt"
        if ckpt.exists():
            found.append((seed, ckpt))
    found.sort()
    return found


def _print_row(label: str, summary: dict) -> None:
    print(
        f"  {label:<22} mean={summary['score_stats']['mean']:7.1f} "
        f"P512={100 * _tile_p(summary, 'P(>=512)'):5.1f}% "
        f"P1024={100 * summary['p_reach_1024']:5.2f}% "
        f"P2048={100 * summary['p_reach_2048']:5.2f}% "
        f"trunc={100 * summary['truncation_rate']:4.1f}%",
        flush=True,
    )


def _summarize_episodes(episodes, *, policy_key: str, policy_label: str) -> dict:
    scores = [e.game_score for e in episodes]
    max_tiles = [e.max_tile for e in episodes]
    lengths = [e.episode_length for e in episodes]
    successes_2048 = sum(e.reached_2048 for e in episodes)
    successes_1024 = sum(e.max_tile >= 1024 for e in episodes)
    ci_2048 = wilson_interval(successes_2048, len(episodes))
    from rl2048.eval.runner import PolicyEvalSummary

    summary = PolicyEvalSummary(
        policy_key=policy_key,
        policy_label=policy_label,
        episodes=len(episodes),
        p_reach_2048=ci_2048.rate,
        p_reach_2048_ci=(ci_2048.lower, ci_2048.upper),
        p_reach_1024=successes_1024 / len(episodes) if episodes else 0.0,
        tile_reach_probs=tile_reach_probs(max_tiles),
        score_stats=distribution_summary(scores),
        max_tile_stats=distribution_summary(max_tiles),
        length_stats=distribution_summary(lengths),
        truncation_rate=float(np.mean([e.truncated for e in episodes])) if episodes else 0.0,
        raw_episodes=episodes,
    )
    return summary_to_dict(summary)


def eval_policy(
    policy,
    *,
    label: str,
    seeds: list[int],
    key: str,
    max_steps: int,
    stop_on_2048: bool,
    progress_every: int = 0,
    partial_path: Path | None = None,
) -> dict:
    if progress_every <= 0:
        summary = evaluate_policy(
            policy,
            policy_key=key,
            policy_label=label,
            seeds=seeds,
            max_episode_steps=max_steps,
            stop_on_2048=stop_on_2048,
        )
        return summary_to_dict(summary)

    env = Game2048Env(max_episode_steps=max_steps)
    episodes = []
    done_seeds: set[int] = set()
    if partial_path is not None and partial_path.exists():
        raw = json.loads(partial_path.read_text(encoding="utf-8"))
        from rl2048.eval.runner import EpisodeResult

        for row in raw.get("episodes_raw", []):
            ep = EpisodeResult(**row)
            episodes.append(ep)
            done_seeds.add(int(ep.seed))
        print(f"  resumed {len(episodes)} episodes from {partial_path}", flush=True)

    remaining = [s for s in seeds if s not in done_seeds]
    for i, seed in enumerate(remaining, start=1):
        ep = run_episode(
            env,
            policy,
            seed=seed,
            policy_key=key,
            stop_on_2048=stop_on_2048,
        )
        episodes.append(ep)
        n = len(episodes)
        if n % progress_every == 0 or i == len(remaining):
            n2048 = sum(e.reached_2048 for e in episodes)
            n1024 = sum(e.max_tile >= 1024 for e in episodes)
            mean = float(np.mean([e.game_score for e in episodes]))
            print(
                f"  {label} {n}/{len(seeds)} mean={mean:.1f} "
                f"P1024={100 * n1024 / n:.2f}% P2048={100 * n2048 / n:.2f}%",
                flush=True,
            )
            if partial_path is not None:
                from dataclasses import asdict

                partial_path.parent.mkdir(parents=True, exist_ok=True)
                partial_path.write_text(
                    json.dumps(
                        {
                            "n": n,
                            "episodes_raw": [asdict(e) for e in episodes],
                        }
                    ),
                    encoding="utf-8",
                )
    return _summarize_episodes(episodes, policy_key=key, policy_label=label)


def promising(greedy: dict, search: dict) -> bool:
    """Search helps if P(2048) or P(1024) rises, or mean is clearly up."""
    d2048 = search["p_reach_2048"] - greedy["p_reach_2048"]
    d1024 = search["p_reach_1024"] - greedy["p_reach_1024"]
    dmean = search["score_stats"]["mean"] - greedy["score_stats"]["mean"]
    if d2048 > 0:
        return True
    if d1024 >= 0.02:
        return True
    if dmean >= 200:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy vs expectimax eval.")
    parser.add_argument("--smoke-episodes", type=int, default=200)
    parser.add_argument("--full-episodes", type=int, default=1000)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--force-full", action="store_true")
    parser.add_argument("--seeds", type=str, default="all", help="all or comma list e.g. 2")
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--stop-on-2048", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--expectimax-only", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--depth", type=int, default=2, help="Max expectimax depth (1 or 2).")
    parser.add_argument("--adaptive-depth", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--corner-tiebreak", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--corner-margin", type=float, default=2.0)
    args = parser.parse_args()

    ckpts = latest_phase_a_ckpts()
    if not ckpts:
        raise SystemExit(f"No Phase A checkpoints under {PHASE_A_ROOT}")
    if args.seeds != "all":
        want = {int(x) for x in args.seeds.split(",")}
        ckpts = [(s, p) for s, p in ckpts if s in want]
    print("Checkpoints:", flush=True)
    for seed, path in ckpts:
        print(f"  seed{seed}: {path}", flush=True)
    print(
        f"max_steps={args.max_steps} stop_on_2048={args.stop_on_2048} "
        f"expectimax_only={args.expectimax_only} depth={args.depth} "
        f"adaptive={args.adaptive_depth} corner={args.corner_tiebreak}",
        flush=True,
    )

    def make_expectimax(ckpt):
        return ExpectimaxDQNPolicy.from_checkpoint(
            ckpt,
            depth=args.depth,
            adaptive=args.adaptive_depth,
            corner_tiebreak=args.corner_tiebreak,
            corner_margin=args.corner_margin,
        )

    smoke_seeds = val_seeds(args.smoke_episodes)
    full_seeds = val_seeds(args.full_episodes)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / (
        f"expectimax_d{args.depth}_a{int(args.adaptive_depth)}"
        f"_c{int(args.corner_tiebreak)}.json"
    )
    payload: dict = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "max_episode_steps": args.max_steps,
        "stop_on_2048": args.stop_on_2048,
        "depth": args.depth,
        "adaptive_depth": args.adaptive_depth,
        "corner_tiebreak": args.corner_tiebreak,
        "corner_margin": args.corner_margin,
        "smoke": [],
        "full": [],
    }
    if out.exists():
        try:
            payload = json.loads(out.read_text(encoding="utf-8"))
            payload["timestamp"] = datetime.now(timezone.utc).isoformat()
            payload.setdefault("smoke", [])
            payload.setdefault("full", [])
        except json.JSONDecodeError:
            pass

    run_full = bool(args.force_full or args.skip_smoke)
    smoke_any_good = False

    def _eval(policy, label, seeds, key, partial_name=None):
        partial = OUT_DIR / partial_name if partial_name else None
        return eval_policy(
            policy,
            label=label,
            seeds=seeds,
            key=key,
            max_steps=args.max_steps,
            stop_on_2048=args.stop_on_2048,
            progress_every=args.progress_every,
            partial_path=partial,
        )

    if not args.skip_smoke:
        print(f"\n=== Smoke {args.smoke_episodes} val seeds ===", flush=True)
        for seed, ckpt in ckpts:
            print(f"\n-- Phase A seed {seed} --", flush=True)
            greedy = None
            if not args.expectimax_only:
                greedy = _eval(DQNPolicy.from_checkpoint(ckpt, decode="greedy"), f"greedy seed{seed}", smoke_seeds, "dqn")
                _print_row("greedy", greedy)
            search = _eval(
                make_expectimax(ckpt),
                f"expectimax seed{seed}",
                smoke_seeds,
                "dqn_expectimax",
            )
            _print_row("expectimax", search)
            good = True if greedy is None else promising(greedy, search)
            print(f"  promising={good}", flush=True)
            smoke_any_good = smoke_any_good or good
            payload["smoke"].append(
                {
                    "train_seed": seed,
                    "checkpoint": str(ckpt),
                    "greedy": greedy,
                    "expectimax": search,
                    "promising": good,
                }
            )
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        run_full = run_full or smoke_any_good
        if not run_full:
            print("\nSmoke did not show a lift; skipping full val. Use --force-full to run anyway.", flush=True)

    if run_full:
        print(f"\n=== Full val {args.full_episodes} ===", flush=True)
        for seed, ckpt in ckpts:
            print(f"\n-- Phase A seed {seed} --", flush=True)
            greedy = None
            if not args.expectimax_only:
                greedy = _eval(DQNPolicy.from_checkpoint(ckpt, decode="greedy"), f"greedy seed{seed}", full_seeds, "dqn")
                _print_row("greedy", greedy)
            search = _eval(
                make_expectimax(ckpt),
                f"expectimax seed{seed}",
                full_seeds,
                "dqn_expectimax",
                partial_name=f"expectimax_d{args.depth}_a{int(args.adaptive_depth)}_c{int(args.corner_tiebreak)}_seed{seed}_partial.json",
            )
            _print_row("expectimax", search)
            row = {
                "train_seed": seed,
                "checkpoint": str(ckpt),
                "greedy": greedy,
                "expectimax": search,
            }
            payload["full"] = [r for r in payload.get("full", []) if r.get("train_seed") != seed]
            payload["full"].append(row)
            payload["full"].sort(key=lambda r: r["train_seed"])
            out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"  checkpointed {out}", flush=True)

    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nSaved {out}", flush=True)


if __name__ == "__main__":
    main()
