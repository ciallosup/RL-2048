"""Roll out baseline / RL policies and write README demo GIFs."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl2048.env import Game2048Env
from rl2048.policies.base import Policy, PolicyContext
from rl2048.policies.registry import get_policy
from rl2048.viz.board_image import write_gif

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
DEFAULT_CHECKPOINT = ROOT / "checkpoints" / "phaseA_dueling_seed0.pt"


def rollout(
    policy: Policy,
    *,
    seed: int,
    max_steps: int,
    stop_tile: int | None = None,
) -> tuple[list[np.ndarray], dict]:
    env = Game2048Env(max_episode_steps=max_steps)
    obs, info = env.reset(seed=seed)
    ctx = PolicyContext(env=env, obs=obs, info=info)
    policy.reset(ctx)
    boards = [env.board.copy()]
    while True:
        action = policy.select_action(ctx)
        if action is None:
            break
        obs, _reward, terminated, truncated, info = env.step(action)
        ctx.obs = obs
        ctx.info = info
        ctx.done = terminated or truncated
        boards.append(env.board.copy())
        if stop_tile is not None and int(info.get("max_tile", 0)) >= stop_tile:
            break
        if terminated or truncated:
            break
    return boards, info


def _rl_policy(checkpoint: Path, decode: str) -> Policy:
    from rl2048.policies.dqn_policy import DQNPolicy

    return DQNPolicy.from_checkpoint(checkpoint, decode=decode)


def export_demos(args: argparse.Namespace) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)

    random_boards, random_info = rollout(get_policy("random"), seed=args.random_seed, max_steps=200)
    write_gif(
        random_boards,
        ASSETS / "demo_random.gif",
        caption=f"随机  ·  最大块 {int(random_info.get('max_tile', 0))}",
        stride=1,
        duration_ms=110,
    )
    print(f"random: {len(random_boards)} frames, max_tile={random_info.get('max_tile')}")

    heuristic_boards, heuristic_info = rollout(
        get_policy("heuristic"),
        seed=args.heuristic_seed,
        max_steps=280,
        stop_tile=512,
    )
    write_gif(
        heuristic_boards,
        ASSETS / "demo_heuristic.gif",
        caption=f"启发式  ·  最大块 {int(heuristic_info.get('max_tile', 0))}",
        stride=2,
        duration_ms=90,
    )
    print(f"heuristic: {len(heuristic_boards)} frames, max_tile={heuristic_info.get('max_tile')}")

    if args.skip_rl:
        return
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    rl_policy = _rl_policy(checkpoint, args.decode)
    rl_boards, rl_info = rollout(
        rl_policy,
        seed=args.rl_seed,
        max_steps=args.rl_max_steps,
        stop_tile=args.stop_tile,
    )
    write_gif(
        rl_boards,
        ASSETS / "demo_rl.gif",
        caption=f"RL {args.decode}  ·  最大块 {int(rl_info.get('max_tile', 0))}",
        stride=args.rl_stride,
        duration_ms=70,
        last_hold_ms=1600,
    )
    print(f"rl: {len(rl_boards)} frames, max_tile={rl_info.get('max_tile')}")


def export_screenshot(args: argparse.Namespace) -> None:
    import ctypes

    from PIL import ImageGrab

    from rl2048.policies.dqn_policy import DECODE_GREEDY
    from rl2048.viz.app import VisualizerApp

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass

    app = VisualizerApp()
    app.root.geometry("1320x960")
    app.root.lift()
    app.root.attributes("-topmost", True)
    app.root.update()
    checkpoint = Path(args.checkpoint)
    if checkpoint.exists():
        app.rl_decode_var.set(DECODE_GREEDY)
        app._load_checkpoint(str(checkpoint))
        app.root.update()
        for _ in range(28):
            app.do_step()
            app.root.update()
    else:
        app.selected_policy_key.set("heuristic")
        app._on_policy_change()
        for _ in range(36):
            app.do_step()
            app.root.update()
    app.root.update_idletasks()
    app.root.update()
    import time

    time.sleep(0.25)
    app.root.update()
    hwnd = int(app.root.winfo_id())
    x = app.root.winfo_rootx()
    y = app.root.winfo_rooty()
    w = app.root.winfo_width()
    h = app.root.winfo_height()
    try:
        dpi = ctypes.windll.user32.GetDpiForWindow(hwnd)
        scale = dpi / 96.0 if dpi else 1.0
    except Exception:
        scale = 1.0
    # If the process is already DPI-aware, Tk coordinates are physical pixels.
    bbox = (x, y, x + w, y + h)
    image = ImageGrab.grab(bbox=bbox)
    if image.size[0] < w * 0.8:
        bbox = (int(x * scale), int(y * scale), int((x + w) * scale), int((y + h) * scale))
        image = ImageGrab.grab(bbox=bbox)
    out = ASSETS / "viz_ui.png"
    ASSETS.mkdir(parents=True, exist_ok=True)
    image.save(out)
    print(f"screenshot: {out} ({image.size[0]}x{image.size[1]}) bbox={bbox}")
    app.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--decode", default="2ply")
    parser.add_argument("--random-seed", type=int, default=12)
    parser.add_argument("--heuristic-seed", type=int, default=27)
    parser.add_argument("--rl-seed", type=int, default=0)
    parser.add_argument("--rl-max-steps", type=int, default=900)
    parser.add_argument("--rl-stride", type=int, default=4)
    parser.add_argument("--stop-tile", type=int, default=1024)
    parser.add_argument("--skip-rl", action="store_true")
    parser.add_argument("--screenshot", action="store_true")
    args = parser.parse_args()
    if args.screenshot:
        export_screenshot(args)
        return
    export_demos(args)


if __name__ == "__main__":
    main()
