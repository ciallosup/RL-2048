"""Tkinter visualizer: pick a policy, watch 2048 play out."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, font as tkfont, ttk

from rl2048.core import ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT, ACTION_UP, ACTION_NAMES
from rl2048.env import Game2048Env
from rl2048.policies.base import Policy, PolicyContext
from rl2048.policies.dqn_policy import DQNPolicy
from rl2048.policies.manual import ManualPolicy
from rl2048.policies.registry import get_policy, list_policies
from rl2048.rl.checkpoint_catalog import discover_checkpoints
from rl2048.viz import colors

DEFAULT_STEP_MS = 350


class VisualizerApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("2048 DRL — 策略可视化")
        self.root.configure(bg=_hex(colors.BG_COLOR))
        self.root.minsize(860, 680)

        self.env = Game2048Env(max_episode_steps=500)
        self.ctx = PolicyContext(env=self.env)
        self.selected_policy_key = tk.StringVar(value="random")
        self.checkpoint_var = tk.StringVar(value="")
        self.policy: Policy = get_policy("random")
        self.checkpoint_path: str | None = None
        self.checkpoint_meta: dict | None = None
        self.auto_play = False
        self.game_over = False
        self.last_action: int | None = None
        self.last_reward = 0.0
        self.step_delay_ms = DEFAULT_STEP_MS
        self.episode_seed = 0
        self._after_id: str | None = None
        self._checkpoints = discover_checkpoints()
        self._checkpoint_by_label: dict[str, str] = {}

        self._build_ui()
        self.reset_game()
        self.root.bind("<Key>", self._on_key)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.title_font = tkfont.Font(family="Microsoft YaHei", size=20, weight="bold")
        self.label_font = tkfont.Font(family="Microsoft YaHei", size=11)
        self.tile_font_l = tkfont.Font(family="Arial", size=28, weight="bold")
        self.tile_font_s = tkfont.Font(family="Arial", size=18, weight="bold")

        top = tk.Frame(self.root, bg=_hex(colors.BG_COLOR))
        top.pack(fill=tk.X, padx=16, pady=(16, 8))

        tk.Label(
            top,
            text="2048 DRL",
            font=self.title_font,
            bg=_hex(colors.BG_COLOR),
            fg=_hex(colors.TEXT_DARK),
        ).pack(side=tk.LEFT)

        self.score_var = tk.StringVar(value="分数: 0")
        self.max_var = tk.StringVar(value="最大块: 0")
        for var in (self.score_var, self.max_var):
            tk.Label(
                top,
                textvariable=var,
                font=self.label_font,
                bg=_hex(colors.PANEL_COLOR),
                fg=_hex(colors.TEXT_DARK),
                padx=12,
                pady=6,
            ).pack(side=tk.LEFT, padx=8)

        body = tk.Frame(self.root, bg=_hex(colors.BG_COLOR))
        body.pack(fill=tk.BOTH, expand=True, padx=16, pady=8)

        board_frame = tk.Frame(body, bg=_hex(colors.GRID_COLOR), padx=8, pady=8)
        board_frame.pack(side=tk.LEFT)

        self.tile_labels: list[list[tk.Label]] = []
        for row in range(4):
            row_labels: list[tk.Label] = []
            row_frame = tk.Frame(board_frame, bg=_hex(colors.GRID_COLOR))
            row_frame.pack()
            for col in range(4):
                lbl = tk.Label(
                    row_frame,
                    width=8,
                    height=3,
                    font=self.tile_font_l,
                    bg=_hex(colors.tile_bg(0)),
                    fg=_hex(colors.tile_fg(2)),
                )
                lbl.grid(row=0, column=col, padx=4, pady=4)
                row_labels.append(lbl)
            self.tile_labels.append(row_labels)

        side = tk.Frame(body, bg=_hex(colors.BG_COLOR))
        side.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(24, 0))

        tk.Label(
            side,
            text="选择策略",
            font=self.label_font,
            bg=_hex(colors.BG_COLOR),
            fg=_hex(colors.TEXT_DARK),
        ).pack(anchor=tk.W)

        for key, label in list_policies():
            ttk.Radiobutton(
                side,
                text=label,
                value=key,
                variable=self.selected_policy_key,
                command=self._on_builtin_policy_change,
            ).pack(anchor=tk.W, pady=2)

        rl_frame = tk.LabelFrame(
            side,
            text="RL 模型 (checkpoint .pt)",
            bg=_hex(colors.BG_COLOR),
            fg=_hex(colors.TEXT_DARK),
            font=self.label_font,
        )
        rl_frame.pack(fill=tk.X, pady=(12, 8))

        ckpt_labels = [""] + [self._checkpoint_label(item) for item in self._checkpoints]
        self.checkpoint_combo = ttk.Combobox(
            rl_frame,
            textvariable=self.checkpoint_var,
            values=ckpt_labels,
            state="readonly",
            width=42,
        )
        self.checkpoint_combo.pack(fill=tk.X, padx=8, pady=(8, 4))
        self.checkpoint_combo.bind("<<ComboboxSelected>>", self._on_checkpoint_selected)

        btn_row_ckpt = tk.Frame(rl_frame, bg=_hex(colors.BG_COLOR))
        btn_row_ckpt.pack(fill=tk.X, padx=8, pady=(0, 8))
        ttk.Button(btn_row_ckpt, text="浏览 .pt...", command=self._browse_checkpoint).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row_ckpt, text="刷新列表", command=self._refresh_checkpoint_list).pack(side=tk.LEFT)
        ttk.Button(btn_row_ckpt, text="清除 RL", command=self._clear_checkpoint).pack(side=tk.LEFT, padx=(8, 0))

        self.checkpoint_info_var = tk.StringVar(value="未加载 checkpoint")
        tk.Label(
            rl_frame,
            textvariable=self.checkpoint_info_var,
            font=self.label_font,
            bg=_hex(colors.BG_COLOR),
            fg=_hex(colors.ACCENT),
            wraplength=320,
            justify=tk.LEFT,
        ).pack(anchor=tk.W, padx=8, pady=(0, 8))

        btn_row1 = tk.Frame(side, bg=_hex(colors.BG_COLOR))
        btn_row1.pack(fill=tk.X, pady=(8, 4))
        self.btn_start = ttk.Button(btn_row1, text="开始", command=self.start_auto)
        self.btn_start.pack(side=tk.LEFT, padx=(0, 8))
        self.btn_pause = ttk.Button(btn_row1, text="暂停", command=self.pause_auto, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT)

        btn_row2 = tk.Frame(side, bg=_hex(colors.BG_COLOR))
        btn_row2.pack(fill=tk.X, pady=4)
        ttk.Button(btn_row2, text="单步", command=self.do_step).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row2, text="重置", command=self.reset_game).pack(side=tk.LEFT)

        btn_row3 = tk.Frame(side, bg=_hex(colors.BG_COLOR))
        btn_row3.pack(fill=tk.X, pady=4)
        ttk.Button(btn_row3, text="加速", command=lambda: self._adjust_speed(-50)).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(btn_row3, text="减速", command=lambda: self._adjust_speed(50)).pack(side=tk.LEFT)

        self.status_var = tk.StringVar()
        tk.Label(
            side,
            textvariable=self.status_var,
            font=self.label_font,
            bg=_hex(colors.BG_COLOR),
            fg=_hex(colors.TEXT_DARK),
            justify=tk.LEFT,
        ).pack(anchor=tk.W, pady=(16, 0))

        tk.Label(
            side,
            text="快捷键: Space 开始/暂停 | R 重置 | N 单步 | 方向键/WASD 人工模式",
            font=self.label_font,
            bg=_hex(colors.BG_COLOR),
            fg=_hex(colors.ACCENT),
            wraplength=320,
            justify=tk.LEFT,
        ).pack(side=tk.BOTTOM, anchor=tk.W, pady=(12, 0))

    def _checkpoint_label(self, item: dict) -> str:
        label = item.get("label") or Path(item["path"]).name
        self._checkpoint_by_label[label] = item["path"]
        return label

    def _refresh_checkpoint_list(self) -> None:
        self._checkpoints = discover_checkpoints()
        self._checkpoint_by_label.clear()
        values = [""] + [self._checkpoint_label(item) for item in self._checkpoints]
        self.checkpoint_combo["values"] = values

    def _browse_checkpoint(self) -> None:
        path = filedialog.askopenfilename(
            title="选择 DQN checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*.*")],
            initialdir=str(Path("results/runs").resolve()) if Path("results/runs").exists() else ".",
        )
        if path:
            self._load_checkpoint(path)

    def _on_checkpoint_selected(self, _event: object | None = None) -> None:
        label = self.checkpoint_var.get().strip()
        path = self._checkpoint_by_label.get(label, label)
        if path:
            self._load_checkpoint(path)
        else:
            self._clear_checkpoint()

    def _load_checkpoint(self, path: str) -> None:
        ckpt_path = str(Path(path).resolve())
        self.checkpoint_path = ckpt_path
        self.policy = DQNPolicy.from_checkpoint(ckpt_path)
        meta = next((item for item in self._checkpoints if item["path"] == ckpt_path), None)
        if meta is None:
            meta = next((item for item in discover_checkpoints(limit=200) if item["path"] == ckpt_path), {})
        self.checkpoint_meta = meta or {
            "train_seed": self.policy.config.train_seed,
            "env_steps": self.policy.meta.get("env_steps"),
            "run_name": self.policy.config.run_name,
        }
        seed = self.checkpoint_meta.get("train_seed")
        steps = self.checkpoint_meta.get("env_steps")
        run = self.checkpoint_meta.get("run_name", Path(ckpt_path).parent.name)
        label = (meta or {}).get("label") or Path(ckpt_path).name
        self.checkpoint_var.set(label)
        self._checkpoint_by_label[label] = ckpt_path
        self.checkpoint_info_var.set(
            f"已加载: {Path(ckpt_path).name}\n训练 seed={seed}, 步数={steps}, run={run}"
        )
        self.reset_game()

    def _clear_checkpoint(self) -> None:
        self.checkpoint_path = None
        self.checkpoint_meta = None
        self.checkpoint_var.set("")
        self.checkpoint_info_var.set("未加载 checkpoint")
        self._on_builtin_policy_change()

    def _on_builtin_policy_change(self) -> None:
        if self.checkpoint_path:
            return
        was_auto = self.auto_play
        key = self.selected_policy_key.get()
        self.policy = get_policy(key)
        self.reset_game()
        if key != "manual" and was_auto:
            self.start_auto()

    def _adjust_speed(self, delta: int) -> None:
        self.step_delay_ms = int(min(1500, max(50, self.step_delay_ms + delta)))
        self._refresh_status()

    def reset_game(self) -> None:
        self._cancel_timer()
        import time

        self.episode_seed = int(time.time() * 1000) % 1_000_000
        obs, info = self.env.reset(seed=self.episode_seed)
        self.ctx.obs = obs
        self.ctx.info = info
        self.ctx.done = False
        self.policy.reset(self.ctx)
        self.auto_play = False
        self.game_over = False
        self.last_action = None
        self.last_reward = 0.0
        self.btn_start.state(["!disabled"])
        self.btn_pause.state(["disabled"])
        self._refresh_board()
        self._refresh_status()

    def start_auto(self) -> None:
        if self.game_over:
            return
        if self.checkpoint_path is None and self.selected_policy_key.get() == "manual":
            return
        self.auto_play = True
        self.btn_start.state(["disabled"])
        self.btn_pause.state(["!disabled"])
        self._schedule_step()

    def pause_auto(self) -> None:
        self.auto_play = False
        self._cancel_timer()
        self.btn_start.state(["!disabled"])
        self.btn_pause.state(["disabled"])
        self._refresh_status()

    def _cancel_timer(self) -> None:
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
            self._after_id = None

    def _schedule_step(self) -> None:
        if self.auto_play and not self.game_over:
            self._after_id = self.root.after(self.step_delay_ms, self._auto_tick)

    def _auto_tick(self) -> None:
        self._after_id = None
        if self.auto_play and not self.game_over:
            self.do_step()
            self._schedule_step()

    def do_step(self) -> None:
        if self.game_over:
            return
        action = self.policy.select_action(self.ctx)
        if action is None:
            return
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.ctx.obs = obs
        self.ctx.info = info
        self.ctx.done = terminated or truncated
        self.last_action = action
        self.last_reward = reward
        if self.ctx.done:
            self.game_over = True
            self.pause_auto()
        self._refresh_board()
        self._refresh_status()

    def _refresh_board(self) -> None:
        board = self.env.board
        for row in range(4):
            for col in range(4):
                value = int(board[row, col])
                lbl = self.tile_labels[row][col]
                lbl.configure(
                    text=str(value) if value else "",
                    bg=_hex(colors.tile_bg(value)),
                    fg=_hex(colors.tile_fg(value) if value else colors.TEXT_DARK),
                    font=self.tile_font_s if value >= 1024 else self.tile_font_l,
                )
        self.score_var.set(f"分数: {self.ctx.info.get('game_score', 0)}")
        self.max_var.set(f"最大块: {self.ctx.info.get('max_tile', 0)}")

    def _active_policy_label(self) -> str:
        if self.checkpoint_path:
            return f"RL: {Path(self.checkpoint_path).name}"
        return self.selected_policy_key.get()

    def _refresh_status(self) -> None:
        lines = [
            f"策略: {self._active_policy_label()}",
            f"种子: {self.episode_seed}",
            f"步数: {self.ctx.info.get('episode_length', 0)}",
            f"速度: {self.step_delay_ms} ms/步",
            f"到达 2048: {'是' if self.ctx.info.get('reached_2048') else '否'}",
        ]
        if self.last_action is not None:
            lines.append(f"上一步: {ACTION_NAMES[self.last_action]} (+{int(self.last_reward)})")
        if self.game_over:
            lines.append("状态: 游戏结束")
        elif self.auto_play:
            lines.append("状态: 自动运行中")
        elif self.checkpoint_path is None and self.selected_policy_key.get() == "manual":
            lines.append("状态: 人工 — 方向键 / WASD")
        else:
            lines.append("状态: 就绪 — 点击「开始」")
        self.status_var.set("\n".join(lines))

    def _on_key(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if key == "space":
            if self.auto_play:
                self.pause_auto()
            else:
                self.start_auto()
        elif key == "r":
            self.reset_game()
        elif key == "n":
            self.do_step()
        elif self.checkpoint_path is None and isinstance(self.policy, ManualPolicy):
            action_map = {
                "up": ACTION_UP,
                "w": ACTION_UP,
                "down": ACTION_DOWN,
                "s": ACTION_DOWN,
                "left": ACTION_LEFT,
                "a": ACTION_LEFT,
                "right": ACTION_RIGHT,
                "d": ACTION_RIGHT,
            }
            if key in action_map:
                self.policy.queue_action(action_map[key])
                self.do_step()

    def _on_close(self) -> None:
        self._cancel_timer()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def _hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def main() -> None:
    VisualizerApp().run()


if __name__ == "__main__":
    main()
