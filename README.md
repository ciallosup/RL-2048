# RL-2048

用深度强化学习下 2048，并提供一套**可复现、可对照**的评测与可视化接口：同一规则、同一固定种子集，把强化学习策略和随机、启发式、一步贪心等传统方法放在一起比较。

当前推荐用法：加载发布权重，默认用 **2-ply expectimax** 推理（可视化里也可切到贪心 Q 或 1-ply）。过程记录见 [`docs/updates.md`](docs/updates.md)。

## 结果对照

同一套标准 2048 规则（到达 2048 **不结束**）。主指标是 **P(2048)**，辅看均分与 P(1024)。

| 方法 | 均分 | P(1024) | P(2048) | 说明 |
|------|-----:|--------:|--------:|------|
| 随机 | 1097 | 0% | 0% | val 1000 |
| 一步贪心（即时合并分） | 3109 | 0% | 0% | val 1000 |
| 启发式（空格 / 单调 / 平滑 / 角落） | ~3850 | 1.8% | 0% | val 1000 |
| E1 MLP Double DQN | 2578 | — | 0% | val 1000，未过启发式 |
| Phase A Dueling CNN（贪心 Q） | 6123 | 16.9% | 0.27% | val 1000，3 seeds |
| Phase A + 1-ply expectimax | ~15820 | 82.6% | 26.4% | val 1000，play-out |
| **Phase A + 2-ply expectimax** | **20754** | **99.5%** | **93.5%** | seed0，val 200，摸到 2048 即停 |

发布权重：[`checkpoints/phaseA_dueling_seed0.pt`](checkpoints/README.md)（Dueling CNN，5M steps，已去掉 optimizer）。2-ply 为默认推理；贪心 Q 仍远低于启发式之后的搜索策略，说明网络主要提供局面价值，残局靠 expectimax。

评测脚本会报告 P(2048) 的 Wilson 95% CI、P(max≥2^k) 曲线、分数 / 最大块 / 步数分布。种子池在 `data/seeds/`（git 忽略，运行评估时生成）。

## 安装

依赖装在项目目录内，不污染系统 Python。需要 Python 3.10+；可视化需要 **tkinter**（Miniconda / 官方安装一般自带）。

**Windows（venv，推荐）**

```powershell
.\scripts\setup_env.ps1
.\.venv\Scripts\Activate.ps1
pytest
```

**Windows（Conda，项目内 prefix）**

```powershell
.\scripts\setup_env.ps1 -Mode conda
conda activate ./.conda/env
```

**Linux**

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

训练再装：`pip install -e ".[dev,train]"`。云 GPU（AutoDL）见 [`docs/autodl.md`](docs/autodl.md)，不要把训练产物写进系统盘。

## 可视化（对照界面）

```powershell
.\.venv\Scripts\Activate.ps1
rl2048-play
```

加载 `checkpoints/phaseA_dueling_seed0.pt` 后默认 **2-ply**，可在右侧切换 **贪心 Q / 1-ply / 2-ply**。基线策略（随机、启发式、一步贪心、固定优先级、人工）无需 checkpoint。

| 操作 | 功能 |
|------|------|
| 右侧面板 | 选择基线；加载模型后选择推理方式 |
| 开始 / 暂停 | 自动按当前策略下棋 |
| 单步 | 前进一步 |
| 重置 | 新局（新种子） |
| 加速 / 减速 | 自动步进间隔 |
| 方向键 / WASD | 人工策略下移动 |
| Space | 开始 / 暂停 |
| R / N | 重置 / 单步 |

## 命令行评估

```powershell
rl2048-eval                                 # 基线，dev 200 局
rl2048-eval --episodes 1000 --seed-set val  # 基线，val 1000
rl2048-eval --checkpoint checkpoints/phaseA_dueling_seed0.pt --episodes 200 --seed-set val
# 默认 2-ply。只量网络本身：加 --decode greedy --max-steps 1200
```

专门对照 1-ply / 2-ply：

```powershell
python scripts/eval_expectimax.py --skip-smoke --force-full --seeds 0 --max-steps 4000 --stop-on-2048 --expectimax-only
```

## 训练（可选）

```powershell
rl2048-train --config configs/dqn_baseline.yaml
rl2048-train --config configs/autodl/opt_tile1024.yaml --train-seed 0
```

当前最强权重对应 `configs/autodl/opt_tile1024.yaml`：one-hot `C×4×4`、Dueling CNN、D4 增强、`n_step=5`、`gamma=0.995`、训练奖励 `log1p`。PER 已实现但默认关闭（与 D4 冲突，见 [`docs/per_notes.md`](docs/per_notes.md)）。实验沿革见 [`docs/updates.md`](docs/updates.md)。

## 规则（评测协议）

| 项目 | 定义 |
|------|------|
| 目标 | 存活、累积分数、做更大块；到达 2048 **不终止** |
| 观察 | 4×4，指数编码（空=0，2/4/8…=1/2/3…） |
| 动作 | 上 / 下 / 左 / 右；仅棋盘变化后才生成新块 |
| 生成 | 空格均匀抽样；2 的概率 0.9，4 的概率 0.1 |
| 终止 | 没有合法移动 |
| 截断 | 达到 `max_episode_steps` 时 `truncated=True` |
| 成功 | 首次出现 ≥2048，记在 `info['reached_2048']` |

```python
from rl2048 import Game2048Env

env = Game2048Env(max_episode_steps=4000)
obs, info = env.reset(seed=42)
obs, reward, terminated, truncated, info = env.step(action)
```

新策略实现 `Policy` 协议并注册到 `rl2048.policies.registry.POLICY_REGISTRY`，即可出现在可视化里。

## 文档

- [`docs/updates.md`](docs/updates.md) — 算法栈、对照数字、走过的弯路
- [`docs/autodl.md`](docs/autodl.md) — AutoDL 云上训练
- [`docs/per_notes.md`](docs/per_notes.md) — 为何默认关掉 PER
- [`checkpoints/README.md`](checkpoints/README.md) — 发布权重
