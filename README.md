# 2048 深度强化学习实验



依据 `2048_DRL_实验路线图.pdf` 实现的 2048 DRL 项目。当前进度：**第 1–5 部分（Masked Double DQN 基线）**。



## 环境安装（项目内虚拟环境）



所有依赖安装在项目目录内，不污染系统 Python。二选一：



### 方式 A：venv（推荐，Windows）



```powershell
.\scripts\setup_env.ps1
# 脚本会优先选用带 tkinter 的 Python（如 Miniconda）
.\.venv\Scripts\Activate.ps1
```



### 方式 B：Conda（项目内 prefix）



```powershell

.\scripts\setup_env.ps1 -Mode conda

# 或手动：

conda env create -f environment.yml -p .conda/env

conda activate ./.conda/env

```



激活后运行测试：`pytest`



## 可视化界面



图形化观察各策略的棋盘与移动（随机、贪心、固定优先级、人工；后续 RL 策略可注册到同一接口）。

> 需要 Python 含 **tkinter**（Miniconda / 官方完整安装通常可用；部分精简 Python 3.14 不含）。

```powershell
.\.venv\Scripts\Activate.ps1
rl2048-play
```



**操作说明：**



| 操作 | 功能 |

|------|------|

| 右侧面板 | 选择策略 |

| 开始 / 暂停 | 自动按策略下棋 |

| 单步 | 手动前进一步 |

| 重置 | 新局（新种子） |

| 加速 / 减速 | 调整自动步进间隔 |

| 方向键 / WASD | 人工策略下移动 |

| Space | 开始/暂停 |

| R / N | 重置 / 单步 |

## 基线策略与评估（路线图 §4）

已实现四条非神经网络基线，并通过固定种子集评估：

| 策略 key | 说明 |
|----------|------|
| `random` | 合法动作均匀随机 |
| `fixed` | 固定优先级：左→下→右→上 |
| `heuristic` | 空格 / 单调性 / 平滑度 / 角落加权 |
| `greedy` | Oracle 一步贪心（即时合并分最高） |

```powershell
.\.venv\Scripts\Activate.ps1
rl2048-eval                          # 默认 dev 200 局
rl2048-eval --episodes 1000 --seed-set val   # 更大验证集
```

**报告指标：** P(2048) + Wilson 95% CI、P(1024)、P(max≥2^k) 曲线、分数/最大块/步数分布、截断率。

结果保存至 `results/baselines/latest.json`，种子池保存至 `data/seeds/`。

**E0 验收（200 局 dev）：** 启发式均分与均最大块显著优于随机 → 可进入 DQN 阶段。

## Masked Double DQN 训练（路线图 §5）

安装训练依赖：

```powershell
pip install -e ".[dev,train]"
```

训练：

```powershell
rl2048-train --config configs/dqn_baseline.yaml
rl2048-train --config configs/experiments/e1_smoke.yaml --train-seed 0   # 100k 冒烟
```

评估 checkpoint（dev/val 固定种子）：

```powershell
rl2048-eval --checkpoint results/runs/.../checkpoint_final.pt --episodes 200 --seed-set dev
```

多 seed 实验（E1/E2）：

```powershell
python scripts/run_experiment.py --config configs/experiments/e1_min_dqn.yaml --train-seeds 5
python scripts/run_experiment.py --config configs/experiments/e2_vanilla_dqn.yaml --train-seeds 5
```

可视化已训练策略：

```powershell
$env:RL2048_CHECKPOINT = "results/runs/.../checkpoint_final.pt"
rl2048-play   # 策略列表中出现 DQN (checkpoint)
```

**默认超参：** Double DQN、`gamma=0.99`、MLP `[256,256]`、raw merge reward、masked ε-greedy、truncated 仍 bootstrap。`use_double_dqn: false` 切换为 vanilla DQN（E2 对照）。

## 已冻结的任务定义（路线图 §2）



| 项目 | 定义 |

|------|------|

| 任务目标 | 存活尽可能久、累积更高分、制造更大方块；达到 2048 **不终止** |

| 观察 | 4×4 棋盘，指数编码（空=0，2/4/8…=1/2/3…），展平为 16 维向量 |

| 动作 | 上/下/左/右；**仅有效动作**（棋盘发生变化）后生成新方块 |

| 生成 | 空格均匀抽样；2 概率 0.9，4 概率 0.1 |

| 自然终止 | 不存在任何能改变棋盘的动作 |

| 截断 | 达到 `max_episode_steps` 时 `truncated=True`，`terminated=False` |

| 成功事件 | 首次出现 ≥2048 的方块，记录在 `info['reached_2048']` |



## 环境 API（路线图 §3.1）



```python

from rl2048 import Game2048Env



env = Game2048Env(max_episode_steps=500)  # 建议 ≥436（3000 局校准 P99×1.1）

obs, info = env.reset(seed=42)

obs, reward, terminated, truncated, info = env.step(action)

```



## 策略扩展（供 RL 接入）



实现 `Policy` 协议并注册到 `rl2048.policies.registry.POLICY_REGISTRY` 即可出现在可视化下拉中：



```python

from rl2048.policies.registry import POLICY_REGISTRY



class MyRLPolicy:

    key = "rl"

    label = "我的 DQN"

    def reset(self, ctx): ...

    def select_action(self, ctx) -> int: ...



POLICY_REGISTRY["rl"] = MyRLPolicy

```



## 目录结构



```
src/rl2048/
  core.py, env.py, symmetry.py
  policies/     随机 / 启发式 / 贪心 / 固定 / DQN + 注册表
  rl/           DQN agent、replay、trainer、checkpoint
  eval/         评估 runner、指标、报告
  heuristic/    棋盘启发式特征
  viz/          Tkinter 可视化
configs/        dqn_baseline.yaml, experiments/e1_*, e2_*
tests/
scripts/        train_dqn, run_experiment, eval_baselines, setup_env
results/        训练 run 与评估 JSON（git 忽略）
data/seeds/     固定评估种子池（git 忽略）
.venv/ 或 .conda/env/
```

## AutoDL 云 GPU 训练

详见 [docs/autodl.md](docs/autodl.md)。简要流程：

1. 推送到 GitHub → AutoDL 上 `git clone` 到 `/root/autodl-tmp/RL-2048`
2. `bash scripts/setup_autodl.sh` 安装依赖（复用镜像自带 CUDA PyTorch）
3. tmux 内运行 `bash scripts/run_cloud_smoke.sh` 或 `bash scripts/run_cloud_experiments.sh`
4. 本地 `.\scripts\sync_results.ps1` 拉回 `results/`

云专用配置在 `configs/autodl/`（`output_dir` 指向数据盘）。

## 下一步（路线图 E2–E4）

- E2：vanilla vs Double DQN 配对比较
- E3：γ 扫描；E4：n-step


