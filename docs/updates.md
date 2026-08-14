# 更新说明（冲过启发式 → P(1024) → P(2048)）

相对仓库最初的 Masked Double DQN 基线（指数展平 + MLP，E1 均分约 2578）。

## 算法栈

- 观测：one-hot `C×4×4`（`obs_encoding: onehot`）
- 网络：无池化 CNN / **Dueling CNN**
- 数据：D4 对称增强（向量化）
- 回报：`n_step=5`，`gamma=0.995`，训练奖励 `log1p`（评估仍用原始分数）
- 采集：`num_envs` 可走 `AsyncVectorEnv`（`SAME_STEP` + `final_obs`）
- 可选 PER：实现了，但 **与 D4 增强冲突，默认关闭**（见 [per_notes.md](per_notes.md)）
- 评测：除贪心 Q 外，支持 **1-ply expectimax**（对 spawn 求期望 V）

## 主结果（val 1000，`max_episode_steps=1200`）

| 设定 | 均分 | P(1024) | P(2048) |
|---|---:|---:|---:|
| 随机 | 1097 | 0% | 0% |
| 启发式 | ~3850 | 1.8% | 0% |
| E1 MLP Double DQN | 2578 | — | — |
| one-hot+CNN 2M | 5040 | 0.6% | 0% |
| **Phase A Dueling 5M×3（贪心）** | **6123** | **16.9%** | **0.27%** |
| 同上 + **1-ply expectimax** | **~15820** | **82.6%** | **26.4%** |

Phase A 配置：`configs/autodl/opt_tile1024.yaml`。发布权重：`checkpoints/phaseA_dueling_seed0.pt`。

Expectimax 按种子：seed0 **32.2%** / seed1 **31.6%** / seed2 **15.5%** 到达 2048。约 11–31% 的强局会撞上 1200 步截断，因此 P(2048) 仍是下限。

## 实验上走过的弯路

1. **PER + `num_envs=8` + 把 ε 衰减拉到 8M**（旧 Phase B）把模型训崩（均分 ~3859）。消融见下。
2. **再从零训 10M**（Phase B v2：关 PER、n=8、ε 仍 4M）终局弱于 Phase A 5M，4–6M 峰值也冲不过 P(2048)≥1%。
3. 真正缺口是 **1024 之后的残局**，不是步数预算。贪心 Q 是 0 步视野；1-ply 搜索把转化率从 ~1.6% 拉到足以过 1% 门槛。

## 消融：`num_envs` × PER（5M×1）

| | PER off | PER on |
|---|---:|---:|
| `num_envs=1` | **5689** / P1024 12.9% | 5154 / 9.8% |
| `num_envs=8` | **5662** / 11.3% | 4830 / 7.8% |

并行环境可留着提速；PER 先不要开（D4 后写 TD 优先级近似噪声）。

## 怎么跑

```bash
# Phase A 风格训练（数据盘输出）
bash scripts/run_tile_opt.sh phaseA

# 贪心 vs expectimax
python scripts/eval_expectimax.py --skip-smoke --force-full --seeds 0 \
  --max-steps 1200 --no-stop-on-2048
```

可视化：`RL2048_CHECKPOINT=checkpoints/phaseA_dueling_seed0.pt rl2048-play`（下拉中的 DQN 为贪心；expectimax 目前走评估脚本）。
