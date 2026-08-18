# 更新说明（冲过启发式 → P(1024) → P(2048)）

相对仓库最初的 Masked Double DQN 基线（指数展平 + MLP，E1 均分约 2578）。

## 算法栈

- 观测：one-hot `C×4×4`（`obs_encoding: onehot`）
- 网络：无池化 CNN / **Dueling CNN**
- 数据：D4 对称增强（向量化）
- 回报：`n_step=5`，`gamma=0.995`，训练奖励 `log1p`（评估仍用原始分数）
- 采集：`num_envs` 可走 `AsyncVectorEnv`（`SAME_STEP` + `final_obs`）
- 可选 PER：实现了，但 **与 D4 增强冲突，默认关闭**（见 [per_notes.md](per_notes.md)）
- 评测：除贪心 Q 外，支持 **1-ply / 2-ply expectimax**（对 spawn 求期望 V；2-ply 默认开角落破同分）

## 主结果

| 设定 | 均分 | P(1024) | P(2048) | 协议 |
|---|---:|---:|---:|---|
| 随机 | 1097 | 0% | 0% | val 1000 |
| 启发式 | ~3850 | 1.8% | 0% | val 1000 |
| E1 MLP Double DQN | 2578 | — | — | val 1000 |
| one-hot+CNN 2M | 5040 | 0.6% | 0% | val 1000 |
| **Phase A Dueling 5M×3（贪心）** | **6123** | **16.9%** | **0.27%** | val 1000，max_steps=1200 |
| 同上 + **1-ply expectimax** | **~15820** | **82.6%** | **26.4%** | val 1000，max_steps=1200，play-out |
| Phase A seed0 + 1-ply | 15658 | 83.8% | **30.0%** | val 前 80，max_steps=4000，stop@2048 |
| 同上 + **2-ply + 角落破同分** | 21098 | 100% | 96.2% | 同上 |
| **Phase A seed0 + 2-ply + 角落破同分** | **20754** | **99.5%** | **93.5%** | **val 200**，max_steps=4000，stop@2048 |

Phase A 配置：`configs/autodl/opt_tile1024.yaml`。发布权重：`checkpoints/phaseA_dueling_seed0.pt`。

1-ply 按种子（val 1000，max_steps=1200，play-out）：seed0 **32.2%** / seed1 **31.6%** / seed2 **15.5%**。约 11–31% 的强局会撞上 1200 步截断，因此该列的分数 / P(4096) 是下限；对 P(2048) 本身影响不大（多数截断局已经有 2048）。

2-ply 主数字来自 **val 200**（187/200，Wilson 95% CI：**89.2–96.2%**）。前 80 局对照里 2-ply 为 96.2%、1-ply 为 30.0%。均分是「第一次摸到 2048 或死亡」时的分数，不能直接和 play-out 的 15820 比。

## 2-ply 为什么涨这么多

贪心 Q 的动作分差只有约 2–4（绝对值 ~350），1-ply 几乎是在噪声里选。2-ply 多看一层「spawn 之后自己的应手」，能避开把 1024 残局走死的步。角落破同分：分差 ≤2 时优先把最大块留在角落——网络用了 D4 增强，本身不绑定某个角落，这一步补上「承诺」。

搜 2-ply 时用行 LUT 走子 + 批量合法掩码，seed0 大约 19 秒/局（RTX 4080）。

## 实验上走过的弯路

1. **PER + `num_envs=8` + 把 ε 衰减拉到 8M**（旧 Phase B）把模型训崩（均分 ~3859）。消融见下。
2. **再从零训 10M**（Phase B v2：关 PER、n=8、ε 仍 4M）终局弱于 Phase A 5M，4–6M 峰值也冲不过 P(2048)≥1%。
3. 真正缺口是 **1024 之后的残局**，不是步数预算。贪心 Q 是 0 步视野；1-ply 把 P(2048) 从 0.27% 拉到 ~30%；**2-ply + 角落破同分** 再把转化率拉到 val 200 上的 **93.5%**。

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
  --max-steps 4000 --stop-on-2048

# 复现 1-ply
python scripts/eval_expectimax.py --skip-smoke --force-full --seeds 0 \
  --max-steps 4000 --stop-on-2048 --depth 1 --no-corner-tiebreak
```

可视化：`rl2048-play`，加载 checkpoint 后默认 **2-ply**，可切换贪心 Q / 1-ply / 2-ply / 残局 3-ply。

## 冲 4096 与贪心微调（C1 / C1b）

Phase A 2-ply **打完局**（val 前 20，`max_steps=8000`，不停 2048）：均分 **50523**，P(2048)=100%，**P(4096)=55%**。残局 3-ply（2048 且空格≤4，或 1024 且空格≤3）把 P(4096) 拉到 **70%**，但约 8 min/局，默认推理仍用 2-ply。

想让**纯贪心 Q** 自己看懂 2048/4096，试过两次从 Phase A 微调，**都没有超过发布权重**：

| 设定 | 贪心均分 | 贪心 P(1024) | 2-ply 均分 | 2-ply P(4096) |
|---|---:|---:|---:|---:|
| Phase A | **5727** | **13.5%** | **50523** | **55%** |
| C1（在线 Q 当 2-ply 叶子） | 2542 | 0% | 18874 | 0% |
| C1b（冻结 Phase A 当老师 + BC） | 5406 | 6.5% | 50695 | 55% |

C1：`configs/autodl/c1_finetune_4096.yaml`。叶子一漂，老师自己就不会打 4096，replay 变成弱局。C1b：`configs/autodl/c1b_frozen_teacher.yaml`。采集稳住了（训练局 P(2048)≈81%、P(4096)≈17%），但学生贪心没变强，2-ply 与 Phase A 持平。发布权重仍用 `checkpoints/phaseA_dueling_seed0.pt`。

