# PER 问题备忘（暂缓修复）

状态：**先关闭 PER 冲 2048**；优先级与 D4 解耦等问题保留，以后再改。

相关消融：`configs/autodl/abl_n{1,8}_p{0,1}.yaml`，汇总见  
`/root/autodl-tmp/RL-2048/results/experiments/ablation_nenv_per.txt`。

## 消融结果（5M × seed0，val 1000）

| cell | num_envs | PER | mean | P(1024) |
|---|---:|---|---:|---:|
| n1_p0 | 1 | off | **5689** | **12.9%** |
| n1_p1 | 1 | on | 5154 | 9.8% |
| n8_p0 | 8 | off | **5662** | **11.3%** |
| n8_p1 | 8 | on | 4830 | 7.8% |

结论：

- **`num_envs=8` 基本无辜**（关 PER 时与对照同级）→ 可继续用于提速。
- **PER 是主因**（单独约 −535 分 / −3pp P1024；叠并行更差）。

## 机制（已核实）

1. **D4 × PER 优先级噪声（实现缺陷）**  
   当前：`sample → 随机 D4 增强 → 算 TD → update_priorities`。  
   下次再采同一条时变换不同。实测恒等 vs 随机 D4 的 TD 相关仅 **≈0.14**，高 TD top-16 重叠 ≈61%。优先级近似噪声。

2. **后期贪心起飞失败**  
   PER 在 ~3.5M 前训练局分不差甚至更好；ε→0.01 后无 PER 局分猛跳（~2k→5k+），有 PER 明显偏弱。  
   伴随：IS 压低 loss，但 batch TD 很高；`mean_q` 系统性偏低；高优先级槽更偏终局/高奖励。

3. **与任务匹配度一般**  
   log1p + n-step 下奖励已较密，经典 PER「抓稀有大回报」红利小，却放大重尾 TD。

## 若以后重开 PER（未做）

- priority 用**未增强**样本的 TD（或对 8 个 D4 平均后再写）
- 降低 `alpha` / 改 rank-based；`max_priority` 衰减或 clip
- 更晚启用 PER（Q 稳定后）
- 先跑 `n1_p1_fixed` 5M 对照，确认能收回分差再上长训

## 代码锚点

- `src/rl2048/rl/buffer.py` — `PrioritizedReplayBuffer`
- `src/rl2048/rl/agent.py` — `_augment_batch` 在 `compute_loss` 内、写 priority 之前
- `src/rl2048/rl/trainer.py` — `update_priorities(metrics.indices, metrics.td_errors)`
