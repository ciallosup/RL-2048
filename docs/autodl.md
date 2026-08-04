# AutoDL 云 GPU 训练指南

## 实例选型

- **镜像**：PyTorch 2.5.1 / Python 3.12 / Ubuntu 22.04 / CUDA 12.4
- **GPU**：RTX 3060 / 3080 / 4090 任选最便宜单卡即可
- **项目目录**：`/root/autodl-tmp/RL-2048`（数据盘，实例释放后通常保留）

## 1. 本地：Git 推送

在 GitHub 创建私有仓库后运行：

```powershell
cd F:\RL-2048
.\scripts\github_bootstrap.ps1 -RemoteUrl https://github.com/<user>/RL-2048.git
```

或手动：

```powershell
git remote add origin https://github.com/<user>/RL-2048.git
git push -u origin main
```

## 2. SSH 连接

将 [`scripts/ssh_config.example`](../scripts/ssh_config.example) 中的内容复制到 `C:\Users\<you>\.ssh\config`，填入 AutoDL 控制台提供的 HostName 和 Port。

```bash
ssh autodl-rl2048
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Cursor Remote-SSH

1. `Ctrl+Shift+P` → `Remote-SSH: Connect to Host` → `autodl-rl2048`
2. 打开文件夹 `/root/autodl-tmp/RL-2048`
3. 训练任务仍应在 **tmux** 内运行

## 3. 云上：克隆与环境

```bash
cd /root/autodl-tmp
git clone https://github.com/<user>/RL-2048.git
cd RL-2048
bash scripts/setup_autodl.sh
```

或使用一键脚本（在任意目录下载后执行，或 clone 后）：

```bash
bash scripts/autodl_bootstrap.sh https://github.com/<user>/RL-2048.git
```

## 4. 训练工作流（tmux）

```bash
tmux new -s train
cd /root/autodl-tmp/RL-2048
source .venv/bin/activate

# 冒烟测试（100k steps + baseline eval）
bash scripts/run_cloud_smoke.sh

# 正式实验（baseline + E1/E2 各 5 seeds）
bash scripts/run_cloud_experiments.sh
```

Detach：`Ctrl+B`, `D`  
重连：`tmux attach -t train`

## 5. 结果回传本地

```powershell
.\scripts\sync_results.ps1 -Host autodl-rl2048
```

本地可视化：

```powershell
$env:RL2048_CHECKPOINT = "results/runs/e1/.../checkpoint_final.pt"
rl2048-play
```

## 云专用配置

[`configs/autodl/`](../configs/autodl/) 中的 YAML 与本地实验配置超参一致；`output_dir` 为相对路径（`results/...`）。请把仓库放在数据盘 `/root/autodl-tmp/RL-2048`，这样 checkpoint 不会写到系统盘。

## 注意事项

| 风险 | 应对 |
|------|------|
| SSH 断开杀训练 | 使用 tmux |
| pip 覆盖 CUDA torch | `setup_autodl.sh` 使用 `--no-deps`，且 venv 带 `--system-site-packages` 复用镜像 torch |
| 系统盘清空 | 代码与 results 均放 `/root/autodl-tmp` |
| 费用 | 先跑 `e1_smoke.yaml` 估算耗时 |

## 正式多 seed 实验（E1 / E2）

在 tmux 中：

```bash
cd /root/autodl-tmp/RL-2048
git pull   # 或 checkout 含修复的分支
bash scripts/setup_autodl.sh   # 仅首次 / 依赖变更时
source .venv/bin/activate

tmux new -s train
# E1 Double DQN + E2 Vanilla DQN，各 5 个 train seed；val 1000 局评估
bash scripts/run_cloud_experiments.sh
# 或只跑训练、跳过基线评估：
# SKIP_BASELINE=1 bash scripts/run_cloud_experiments.sh
# 改 seed 数：TRAIN_SEEDS=3 bash scripts/run_cloud_experiments.sh
```

产出：

| 文件 | 内容 |
|------|------|
| `results/experiments/e1_latest.json` | Double DQN 各 seed 评估 |
| `results/experiments/e2_latest.json` | Vanilla DQN 各 seed 评估 |
| `results/experiments/e1_e2_compare.json` | 均值±标准差汇总 |
| `results/runs/e1/`、`results/runs/e2_vanilla/` | 各 seed checkpoint |

单 seed 约 500k steps；RTX 4080 上冒烟 100k ≈ 3 分钟，正式 5×2 seeds 预计数小时量级，务必放在 tmux 内。
