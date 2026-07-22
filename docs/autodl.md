# AutoDL 云 GPU 训练指南

## 实例选型

- **镜像**：PyTorch 2.5.1 / Python 3.12 / Ubuntu 22.04 / CUDA 12.4
- **GPU**：RTX 3060 / 3080 / 4090 任选最便宜单卡即可
- **项目目录**：`/root/autodl-tmp/RL-2048`（数据盘，实例释放后通常保留）

## 1. 本地：Git 推送

```powershell
cd F:\RL-2048
git init
git add .
git commit -m "Initial commit: RL-2048 DQN baseline and eval pipeline"
git remote add origin https://github.com/<user>/RL-2048.git
git branch -M main
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

[`configs/autodl/`](../configs/autodl/) 中的 YAML 将 `output_dir` 指向数据盘绝对路径，避免 checkpoint 写入系统盘。

## 注意事项

| 风险 | 应对 |
|------|------|
| SSH 断开杀训练 | 使用 tmux |
| pip 覆盖 CUDA torch | `setup_autodl.sh` 使用 `--no-deps` |
| 系统盘清空 | 代码与 results 均放 `/root/autodl-tmp` |
| 费用 | 先跑 `e1_smoke.yaml` 估算耗时 |
