# 发布权重

| 文件 | 训练 | 说明 |
|---|---|---|
| `phaseA_dueling_seed0.pt` | Phase A，seed 0，5M env steps | 当前发布的推理权重（已去掉 optimizer） |

这是冲 2048 时推荐使用的 checkpoint：1-ply expectimax 在 val 1000 上 **P(2048)=32.2%**（贪心仅 0.3%）。

同栈另外两个训练种子未放进仓库（各约 43MB）。贪心均分最高的是 seed 2（约 6989），但 expectimax 的 P(2048) 低于 seed 0/1。

加载：

```bash
# 贪心
rl2048-eval --checkpoint checkpoints/phaseA_dueling_seed0.pt --episodes 200 --seed-set val --max-steps 1200

# 1-ply expectimax（推荐）
python scripts/eval_expectimax.py --skip-smoke --force-full --seeds 0 \
  --max-steps 1200 --no-stop-on-2048 --expectimax-only
```

`load_checkpoint` 与训练时格式兼容；本文件不含 optimizer，不能直接接着训练。
