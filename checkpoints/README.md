# 发布权重

| 文件 | 训练 | 说明 |
|---|---|---|
| `phaseA_dueling_seed0.pt` | Phase A，seed 0，5M env steps | 当前发布的推理权重（已去掉 optimizer） |

这是冲 2048 时推荐使用的 checkpoint。评测期 **2-ply expectimax + 角落破同分** 在 val 200 上 **P(2048)=93.5%**（同协议 1-ply 约 30%；贪心仅 0.3%）。

同栈另外两个训练种子未放进仓库（各约 43MB）。贪心均分最高的是 seed 2（约 6989），但 1-ply expectimax 的 P(2048) 低于 seed 0/1。

加载：

```bash
# 贪心 Q（量网络本身）
rl2048-eval --checkpoint checkpoints/phaseA_dueling_seed0.pt --episodes 200 --seed-set val --decode greedy --max-steps 1200

# 2-ply expectimax（默认推理）
python scripts/eval_expectimax.py --skip-smoke --force-full --seeds 0 \
  --max-steps 4000 --stop-on-2048 --expectimax-only
```

`load_checkpoint` 与训练时格式兼容；本文件不含 optimizer，不能直接接着训练。
