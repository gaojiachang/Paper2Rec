# Taobao DIN / SIM 实验

本目录在同一训练器中实现三种候选相关模型：

- `din`：只使用最近 20 条短期行为的 DIN Target Attention 基线。
- `sim`：只使用最多 500 条长期行为的 Hard Search + ESU 基线。
- `ours`：拼接 DIN 短期兴趣与 SIM 长期兴趣的融合模型。

这里的 `sim` 是**仅长期兴趣分支的消融基线**，不是完整的 SIM 最终预测结构；短长期融合版本是 `ours`。

## 数据与缓存

输入为 `data/processed/taobao-userbehavior/` 下的清洗 Parquet 及其 `dataset/` 目标/候选文件。首次运行会在 `dataset/cache/` 生成连续 ID、用户 offsets 和完整时间序列缓存；之后会校验来源文件与映射规则后复用。使用 `--rebuild-cache` 可强制重建。

Item 和 Category 映射遵循：`0=PAD`、`1=OOV`、训练商品池中的真实 ID 从 `2` 开始。训练外历史商品及其类别均映射为 OOV；Valid/Test 的正负候选必须来自训练商品池。

## 运行
使用 `nohup` 在后台运行时，先创建日志目录：
```bash
nohup python -u papers/sim_din/trainer.py \
  --model din \
  > papers/sim_din/logs/din.txt 2>&1 &
```

```bash
nohup python -u papers/sim_din/trainer.py \
  --model sim \
  > papers/sim_din/logs/sim.txt 2>&1 &
```

```bash
nohup python -u papers/sim_din/trainer.py \
  --model ours \
  > papers/sim_din/logs/ours.txt 2>&1 &
```

## 实验协议

- 短期历史是目标前最后 20 条行为；长期历史是目标前的 `[-520:-20]`，两者均左侧填充。
- Hard Search 对每个候选单独在 GPU 上执行，按候选类别保留长期历史中最后 50 条匹配行为；无匹配时长期兴趣为零向量。
- 训练组为 1 正 + 2 同类别优先负例 + 2 全局负例。负例按 `train_frequency^0.75` 加权、排除用户完整浏览史，并在同类别不足时用全局池补齐。
- Valid/Test 每组固定为 1 正 + 99 负。每轮评估固定的 50,000 个分层 Valid 组；子集 AUC 创新高时才进行全量 Valid 确认。最佳模型由全量 Valid AUC 决定，最终只运行一次 Test。
- 早停同时监控固定 Valid 子集的 AUC、HR@10、NDCG@10、MRR；任意一项创历史新高就重置计数，四项连续 50 个 epoch 都未创新高时停止训练。
- 指标为 AUC、HR@10、NDCG@10、MRR。训练指标按每组 1 正 + 4 负统计，Valid/Test 指标按每组 1 正 + 99 负统计，两者不可直接比较；训练 HR@10 因每组只有 5 个候选而恒为 1。并列分数采用平均排名，避免正样本在候选第 0 位带来的偏置。
