# Taobao DIN / SIM 实验

同一训练器提供三种候选相关模型：

- `din`：最近 20 条短期行为上的 DIN Target Attention。
- `sim`：最多 500 条长期行为上的类别 Hard Search + 多头 ESU，是仅长期分支的消融模型。
- `ours`：拼接 DIN 短期兴趣和 SIM 长期兴趣的融合模型。

## 代码结构

- `config.py`：统一命令行参数和默认路径。
- `data.py`：缓存连续 ID、用户 offsets、历史序列，动态构造训练负例并流式读取评估候选。
- `model.py`：DIN 局部激活、类别 Hard Search、长期目标注意力和预测 MLP。
- `trainer.py`：训练、两级 Valid、四指标早停、checkpoint、Test 与 TensorBoard。
- `evaluate.py`：计算 AUC、HR@10、NDCG@10 和 MRR。
- `scripts/sim_din/build_taobao_dataset.py`：从清洗后的淘宝 Parquet 构造日期切分和固定评估候选。

## 构造模型样本

先按照项目根目录 README 清洗淘宝数据，再执行：

```bash
python -u scripts/sim_din/build_taobao_dataset.py
```

脚本将 Train、Valid、Test 按日期切分，生成正样本、商品池以及 Valid/Test 固定负例，输出到 `data/processed/taobao-userbehavior/sim_din/`。首次训练还会在其 `cache/` 下生成连续 ID 映射、用户 offsets 和完整序列缓存。

ID 约定为 `0=PAD`、`1=OOV`，训练商品池中的真实 ID 从 `2` 开始；Valid/Test 候选必须来自训练商品池。

## 关键实现与参数

- `short_len=20`：目标前最近 20 条行为，供 DIN 短期分支使用。
- `long_len=500`：目标前 `[-520:-20]` 的长期行为窗口。
- `hard_search_k=50`：按候选类别从长期窗口保留最近 50 条匹配行为。
- `batch_size=4096`、`eval_batch_size=2048`、`num_workers=8`。
- `learning_rate=1e-3`，CUDA 默认启用 bf16/fp16 AMP。
- `valid_subset_size=50000`：每轮固定评估分层 Valid 子集；子集 AUC 创新高时才执行完整 Valid。
- `patience=50`：固定 Valid 子集的四项指标任意一项创新高即重置，连续 50 轮均无新高时早停。

训练每组为 1 正 + 4 负，Valid/Test 每组为 1 正 + 99 负，两类指标不能直接比较；训练 HR@10 因候选仅有 5 个而恒为 1。最佳 checkpoint 由完整 Valid AUC 决定，最终只执行一次 Test。

## 后台训练

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

输出位于 `outputs/sim_din/<model>/<run_id>/`，包含 `config.json`、`best.pt`、`metrics.json` 和 `tensorboard/`。
