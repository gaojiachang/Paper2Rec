# SASRec 实验

SASRec 来自 Kang 和 McAuley 的 *Self-Attentive Sequential Recommendation*。模型使用因果自注意力编码按时间排列的商品序列，并用最后一个有效位置的隐藏状态预测下一件商品。

## 代码结构

- `train_eval.py`：训练命令入口和参数解析。
- `config.py`：三个数据集的默认路径及超参数。
- `data.py`：校验并 mmap 读取 SASRec 专用离线数组。
- `model.py`：商品/位置 embedding、因果多头自注意力、FFN 和候选打分。
- `trainer.py`：BCE loss、优化器、四指标早停、checkpoint 和 TensorBoard。
- `evaluate.py`：计算 sampled AUC、HR@10、NDCG@10 和 MRR。
- `scripts/sasrec/build_dataset.py`：从公共 5-core TSV 构造模型专用样本。

## 构造模型样本

先在项目根目录完成公共 5-core 清洗，再分别执行：

```bash
python -u scripts/sasrec/build_dataset.py --dataset ml-1m
python -u scripts/sasrec/build_dataset.py --dataset amazon-beauty
python -u scripts/sasrec/build_dataset.py --dataset amazon-books
```

产物位于 `data/processed/<dataset>/sasrec/`，包括定长训练输入、逐位置正负样本、Valid/Test 历史、固定候选、配置和统计文件。训练只 mmap 读取这些数组，不再扫描 TSV 或在线构造样本。

离线构造与训练的 `seed`、`max_seq_len`、`eval_negatives` 必须一致；不一致时训练会直接报错，避免混用样本协议。

## 关键实现与参数

- `max_seq_len`：截取最近行为并在左侧补零；ML-1M 默认 200，Amazon 默认 50。
- `hidden_size=50`：商品 embedding、位置 embedding 和注意力隐藏维度。
- `num_blocks=2`：堆叠两个预归一化 SASRec Block。
- `num_heads`：ML-1M/Beauty 默认 1，Books 默认 2；隐藏维度必须能整除头数。
- `dropout`：ML-1M 默认 0.2，Amazon 默认 0.5。
- `batch_size=128`、`lr=1e-3`、`adam_beta2=0.98`。
- `eval_negatives=100`：每个 Valid/Test 正例固定搭配 100 个用户未交互负例。
- `patience=50`：AUC、HR@10、NDCG@10、MRR 任意一项创新高就重置；四项连续 50 个 epoch 均无新高时早停。

训练时每个有效位置只有 1 个正例和 1 个负例，因此 `train/hr@10` 恒为 1，训练指标不能与 1 正 + 100 负的 Valid/Test 指标直接比较。最佳 checkpoint 仍按 Valid NDCG@10 保存。

## 后台训练

```bash
nohup python -u papers/sasrec/train_eval.py \
  --dataset ml-1m \
  > papers/sasrec/logs/ml-1m.txt 2>&1 &
```

```bash
nohup python -u papers/sasrec/train_eval.py \
  --dataset amazon-beauty \
  > papers/sasrec/logs/amazon-beauty.txt 2>&1 &
```

```bash
nohup python -u papers/sasrec/train_eval.py \
  --dataset amazon-books \
  > papers/sasrec/logs/amazon-books.txt 2>&1 &
```

输出位于 `outputs/sasrec/<dataset>/<run_id>/`，包含 `config.json`、`best.pt`、`metrics.json` 和 `tensorboard/`。
