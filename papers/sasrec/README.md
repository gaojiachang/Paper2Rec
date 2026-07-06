# SASRec 训练命令

## 论文与代码说明

SASRec 来自 Kang 和 McAuley 的 *Self-Attentive Sequential Recommendation*。它把用户历史交互序列看成一个按时间排序的 item 序列，用 causal self-attention 建模“当前位置只能看见过去”的序列依赖，再用最后一个位置的隐状态预测下一个 item。

当前目录是一个最小可运行实现，不走完整工程抽象。核心代码分为：

- `train_eval.py`：命令行入口，解析数据集和超参数。
- `config.py`：三份数据集的默认路径和训练超参数。
- `data.py`：读取 5-core TSV，做 raw id 到连续 id 的映射，并构造 train/valid/test 序列。
- `model.py`：SASRec 模型，包括 item/position embedding、causal self-attention block、FFN 和候选 item 打分。
- `trainer.py`：训练循环、point-wise BCE loss、checkpoint、TensorBoard 和最终 test 评估。
- `evaluate.py`：sampled `HR@10`、`NDCG@10`、`AUC`。

实现约定：

- item id 从 `1` 开始，`0` 保留为 padding。
- 每个用户按时间排序后使用 leave-one-out：倒数第二个 item 做 valid，最后一个 item 做 test。
- 训练时每个有效位置采 1 个负样本，使用 point-wise BCE。
- 评估时每个用户固定采 100 个未交互负样本，target 放在候选列表第 0 位。
- 输出写入 `outputs/sasrec/{dataset}/{run_id}/`，包括 `best.pt`、`config.json`、`metrics.json` 和 `tensorboard/`。


```bash
python -u papers/sasrec/train_eval.py --dataset ml-1m --fast-dev-run
```

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
