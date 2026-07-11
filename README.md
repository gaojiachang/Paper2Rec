# Paper2Rec

Paper2Rec 通过复现经典与前沿推荐系统论文，帮助初学者理解数据处理、模型结构和实验协议。

## 数据来源

原始数据不提交到 GitHub，请手动下载到对应的 `data/raw/<dataset>/`：

- MovieLens 1M: https://files.grouplens.org/datasets/movielens/ml-1m.zip
- Amazon Beauty: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv
- Amazon Books: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv
- Taobao UserBehavior: https://tianchi.aliyun.com/dataset/649

## 环境配置

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 清洗公共数据

MovieLens 与 Amazon 数据统一执行迭代 5-core 过滤，输出 `interactions_5core.tsv`：

```bash
python -u scripts/preprocess_5core.py --dataset all --min-core 5
```

淘宝数据完成行为过滤、频次过滤、排序和 Parquet 转换：

```bash
python -u scripts/preprocess_taobao.py
```

## 常用命令

指定 GPU：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=3
```

查看显卡状态：

```bash
watch -n 1 nvidia-smi
```

查看 TensorBoard：

```bash
tensorboard --logdir outputs
```
