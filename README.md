# Paper2Rec

Paper2Rec 是一个推荐系统论文复现实验仓库。当前主要实现和整理 SASRec 相关的数据处理、训练和评估代码。

## 数据来源

原始数据文件不提交到 GitHub，需要手动下载到 `data/raw/`：

- MovieLens 1M: https://files.grouplens.org/datasets/movielens/ml-1m.zip
- Amazon Beauty ratings: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv
- Amazon Books ratings: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv

## 环境配置

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 5-Core 预处理

```bash
python3 scripts/preprocess_5core.py --dataset all --min-core 5
```

## 常用命令

指定 GPU：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
```

查看显卡状态：

```bash
watch -n 1 nvidia-smi
```

查看 TensorBoard：

```bash
tensorboard --logdir outputs
```
