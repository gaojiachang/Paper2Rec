# Paper2Rec

Paper2Rec 是一个通过[经典+前沿]论文的代码复现，帮助推荐系统初学者快速理解推荐算法的开源仓库。

## 数据来源

原始数据文件不提交到 GitHub，需要手动下载到 `data/raw/`：

- MovieLens 1M: https://files.grouplens.org/datasets/movielens/ml-1m.zip
- Amazon Beauty ratings: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv
- Amazon Books ratings: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv
- taobao-userbehavior: https://tianchi.aliyun.com/dataset/649

## 环境配置

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 5-Core 预处理

```bash
python3 scripts/preprocess_5core.py --dataset all --min-core 5
pytho3 scripts/preprocess_taobao.py
```

## 常用命令

指定 GPU：

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5
```

查看显卡状态：

```bash
watch -n 1 nvidia-smi
```

查看 TensorBoard：

```bash
tensorboard --logdir outputs
```
