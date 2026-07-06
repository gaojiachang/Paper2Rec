# SASRec 训练命令

```bash
mkdir -p papers/sasrec/logs
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
```

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

```bash
tail -f papers/sasrec/logs/amazon-books.txt
```
