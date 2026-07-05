# SASRec 训练命令

```bash
python papers/sasrec/train_eval.py --dataset ml-1m --fast-dev-run
```

```bash
nohup python papers/sasrec/train_eval.py \
  --dataset ml-1m \
  > papers/sasrec/logs/ml-1m.txt 2>&1 &
```

```bash
nohup python papers/sasrec/train_eval.py \
  --dataset amazon-beauty \
  > papers/sasrec/logs/amazon-beauty.txt 2>&1 &
```

```bash
nohup python papers/sasrec/train_eval.py \
  --dataset amazon-books \
  > papers/sasrec/logs/amazon-books.txt 2>&1 &
```
