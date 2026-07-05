# Paper2Rec

Paper2Rec is a recommendation-model playground. The current data pipeline uses
MovieLens 1M and Amazon review rating files for SASRec-style sequential
recommendation experiments.

## Data Sources

Raw data files are not committed to this repository. Download them from:

- MovieLens 1M: https://files.grouplens.org/datasets/movielens/ml-1m.zip
- Amazon Beauty ratings: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Beauty.csv
- Amazon Books ratings: https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/ratings_Books.csv

Expected local layout:

```text
data/raw/ml-1m/
data/raw/amazon-beauty/
data/raw/amazon-books/
data/processed/ml-1m/
data/processed/amazon-beauty/
data/processed/amazon-books/
```

The six data folders are tracked with `.gitkeep`, but their downloaded and
processed contents are ignored.

## Environment

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## 5-Core Preprocessing

```bash
python3 scripts/preprocess_5core.py --dataset all --min-core 5
```
