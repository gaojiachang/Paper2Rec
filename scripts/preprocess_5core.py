#!/usr/bin/env python3
import argparse
import csv
import os
import shutil
from collections import Counter
from pathlib import Path


DATASETS = {
    "ml-1m": {
        "raw": Path("data/raw/ml-1m/ratings.dat"),
        "out": Path("data/processed/ml-1m/interactions_5core.tsv"),
        "format": "ml-1m",
    },
    "amazon-beauty": {
        "raw": Path("data/raw/amazon-beauty/ratings_Beauty.csv"),
        "out": Path("data/processed/amazon-beauty/interactions_5core.tsv"),
        "format": "amazon",
    },
    "amazon-books": {
        "raw": Path("data/raw/amazon-books/ratings_Books.csv"),
        "out": Path("data/processed/amazon-books/interactions_5core.tsv"),
        "format": "amazon",
    },
}


def iter_raw(path, fmt):
    if fmt == "ml-1m":
        with path.open("r", encoding="latin-1", newline="") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                user_id, item_id, rating, timestamp = line.split("::")
                yield user_id, item_id, rating, timestamp
        return

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            user_id, item_id, rating, timestamp = row[:4]
            yield user_id, item_id, rating, timestamp


def write_normalized(raw_path, fmt, tmp_path):
    rows = 0
    users = set()
    items = set()
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        for user_id, item_id, rating, timestamp in iter_raw(raw_path, fmt):
            writer.writerow((user_id, item_id, rating, timestamp))
            rows += 1
            users.add(user_id)
            items.add(item_id)
    return rows, len(users), len(items)


def count_tsv(path):
    user_counts = Counter()
    item_counts = Counter()
    rows = 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for user_id, item_id, _rating, _timestamp in reader:
            user_counts[user_id] += 1
            item_counts[item_id] += 1
            rows += 1
    return rows, user_counts, item_counts


def filter_tsv(src, dst, user_counts, item_counts, min_core):
    rows = 0
    users = set()
    items = set()
    with src.open("r", encoding="utf-8", newline="") as f, dst.open(
        "w", encoding="utf-8", newline=""
    ) as out:
        reader = csv.reader(f, delimiter="\t")
        writer = csv.writer(out, delimiter="\t", lineterminator="\n")
        for user_id, item_id, rating, timestamp in reader:
            if user_counts[user_id] >= min_core and item_counts[item_id] >= min_core:
                writer.writerow((user_id, item_id, rating, timestamp))
                rows += 1
                users.add(user_id)
                items.add(item_id)
    return rows, len(users), len(items)


def add_header(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8", newline="") as f, dst.open(
        "w", encoding="utf-8", newline=""
    ) as out:
        out.write("user_id\titem_id\trating\ttimestamp\n")
        shutil.copyfileobj(f, out)


def run_dataset(name, min_core):
    spec = DATASETS[name]
    raw_path = spec["raw"]
    out_path = spec["out"]
    work_dir = out_path.parent / ".5core_tmp"
    current = work_dir / "normalized.tsv"
    next_path = work_dir / "filtered.tsv"

    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    print(f"[{name}] normalize: {raw_path}", flush=True)
    rows, users, items = write_normalized(raw_path, spec["format"], current)
    print(f"[{name}] raw rows={rows} users={users} items={items}", flush=True)

    iteration = 0
    while True:
        iteration += 1
        rows_before, user_counts, item_counts = count_tsv(current)
        kept_rows, kept_users, kept_items = filter_tsv(
            current, next_path, user_counts, item_counts, min_core
        )
        removed = rows_before - kept_rows
        print(
            f"[{name}] iter={iteration} rows={kept_rows} users={kept_users} "
            f"items={kept_items} removed={removed}",
            flush=True,
        )
        if removed == 0:
            break
        os.replace(next_path, current)

    print(f"[{name}] write: {out_path}", flush=True)
    add_header(current, out_path)
    shutil.rmtree(work_dir)
    return out_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run iterative user/item k-core filtering.")
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS.keys(), "all"],
        default="all",
        help="Dataset to process.",
    )
    parser.add_argument("--min-core", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        out_path = run_dataset(name, args.min_core)
        print(f"[{name}] done: {out_path}", flush=True)


if __name__ == "__main__":
    main()
