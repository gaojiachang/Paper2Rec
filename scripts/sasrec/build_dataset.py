#!/usr/bin/env python3
"""将 5-core TSV 离线转换为 SASRec 专用训练与评估样本。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from papers.sasrec.config import COMMON_DEFAULTS, DATASET_DEFAULTS  # noqa: E402


def sample_negative(num_items: int, forbidden: set[int], rng: random.Random) -> int:
    """均匀采样一个用户未交互商品。"""
    if len(forbidden) >= num_items:
        raise ValueError("用户已覆盖完整商品池，无法采样负样本。")
    for _ in range(100):
        item = rng.randint(1, num_items)
        if item not in forbidden:
            return item
    return rng.choice([item for item in range(1, num_items + 1) if item not in forbidden])


def make_history(sequence: list[int], max_seq_len: int) -> np.ndarray:
    """截取最近历史并在左侧补零。"""
    output = np.zeros(max_seq_len, dtype=np.int64)
    clipped = sequence[-max_seq_len:]
    if clipped:
        output[-len(clipped) :] = clipped
    return output


def load_sequences(path: Path) -> tuple[list[list[int]], int, int]:
    """读取 TSV、按时间排序并把 raw item ID 映射为从 1 开始的连续 ID。"""
    if not path.is_file():
        raise FileNotFoundError(f"缺少 5-core 文件：{path}")

    user_rows: dict[str, list[tuple[int, str]]] = defaultdict(list)
    item_raw_ids: set[str] = set()
    num_interactions = 0
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in tqdm(reader, desc=f"读取 {path.name}", unit="row"):
            user_id = row["user_id"]
            item_id = row["item_id"]
            timestamp = int(float(row["timestamp"]))
            user_rows[user_id].append((timestamp, item_id))
            item_raw_ids.add(item_id)
            num_interactions += 1

    item_to_id = {
        raw_item: index + 1 for index, raw_item in enumerate(sorted(item_raw_ids))
    }
    sequences: list[list[int]] = []
    for raw_user in tqdm(sorted(user_rows), desc="构造用户序列", unit="user"):
        ordered = sorted(user_rows[raw_user], key=lambda pair: (pair[0], pair[1]))
        sequence = [item_to_id[item_id] for _timestamp, item_id in ordered]
        if len(sequence) >= 3:
            sequences.append(sequence)
    return sequences, len(item_to_id), num_interactions


def build_arrays(
    sequences: list[list[int]],
    num_items: int,
    max_seq_len: int,
    eval_negatives: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """构造训练序列和固定的 Valid/Test sampled-ranking 候选。"""
    num_users = len(sequences)
    shape = (num_users, max_seq_len)
    arrays = {
        "train_sequences": np.zeros(shape, dtype=np.int64),
        "train_positives": np.zeros(shape, dtype=np.int64),
        "train_negatives": np.zeros(shape, dtype=np.int64),
        "valid_histories": np.zeros(shape, dtype=np.int64),
        "valid_candidates": np.zeros((num_users, eval_negatives + 1), dtype=np.int64),
        "test_histories": np.zeros(shape, dtype=np.int64),
        "test_candidates": np.zeros((num_users, eval_negatives + 1), dtype=np.int64),
    }
    valid_rng = random.Random(seed + 17)
    test_rng = random.Random(seed + 29)

    for user_index, full_sequence in enumerate(
        tqdm(sequences, desc="构造 SASRec 样本", unit="user")
    ):
        train_sequence = full_sequence[:-2]
        valid_target = full_sequence[-2]
        test_target = full_sequence[-1]
        seen_items = set(full_sequence)

        clipped = train_sequence[-(max_seq_len + 1) :]
        input_items = clipped[:-1]
        positive_items = clipped[1:]
        start = max_seq_len - len(input_items)
        arrays["train_sequences"][user_index, start:] = input_items
        arrays["train_positives"][user_index, start:] = positive_items

        train_rng = random.Random(seed + user_index)
        for offset, positive in enumerate(positive_items, start=start):
            if positive != 0:
                arrays["train_negatives"][user_index, offset] = sample_negative(
                    num_items, seen_items, train_rng
                )

        arrays["valid_histories"][user_index] = make_history(
            train_sequence, max_seq_len
        )
        arrays["test_histories"][user_index] = make_history(
            [*train_sequence, valid_target], max_seq_len
        )

        for split, target, rng in (
            ("valid", valid_target, valid_rng),
            ("test", test_target, test_rng),
        ):
            candidates = arrays[f"{split}_candidates"][user_index]
            candidates[0] = target
            forbidden = set(seen_items)
            for candidate_index in range(1, eval_negatives + 1):
                negative = sample_negative(num_items, forbidden, rng)
                candidates[candidate_index] = negative
                forbidden.add(negative)

    return arrays


def save_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def replace_directory(staging_dir: Path, output_dir: Path) -> None:
    """用完整暂存目录原子替换旧离线数据。"""
    backup_dir = output_dir.parent / f".{output_dir.name}.backup-{os.getpid()}"
    shutil.rmtree(backup_dir, ignore_errors=True)
    moved_existing = False
    try:
        if output_dir.exists():
            os.replace(output_dir, backup_dir)
            moved_existing = True
        os.replace(staging_dir, output_dir)
        shutil.rmtree(backup_dir, ignore_errors=True)
    except Exception:
        if moved_existing and not output_dir.exists() and backup_dir.exists():
            os.replace(backup_dir, output_dir)
        raise


def build_dataset(
    dataset: str,
    input_path: Path,
    output_dir: Path,
    max_seq_len: int,
    eval_negatives: int,
    seed: int,
) -> None:
    input_path = input_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    staging_dir = output_dir.parent / f".{output_dir.name}.tmp-{os.getpid()}"
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True)
    try:
        sequences, num_items, num_interactions = load_sequences(input_path)
        if not sequences:
            raise ValueError("没有可用于 SASRec 的用户序列。")
        arrays = build_arrays(
            sequences, num_items, max_seq_len, eval_negatives, seed
        )
        for name, array in arrays.items():
            np.save(staging_dir / f"{name}.npy", array, allow_pickle=False)
        stat = input_path.stat()
        save_json(
            staging_dir / "dataset_config.json",
            {
                "dataset": dataset,
                "seed": seed,
                "max_seq_len": max_seq_len,
                "eval_negatives": eval_negatives,
                "source": {
                    "path": str(input_path),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                },
            },
        )
        save_json(
            staging_dir / "dataset_stats.json",
            {
                "num_users": len(sequences),
                "num_items": num_items,
                "num_interactions": num_interactions,
            },
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        replace_directory(staging_dir, output_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    print(
        f"[done] dataset={dataset} users={len(sequences):,} items={num_items:,} "
        f"interactions={num_interactions:,} output_dir={output_dir}",
        flush=True,
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构造 SASRec 专用离线样本。")
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), required=True)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--eval-negatives", type=int, default=None)
    parser.add_argument("--seed", type=int, default=COMMON_DEFAULTS["seed"])
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    defaults = DATASET_DEFAULTS[args.dataset]
    max_seq_len = args.max_seq_len or defaults["max_seq_len"]
    eval_negatives = args.eval_negatives or COMMON_DEFAULTS["eval_negatives"]
    if max_seq_len < 1 or eval_negatives < 1:
        raise ValueError("max_seq_len 和 eval_negatives 必须为正整数。")
    build_dataset(
        dataset=args.dataset,
        input_path=args.input or defaults["source_path"],
        output_dir=args.output_dir or defaults["dataset_dir"],
        max_seq_len=max_seq_len,
        eval_negatives=eval_negatives,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
