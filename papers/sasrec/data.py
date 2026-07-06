from __future__ import annotations

import csv
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


@dataclass
class SequenceData:
    train_sequences: list[list[int]]
    valid_targets: list[int]
    test_targets: list[int]
    seen_items: list[set[int]]
    num_users: int
    num_items: int
    num_interactions: int


def load_sequence_data(path: Path, fast_dev_run: bool, fast_users: int) -> SequenceData:
    if not path.exists():
        raise FileNotFoundError(f"Missing 5-core file: {path}")

    user_rows: dict[str, list[tuple[int, str]]] = defaultdict(list)
    item_raw_ids: set[str] = set()
    num_interactions = 0

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in tqdm(reader, desc=f"load {path.name}", unit="row"):
            user_id = row["user_id"]
            item_id = row["item_id"]
            timestamp = int(float(row["timestamp"]))
            user_rows[user_id].append((timestamp, item_id))
            item_raw_ids.add(item_id)
            num_interactions += 1

    raw_users = sorted(user_rows)
    if fast_dev_run:
        raw_users = raw_users[:fast_users]
        item_raw_ids = {item_id for user in raw_users for _ts, item_id in user_rows[user]}
        num_interactions = sum(len(user_rows[user]) for user in raw_users)

    item_to_id = {raw_item: idx + 1 for idx, raw_item in enumerate(sorted(item_raw_ids))}

    train_sequences: list[list[int]] = []
    valid_targets: list[int] = []
    test_targets: list[int] = []
    seen_items: list[set[int]] = []

    for raw_user in tqdm(raw_users, desc="build user sequences", unit="user"):
        ordered = sorted(user_rows[raw_user], key=lambda pair: (pair[0], pair[1]))
        sequence = [item_to_id[item_id] for _ts, item_id in ordered]
        if len(sequence) < 3:
            continue
        train_sequences.append(sequence[:-2])
        valid_targets.append(sequence[-2])
        test_targets.append(sequence[-1])
        seen_items.append(set(sequence))

    return SequenceData(
        train_sequences=train_sequences,
        valid_targets=valid_targets,
        test_targets=test_targets,
        seen_items=seen_items,
        num_users=len(train_sequences),
        num_items=len(item_to_id),
        num_interactions=num_interactions,
    )


def sample_negative(num_items: int, forbidden: set[int], rng: random.Random) -> int:
    if len(forbidden) >= num_items:
        raise ValueError("Cannot sample a negative item because the user saw every item.")
    for _ in range(100):
        if (item := rng.randint(1, num_items)) not in forbidden:
            return item
    return rng.choice([item for item in range(1, num_items + 1) if item not in forbidden])


class SASRecTrainDataset(Dataset):
    def __init__(
        self,
        train_sequences: list[list[int]],
        seen_items: list[set[int]],
        num_items: int,
        max_seq_len: int,
        seed: int,
    ) -> None:
        self.train_sequences = train_sequences
        self.seen_items = seen_items
        self.num_items = num_items
        self.max_seq_len = max_seq_len
        self.seed = seed

    def __len__(self) -> int:
        return len(self.train_sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sequence = self.train_sequences[index]
        clipped = sequence[-(self.max_seq_len + 1) :]
        input_items = clipped[:-1]
        positive_items = clipped[1:]

        seq = np.zeros(self.max_seq_len, dtype=np.int64)
        pos = np.zeros(self.max_seq_len, dtype=np.int64)
        neg = np.zeros(self.max_seq_len, dtype=np.int64)

        start = self.max_seq_len - len(input_items)
        seq[start:] = input_items
        pos[start:] = positive_items

        rng = random.Random(self.seed + index)
        for offset, item in enumerate(positive_items, start=start):
            if item != 0:
                neg[offset] = sample_negative(self.num_items, self.seen_items[index], rng)

        return {
            "user_ids": torch.tensor(index, dtype=torch.long),
            "sequences": torch.from_numpy(seq),
            "positive_ids": torch.from_numpy(pos),
            "negative_ids": torch.from_numpy(neg),
        }


def make_history(sequence: list[int], max_seq_len: int) -> list[int]:
    clipped = sequence[-max_seq_len:]
    return [0] * (max_seq_len - len(clipped)) + clipped


def build_eval_examples(
    data: SequenceData,
    split: str,
    config,
) -> tuple[torch.Tensor, torch.Tensor]:
    if split not in {"valid", "test"}:
        raise ValueError(f"Unknown split: {split}")

    histories: list[list[int]] = []
    candidates: list[list[int]] = []
    rng = random.Random(config.seed + (17 if split == "valid" else 29))

    user_iter = enumerate(data.train_sequences)
    user_iter = tqdm(
        user_iter,
        total=len(data.train_sequences),
        desc=f"build {split} sampled negatives",
        unit="user",
    )

    for user_idx, train_sequence in user_iter:
        is_valid = split == "valid"
        history = train_sequence if is_valid else [*train_sequence, data.valid_targets[user_idx]]
        target = data.valid_targets[user_idx] if is_valid else data.test_targets[user_idx]
        eval_forbidden = set(data.seen_items[user_idx])
        negatives: list[int] = []
        for _ in range(config.eval_negatives):
            negative = sample_negative(data.num_items, eval_forbidden, rng)
            negatives.append(negative)
            eval_forbidden.add(negative)
        histories.append(make_history(history, config.max_seq_len))
        candidates.append([target, *negatives])

    return torch.tensor(histories, dtype=torch.long), torch.tensor(candidates, dtype=torch.long)
