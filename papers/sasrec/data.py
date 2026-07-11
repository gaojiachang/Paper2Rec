"""读取由 ``scripts/sasrec/build_dataset.py`` 生成的 SASRec 离线样本。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


ARRAY_FILES = (
    "train_sequences.npy",
    "train_positives.npy",
    "train_negatives.npy",
    "valid_histories.npy",
    "valid_candidates.npy",
    "test_histories.npy",
    "test_candidates.npy",
)


@dataclass(frozen=True)
class PreparedSASRecData:
    train_sequences: np.ndarray
    train_positives: np.ndarray
    train_negatives: np.ndarray
    valid_histories: np.ndarray
    valid_candidates: np.ndarray
    test_histories: np.ndarray
    test_candidates: np.ndarray
    num_users: int
    num_items: int
    num_interactions: int


def load_prepared_data(config) -> PreparedSASRecData:
    """校验离线构造参数并以 mmap 方式加载训练和评估数组。"""
    dataset_dir = Path(config.dataset_dir)
    config_path = dataset_dir / "dataset_config.json"
    stats_path = dataset_dir / "dataset_stats.json"
    required = [config_path, stats_path, *(dataset_dir / name for name in ARRAY_FILES)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "SASRec 离线数据不完整，请先运行构建脚本。缺少：" + ", ".join(missing)
        )

    dataset_config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "dataset": config.dataset,
        "seed": config.seed,
        "max_seq_len": config.max_seq_len,
        "eval_negatives": config.eval_negatives,
    }
    mismatches = {
        name: (dataset_config.get(name), value)
        for name, value in expected.items()
        if dataset_config.get(name) != value
    }
    if mismatches:
        details = ", ".join(
            f"{name}: 离线值={actual!r}, 训练值={expected_value!r}"
            for name, (actual, expected_value) in mismatches.items()
        )
        raise ValueError(f"SASRec 离线数据配置与训练配置不一致：{details}")

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    arrays = {
        name.removesuffix(".npy"): np.load(dataset_dir / name, mmap_mode="r")
        for name in ARRAY_FILES
    }
    num_users = int(stats["num_users"])
    if any(len(array) != num_users for array in arrays.values()):
        raise ValueError("SASRec 离线数组的用户数量不一致。")

    if config.fast_dev_run:
        limit = min(config.fast_users, num_users)
        arrays = {name: array[:limit] for name, array in arrays.items()}
        num_users = limit

    return PreparedSASRecData(
        **arrays,
        num_users=num_users,
        num_items=int(stats["num_items"]),
        num_interactions=int(stats["num_interactions"]),
    )


class SASRecTrainDataset(Dataset[dict[str, torch.Tensor]]):
    """直接返回离线生成的定长训练序列及正负样本。"""

    def __init__(
        self,
        sequences: np.ndarray,
        positive_ids: np.ndarray,
        negative_ids: np.ndarray,
    ) -> None:
        self.sequences = sequences
        self.positive_ids = positive_ids
        self.negative_ids = negative_ids

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # torch.tensor 会复制只读 mmap 切片，避免 DataLoader 修改底层离线文件。
        return {
            "sequences": torch.tensor(self.sequences[index], dtype=torch.long),
            "positive_ids": torch.tensor(self.positive_ids[index], dtype=torch.long),
            "negative_ids": torch.tensor(self.negative_ids[index], dtype=torch.long),
        }
