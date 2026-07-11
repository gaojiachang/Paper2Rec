from __future__ import annotations

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from papers.sim_din.config import RunConfig
from papers.sim_din.data import (
    EvaluationBatchIterator,
    TrainGroupDataset,
    _validate_history_positions,
    ensure_cache,
    load_train_targets,
)
from papers.sim_din.evaluate import average_tie_rank_and_auc
from papers.sim_din.model import SimDinModel, hard_search_last_k
from papers.sim_din.trainer import run


def write_fixture(root: Path) -> tuple[Path, Path]:
    dataset_dir = root / "dataset"
    dataset_dir.mkdir(parents=True)
    item_ids = list(range(1, 111))
    categories = [10 if item_id <= 55 else 20 for item_id in item_ids]
    pq.write_table(
        pa.table(
            {
                "item_id": pa.array(item_ids, type=pa.int64()),
                "category_id": pa.array(categories, type=pa.int64()),
                "train_frequency": pa.array([1] * len(item_ids), type=pa.int64()),
                "sampling_weight": pa.array([1.0] * len(item_ids), type=pa.float64()),
            }
        ),
        dataset_dir / "item_catalog.parquet",
    )
    rows = [(1, 1, 10, "pv", timestamp) for timestamp in range(1, 61)]
    rows += [(1, 999, 10, "pv", 61), (1, 1, 10, "pv", 100), (1, 1, 10, "pv", 200)]
    rows += [(2, 2, 10, "pv", 1), (2, 2, 10, "pv", 2)]
    columns = list(zip(*rows))
    clean_path = root / "clean.parquet"
    pq.write_table(
        pa.table(
            {
                "user_id": pa.array(columns[0], type=pa.int64()),
                "item_id": pa.array(columns[1], type=pa.int64()),
                "category_id": pa.array(columns[2], type=pa.int64()),
                "behavior_type": pa.array(columns[3], type=pa.string()),
                "timestamp": pa.array(columns[4], type=pa.int64()),
            }
        ),
        clean_path,
    )
    base = {
        "sample_id": pa.array([0], type=pa.int64()),
        "user_id": pa.array([1], type=pa.int64()),
        "target_item_id": pa.array([1], type=pa.int64()),
        "target_category_id": pa.array([10], type=pa.int64()),
        "target_timestamp": pa.array([100], type=pa.int64()),
        "history_end_position": pa.array([61], type=pa.int64()),
    }
    pq.write_table(pa.table(base), dataset_dir / "train_targets.parquet")
    negatives = list(range(2, 101))
    for split, timestamp, position in (("valid", 100, 61), ("test", 200, 62)):
        columns = dict(base)
        columns["target_timestamp"] = pa.array([timestamp], type=pa.int64())
        columns["history_end_position"] = pa.array([position], type=pa.int64())
        columns["negative_item_ids"] = pa.array([negatives], type=pa.list_(pa.int64()))
        pq.write_table(pa.table(columns), dataset_dir / f"{split}_candidates.parquet")
    (dataset_dir / "dataset_config.json").write_text(json.dumps({"fixture": True}), encoding="utf-8")
    return dataset_dir, clean_path


def fixture_config(root: Path, model: str = "ours") -> RunConfig:
    dataset_dir, clean_path = write_fixture(root)
    return RunConfig(
        model=model,
        dataset_dir=str(dataset_dir),
        clean_path=str(clean_path),
        cache_dir=str(dataset_dir / "cache"),
        output_dir=str(root / "outputs" / model),
        seed=2026,
        item_embedding_dim=32,
        category_embedding_dim=16,
        time_embedding_dim=8,
        time_bucket_count=64,
        short_len=4,
        long_len=8,
        hard_search_k=4,
        attention_heads=4,
        dropout=0.0,
        batch_size=1,
        eval_batch_size=1,
        num_workers=0,
        learning_rate=1e-3,
        epochs=1,
        valid_subset_size=1,
        amp=False,
        device="cpu",
        rebuild_cache=False,
        fast_dev_run=True,
        fast_train_samples=1,
        fast_eval_groups=1,
    )


class SimDinTests(unittest.TestCase):
    def test_cache_oov_history_and_deterministic_training_group(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = fixture_config(Path(temporary))
            cache = ensure_cache(config)
            self.assertEqual(cache.num_users, 2)
            self.assertEqual(int(cache.item_sequences[60]), 1)
            self.assertEqual(int(cache.category_sequences[60]), 1)
            manifest_mtime = (Path(config.cache_dir) / "cache_manifest.json").stat().st_mtime_ns
            cache_reused = ensure_cache(config)
            self.assertEqual(manifest_mtime, (Path(config.cache_dir) / "cache_manifest.json").stat().st_mtime_ns)
            (Path(config.cache_dir) / "valid_subset_ids.npy").unlink()
            cache_reused = ensure_cache(config)
            self.assertTrue((Path(config.cache_dir) / "valid_subset_ids.npy").is_file())
            targets = load_train_targets(cache_reused, Path(config.dataset_dir))
            dataset = TrainGroupDataset(cache_reused, targets, config)
            dataset.set_epoch(3)
            first = dataset[0]
            second = dataset[0]
            self.assertTrue(torch.equal(first["candidate_items"], second["candidate_items"]))
            self.assertEqual(first["candidate_items"].shape[0], 5)
            self.assertEqual(len(set(first["candidate_items"].tolist()[1:])), 4)
            self.assertTrue(torch.all(first["short_items"][-4:] >= 0))
            with self.assertRaises(ValueError):
                _validate_history_positions(
                    cache_reused,
                    np.asarray([0], dtype=np.int64),
                    np.asarray([62], dtype=np.int64),
                    np.asarray([100], dtype=np.int64),
                    "same_second_target",
                )

    def test_hard_search_keeps_last_k_chronological_and_handles_empty(self) -> None:
        long_items = torch.tensor([[1, 2, 3, 4, 5, 0]], dtype=torch.long)
        long_categories = torch.tensor([[7, 8, 7, 7, 8, 0]], dtype=torch.long)
        long_times = torch.tensor([[1, 2, 3, 4, 5, 0]], dtype=torch.long)
        candidates = torch.tensor([[7, 9]], dtype=torch.long)
        items, categories, timestamps, mask = hard_search_last_k(
            long_items, long_categories, long_times, candidates, k=3
        )
        self.assertEqual(items[0, 0].tolist(), [1, 3, 4])
        self.assertEqual(categories[0, 0].tolist(), [7, 7, 7])
        self.assertTrue(mask[0, 0].all())
        self.assertEqual(items[0, 1].tolist(), [0, 0, 0])
        self.assertFalse(mask[0, 1].any())
        self.assertEqual(timestamps[0, 0].tolist(), [1, 3, 4])

    def test_models_evaluation_and_tie_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = fixture_config(Path(temporary), "ours")
            cache = ensure_cache(config)
            targets = load_train_targets(cache, Path(config.dataset_dir))
            dataset = TrainGroupDataset(cache, targets, config)
            sample = dataset[0]
            batch = {name: value.unsqueeze(0) for name, value in sample.items()}
            for model_name in ("din", "sim", "ours"):
                model = SimDinModel(replace(config, model=model_name), cache.num_item_embeddings, cache.num_category_embeddings)
                logits = model(batch)
                self.assertEqual(tuple(logits.shape), (1, 5))
                self.assertTrue(torch.isfinite(logits).all())
            sim_model = SimDinModel(replace(config, model="sim"), cache.num_item_embeddings, cache.num_category_embeddings)
            candidate_embedding = sim_model._behavior_embedding(
                torch.tensor([[2]]), torch.tensor([[2]])
            )
            no_match_interest = sim_model._long_interest(
                torch.tensor([[2, 2, 0, 0]]),
                torch.tensor([[2, 2, 0, 0]]),
                torch.tensor([[1, 2, 0, 0]]),
                torch.tensor([[3]]),
                candidate_embedding,
                torch.tensor([10]),
            )
            self.assertTrue(torch.equal(no_match_interest, torch.zeros_like(no_match_interest)))
            eval_batch = next(iter(EvaluationBatchIterator(cache, Path(config.dataset_dir) / "valid_candidates.parquet", config)))
            self.assertEqual(tuple(eval_batch["candidate_items"].shape), (1, 100))
            ranks, auc = average_tie_rank_and_auc(torch.tensor([[0.0, 0.0, -1.0]]))
            self.assertAlmostEqual(float(ranks[0]), 1.5)
            self.assertAlmostEqual(float(auc[0]), 0.75)

    def test_fast_training_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = fixture_config(Path(temporary), "din")
            summary = run(config)
            self.assertEqual(summary["best_epoch"], 1)
            self.assertNotIn("by_history_length", summary["test"])
            self.assertTrue((Path(config.output_dir) / "best.pt").is_file())
            self.assertTrue((Path(config.output_dir) / "metrics.json").is_file())


if __name__ == "__main__":
    unittest.main()
