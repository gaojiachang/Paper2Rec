"""Caching, dynamic training groups, and streaming fixed-candidate evaluation."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

try:  # Support both ``python trainer.py`` and package imports in tests.
    from .config import RunConfig
    from .utils import file_fingerprint, save_json, stable_seed
except ImportError:  # pragma: no cover - exercised by direct CLI invocation.
    from config import RunConfig
    from utils import file_fingerprint, save_json, stable_seed


CACHE_VERSION = 1
PAD_ID = 0
OOV_ID = 1
FIRST_REAL_ID = 2
NEGATIVE_COUNT = 4
CATEGORY_NEGATIVE_COUNT = 2
GLOBAL_NEGATIVE_COUNT = 2
EVAL_NEGATIVE_COUNT = 99
MIN_HISTORY_LENGTH = 50
HISTORY_BUCKETS = ((50, 100, "50-99"), (100, 200, "100-199"), (200, 500, "200-499"), (500, None, "500+"))


@dataclass
class CachedSequences:
    cache_dir: Path
    user_raw_ids: np.ndarray
    item_raw_ids: np.ndarray
    category_raw_ids: np.ndarray
    catalog_item_category_ids: np.ndarray
    sampling_weights: np.ndarray
    user_offsets: np.ndarray
    item_sequences: np.ndarray
    category_sequences: np.ndarray
    timestamp_sequences: np.ndarray

    def __post_init__(self) -> None:
        self._user_lookup = {int(raw_id): index for index, raw_id in enumerate(self.user_raw_ids)}
        self._category_item_ids: dict[int, np.ndarray] = {}
        for category_id in np.unique(self.catalog_item_category_ids):
            item_positions = np.flatnonzero(self.catalog_item_category_ids == category_id)
            self._category_item_ids[int(category_id)] = (item_positions + FIRST_REAL_ID).astype(np.int64)

        self.item_category_by_id = np.full(self.num_item_embeddings, OOV_ID, dtype=np.int64)
        self.item_category_by_id[PAD_ID] = PAD_ID
        self.item_category_by_id[FIRST_REAL_ID:] = self.catalog_item_category_ids
        self.item_weights_by_id = np.zeros(self.num_item_embeddings, dtype=np.float64)
        self.item_weights_by_id[FIRST_REAL_ID:] = self.sampling_weights
        self.global_pool = WeightedAliasPool(
            np.arange(FIRST_REAL_ID, self.num_item_embeddings, dtype=np.int64),
            self.sampling_weights,
        )
        self.category_pools = {
            category_id: WeightedAliasPool(item_ids, self.item_weights_by_id[item_ids])
            for category_id, item_ids in self._category_item_ids.items()
        }

    @property
    def num_users(self) -> int:
        return len(self.user_raw_ids)

    @property
    def num_item_embeddings(self) -> int:
        return len(self.item_raw_ids) + FIRST_REAL_ID

    @property
    def num_category_embeddings(self) -> int:
        return len(self.category_raw_ids) + FIRST_REAL_ID

    def map_users(self, raw_user_ids: np.ndarray) -> np.ndarray:
        output = np.empty(len(raw_user_ids), dtype=np.int64)
        for index, raw_user_id in enumerate(raw_user_ids):
            try:
                output[index] = self._user_lookup[int(raw_user_id)]
            except KeyError as error:
                raise ValueError(f"Target references unknown raw user ID {raw_user_id}.") from error
        return output

    def map_items(self, raw_item_ids: np.ndarray, *, strict: bool) -> np.ndarray:
        raw_item_ids = np.asarray(raw_item_ids, dtype=np.int64)
        positions = np.searchsorted(self.item_raw_ids, raw_item_ids)
        known = (positions < len(self.item_raw_ids)) & (self.item_raw_ids[np.minimum(positions, len(self.item_raw_ids) - 1)] == raw_item_ids)
        mapped = np.full(raw_item_ids.shape, OOV_ID, dtype=np.int64)
        mapped[known] = positions[known] + FIRST_REAL_ID
        if strict and not np.all(known):
            unknown = int(raw_item_ids[np.flatnonzero(~known)[0]])
            raise ValueError(f"Candidate/target item {unknown} is absent from the train item pool.")
        return mapped

    def map_categories_for_history(self, raw_category_ids: np.ndarray, known_items: np.ndarray) -> np.ndarray:
        """Map categories only when their item belongs to the train pool.

        This implements the locked rule that an out-of-pool historical item and
        its category are both OOV, even if that raw category occurs elsewhere
        in the training catalog.
        """
        raw_category_ids = np.asarray(raw_category_ids, dtype=np.int64)
        positions = np.searchsorted(self.category_raw_ids, raw_category_ids)
        known_categories = (positions < len(self.category_raw_ids)) & (
            self.category_raw_ids[np.minimum(positions, len(self.category_raw_ids) - 1)] == raw_category_ids
        )
        mapped = np.full(raw_category_ids.shape, OOV_ID, dtype=np.uint32)
        eligible = known_items & known_categories
        mapped[eligible] = positions[eligible] + FIRST_REAL_ID
        return mapped


@dataclass(frozen=True)
class TargetArrays:
    sample_ids: np.ndarray
    user_indices: np.ndarray
    item_ids: np.ndarray
    category_ids: np.ndarray
    target_timestamps: np.ndarray
    history_end_positions: np.ndarray

    def __len__(self) -> int:
        return len(self.sample_ids)


class WeightedAliasPool:
    """Alias-table weighted draws with caller-controlled rejection."""

    def __init__(self, item_ids: np.ndarray, weights: np.ndarray) -> None:
        self.item_ids = np.asarray(item_ids, dtype=np.int64)
        weights = np.asarray(weights, dtype=np.float64)
        if not len(self.item_ids):
            raise ValueError("Cannot build an empty negative-sampling pool.")
        if len(self.item_ids) != len(weights) or np.any(weights <= 0) or not np.all(np.isfinite(weights)):
            raise ValueError("Negative-sampling weights must be finite and positive.")
        self.item_set = frozenset(int(item_id) for item_id in self.item_ids)
        self.probability, self.alias = self._build_alias(weights)

    @staticmethod
    def _build_alias(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        count = len(weights)
        scaled = weights / weights.sum() * count
        probability = np.empty(count, dtype=np.float64)
        alias = np.empty(count, dtype=np.int64)
        small = np.flatnonzero(scaled < 1.0).tolist()
        large = np.flatnonzero(scaled >= 1.0).tolist()
        while small and large:
            lower = small.pop()
            higher = large.pop()
            probability[lower] = scaled[lower]
            alias[lower] = higher
            scaled[higher] = scaled[higher] + scaled[lower] - 1.0
            (small if scaled[higher] < 1.0 else large).append(higher)
        for index in [*small, *large]:
            probability[index] = 1.0
            alias[index] = index
        return probability, alias

    def draw_indices(self, rng: np.random.Generator, count: int) -> np.ndarray:
        columns = rng.integers(0, len(self.item_ids), size=count)
        return np.where(rng.random(count) < self.probability[columns], columns, self.alias[columns])


def _parquet_signature(path: Path) -> dict[str, Any]:
    parquet = pq.ParquetFile(path)
    return {
        **file_fingerprint(path),
        "rows": parquet.metadata.num_rows,
        "schema": str(parquet.schema_arrow),
    }


def _cache_manifest(config: RunConfig) -> dict[str, Any]:
    dataset_dir = Path(config.dataset_dir)
    sources = {
        "clean": Path(config.clean_path),
        "catalog": dataset_dir / "item_catalog.parquet",
        "train_targets": dataset_dir / "train_targets.parquet",
        "valid_candidates": dataset_dir / "valid_candidates.parquet",
        "test_candidates": dataset_dir / "test_candidates.parquet",
    }
    config_file = dataset_dir / "dataset_config.json"
    if config_file.exists():
        sources["dataset_config"] = config_file
    return {
        "cache_version": CACHE_VERSION,
        "sources": {name: _parquet_signature(path) if path.suffix == ".parquet" else file_fingerprint(path) for name, path in sources.items()},
        "mapping": {
            "pad_id": PAD_ID,
            "oov_id": OOV_ID,
            "first_real_id": FIRST_REAL_ID,
            "vocabulary": "train_item_pool_only",
            "out_of_pool_history_category": "oov",
        },
        "valid_subset": {
            "seed": config.seed,
            "size": config.valid_subset_size,
            "buckets": [list(bucket) for bucket in HISTORY_BUCKETS],
        },
    }


def _cache_files(cache_dir: Path) -> tuple[Path, ...]:
    return tuple(
        cache_dir / filename
        for filename in (
            "id_mappings.npz",
            "user_offsets.npy",
            "item_sequences.npy",
            "category_sequences.npy",
            "timestamp_sequences.npy",
            "valid_subset_ids.npy",
            "cache_manifest.json",
        )
    )


def _cache_matches(cache_dir: Path, expected: dict[str, Any]) -> bool:
    manifest_path = cache_dir / "cache_manifest.json"
    if not all(path.is_file() for path in _cache_files(cache_dir)):
        return False
    try:
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return actual == expected


def ensure_cache(config: RunConfig) -> CachedSequences:
    cache_dir = Path(config.cache_dir)
    expected = _cache_manifest(config)
    if config.rebuild_cache or not _cache_matches(cache_dir, expected):
        _build_cache(config, expected)
    return load_cache(cache_dir)


def _load_catalog(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    table = pq.read_table(dataset_dir / "item_catalog.parquet", columns=["item_id", "category_id", "sampling_weight"])
    items = np.asarray(table.column("item_id").to_numpy(zero_copy_only=False), dtype=np.int64)
    categories = np.asarray(table.column("category_id").to_numpy(zero_copy_only=False), dtype=np.int64)
    weights = np.asarray(table.column("sampling_weight").to_numpy(zero_copy_only=False), dtype=np.float64)
    order = np.argsort(items, kind="stable")
    items, categories, weights = items[order], categories[order], weights[order]
    if len(items) == 0 or np.any(items[1:] == items[:-1]):
        raise ValueError("item_catalog must contain unique, non-empty item IDs.")
    if np.any(weights <= 0) or not np.all(np.isfinite(weights)):
        raise ValueError("item_catalog sampling weights must be finite and positive.")
    return items, categories, weights


def _map_raw_items(raw_items: np.ndarray, catalog_items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positions = np.searchsorted(catalog_items, raw_items)
    known = (positions < len(catalog_items)) & (catalog_items[np.minimum(positions, len(catalog_items) - 1)] == raw_items)
    mapped = np.full(raw_items.shape, OOV_ID, dtype=np.uint32)
    mapped[known] = positions[known] + FIRST_REAL_ID
    return mapped, known


def _build_cache(config: RunConfig, manifest: dict[str, Any]) -> None:
    cache_dir = Path(config.cache_dir)
    dataset_dir = Path(config.dataset_dir)
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = cache_dir.parent / f".{cache_dir.name}.tmp-{os.getpid()}"
    shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True)

    try:
        catalog_items, catalog_raw_categories, sampling_weights = _load_catalog(dataset_dir)
        category_raw_ids = np.unique(catalog_raw_categories)
        category_positions = np.searchsorted(category_raw_ids, catalog_raw_categories)
        catalog_mapped_categories = (category_positions + FIRST_REAL_ID).astype(np.uint32)

        clean_path = Path(config.clean_path)
        parquet = pq.ParquetFile(clean_path)
        interaction_count = parquet.metadata.num_rows
        item_output = np.lib.format.open_memmap(
            staging_dir / "item_sequences.npy", mode="w+", dtype=np.uint32, shape=(interaction_count,)
        )
        category_output = np.lib.format.open_memmap(
            staging_dir / "category_sequences.npy", mode="w+", dtype=np.uint32, shape=(interaction_count,)
        )
        timestamp_output = np.lib.format.open_memmap(
            staging_dir / "timestamp_sequences.npy", mode="w+", dtype=np.int64, shape=(interaction_count,)
        )

        user_raw_ids: list[int] = []
        user_offsets = [0]
        current_user: int | None = None
        position = 0
        for batch in parquet.iter_batches(
            batch_size=1_000_000, columns=["user_id", "item_id", "category_id", "timestamp"]
        ):
            raw_users = np.asarray(batch.column("user_id").to_numpy(zero_copy_only=False), dtype=np.int64)
            raw_items = np.asarray(batch.column("item_id").to_numpy(zero_copy_only=False), dtype=np.int64)
            raw_categories = np.asarray(batch.column("category_id").to_numpy(zero_copy_only=False), dtype=np.int64)
            timestamps = np.asarray(batch.column("timestamp").to_numpy(zero_copy_only=False), dtype=np.int64)
            mapped_items, known_items = _map_raw_items(raw_items, catalog_items)
            category_positions = np.searchsorted(category_raw_ids, raw_categories)
            known_categories = (category_positions < len(category_raw_ids)) & (
                category_raw_ids[np.minimum(category_positions, len(category_raw_ids) - 1)] == raw_categories
            )
            mapped_categories = np.full(raw_categories.shape, OOV_ID, dtype=np.uint32)
            eligible_categories = known_items & known_categories
            mapped_categories[eligible_categories] = category_positions[eligible_categories] + FIRST_REAL_ID
            if np.any(known_items & ~known_categories):
                offending = int(raw_items[np.flatnonzero(known_items & ~known_categories)[0]])
                raise ValueError(f"Train-pool item {offending} has a category absent from item_catalog.")
            if np.any(eligible_categories):
                expected_categories = catalog_mapped_categories[np.searchsorted(catalog_items, raw_items[eligible_categories])]
                if np.any(mapped_categories[eligible_categories] != expected_categories):
                    offending = int(raw_items[np.flatnonzero(eligible_categories)[np.flatnonzero(mapped_categories[eligible_categories] != expected_categories)[0]]])
                    raise ValueError(f"Clean item {offending} has a category inconsistent with item_catalog.")

            item_output[position : position + len(batch)] = mapped_items
            category_output[position : position + len(batch)] = mapped_categories
            timestamp_output[position : position + len(batch)] = timestamps

            boundaries = np.append(np.flatnonzero(raw_users[1:] != raw_users[:-1]) + 1, len(raw_users))
            start = 0
            for end in boundaries:
                raw_user = int(raw_users[start])
                if current_user is None:
                    current_user = raw_user
                    user_raw_ids.append(raw_user)
                elif raw_user != current_user:
                    if raw_user <= current_user:
                        raise ValueError("Clean input must be sorted by strictly increasing user groups.")
                    user_offsets.append(position + start)
                    user_raw_ids.append(raw_user)
                    current_user = raw_user
                start = int(end)
            position += len(batch)

        if position != interaction_count:
            raise AssertionError(f"Cache wrote {position} interactions, expected {interaction_count}.")
        user_offsets.append(position)
        del item_output, category_output, timestamp_output

        np.save(staging_dir / "user_offsets.npy", np.asarray(user_offsets, dtype=np.int64))
        np.savez_compressed(
            staging_dir / "id_mappings.npz",
            user_raw_ids=np.asarray(user_raw_ids, dtype=np.int64),
            item_raw_ids=catalog_items,
            category_raw_ids=category_raw_ids,
            catalog_item_category_ids=catalog_mapped_categories,
            sampling_weights=sampling_weights,
        )
        _write_valid_subset(
            dataset_dir / "valid_candidates.parquet",
            staging_dir / "valid_subset_ids.npy",
            config.seed,
            config.valid_subset_size,
        )
        save_json(staging_dir / "cache_manifest.json", manifest)
        _replace_cache_directory(staging_dir, cache_dir)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise


def _replace_cache_directory(staging_dir: Path, cache_dir: Path) -> None:
    backup_dir = cache_dir.parent / f".{cache_dir.name}.backup-{os.getpid()}"
    shutil.rmtree(backup_dir, ignore_errors=True)
    moved_existing = False
    try:
        if cache_dir.exists():
            os.replace(cache_dir, backup_dir)
            moved_existing = True
        os.replace(staging_dir, cache_dir)
    except Exception:
        if moved_existing and not cache_dir.exists() and backup_dir.exists():
            os.replace(backup_dir, cache_dir)
        raise
    finally:
        shutil.rmtree(backup_dir, ignore_errors=True)


def _bucket_indices(history_lengths: np.ndarray) -> list[np.ndarray]:
    output = []
    for lower, upper, _name in HISTORY_BUCKETS:
        mask = history_lengths >= lower
        if upper is not None:
            mask &= history_lengths < upper
        output.append(np.flatnonzero(mask))
    return output


def _write_valid_subset(valid_path: Path, output_path: Path, seed: int, desired_size: int) -> None:
    table = pq.read_table(valid_path, columns=["sample_id", "history_end_position"])
    sample_ids = np.asarray(table.column("sample_id").to_numpy(zero_copy_only=False), dtype=np.int64)
    lengths = np.asarray(table.column("history_end_position").to_numpy(zero_copy_only=False), dtype=np.int64)
    if len(sample_ids) <= desired_size:
        np.save(output_path, np.sort(sample_ids))
        return

    buckets = _bucket_indices(lengths)
    proportions = np.asarray([len(bucket) / len(sample_ids) for bucket in buckets], dtype=np.float64)
    raw_allocation = proportions * desired_size
    allocations = np.floor(raw_allocation).astype(int)
    nonempty = np.asarray([len(bucket) > 0 for bucket in buckets])
    allocations[(allocations == 0) & nonempty] = 1
    allocations = np.minimum(allocations, np.asarray([len(bucket) for bucket in buckets]))
    remaining = desired_size - int(allocations.sum())
    fractional_order = np.argsort(-(raw_allocation - np.floor(raw_allocation)), kind="stable")
    while remaining > 0:
        changed = False
        for bucket_index in fractional_order:
            if allocations[bucket_index] < len(buckets[bucket_index]):
                allocations[bucket_index] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            break
    while remaining < 0:
        for bucket_index in reversed(fractional_order):
            if allocations[bucket_index] > 1:
                allocations[bucket_index] -= 1
                remaining += 1
                if remaining == 0:
                    break

    rng = np.random.default_rng(seed + 17)
    selected = [
        sample_ids[rng.choice(bucket, size=allocation, replace=False)]
        for bucket, allocation in zip(buckets, allocations)
        if allocation > 0
    ]
    result = np.sort(np.concatenate(selected))
    if len(result) != desired_size:
        raise AssertionError(f"Expected {desired_size} fixed validation IDs, got {len(result)}.")
    np.save(output_path, result)


def load_cache(cache_dir: Path) -> CachedSequences:
    with np.load(cache_dir / "id_mappings.npz", allow_pickle=False) as mappings:
        user_raw_ids = mappings["user_raw_ids"]
        item_raw_ids = mappings["item_raw_ids"]
        category_raw_ids = mappings["category_raw_ids"]
        catalog_item_category_ids = mappings["catalog_item_category_ids"]
        sampling_weights = mappings["sampling_weights"]
    offsets = np.load(cache_dir / "user_offsets.npy", mmap_mode="r")
    item_sequences = np.load(cache_dir / "item_sequences.npy", mmap_mode="r")
    category_sequences = np.load(cache_dir / "category_sequences.npy", mmap_mode="r")
    timestamp_sequences = np.load(cache_dir / "timestamp_sequences.npy", mmap_mode="r")
    if len(offsets) != len(user_raw_ids) + 1 or int(offsets[-1]) != len(item_sequences):
        raise ValueError("Sequence cache offsets are inconsistent with the stored arrays.")
    if not (len(item_sequences) == len(category_sequences) == len(timestamp_sequences)):
        raise ValueError("Sequence cache arrays have inconsistent lengths.")
    return CachedSequences(
        cache_dir=cache_dir,
        user_raw_ids=user_raw_ids,
        item_raw_ids=item_raw_ids,
        category_raw_ids=category_raw_ids,
        catalog_item_category_ids=catalog_item_category_ids,
        sampling_weights=sampling_weights,
        user_offsets=offsets,
        item_sequences=item_sequences,
        category_sequences=category_sequences,
        timestamp_sequences=timestamp_sequences,
    )


def load_valid_subset_ids(cache: CachedSequences) -> np.ndarray:
    return np.load(cache.cache_dir / "valid_subset_ids.npy", mmap_mode="r")


def _read_target_arrays(cache: CachedSequences, path: Path, limit: int | None = None) -> TargetArrays:
    table = pq.read_table(
        path,
        columns=[
            "sample_id",
            "user_id",
            "target_item_id",
            "target_category_id",
            "target_timestamp",
            "history_end_position",
        ],
    )
    sample_ids = np.asarray(table.column("sample_id").to_numpy(zero_copy_only=False), dtype=np.int64)
    raw_users = np.asarray(table.column("user_id").to_numpy(zero_copy_only=False), dtype=np.int64)
    raw_items = np.asarray(table.column("target_item_id").to_numpy(zero_copy_only=False), dtype=np.int64)
    raw_categories = np.asarray(table.column("target_category_id").to_numpy(zero_copy_only=False), dtype=np.int64)
    target_timestamps = np.asarray(table.column("target_timestamp").to_numpy(zero_copy_only=False), dtype=np.int64)
    positions = np.asarray(table.column("history_end_position").to_numpy(zero_copy_only=False), dtype=np.int64)
    if limit is not None:
        sample_ids = sample_ids[:limit]
        raw_users = raw_users[:limit]
        raw_items = raw_items[:limit]
        raw_categories = raw_categories[:limit]
        target_timestamps = target_timestamps[:limit]
        positions = positions[:limit]
    user_indices = cache.map_users(raw_users)
    item_ids = cache.map_items(raw_items, strict=True)
    categories = cache.item_category_by_id[item_ids]
    expected_raw_categories = cache.category_raw_ids[categories - FIRST_REAL_ID]
    if not np.array_equal(raw_categories, expected_raw_categories):
        raise ValueError(f"Target category mismatch in {path.name} against item_catalog.")
    _validate_history_positions(cache, user_indices, positions, target_timestamps, path.name)
    return TargetArrays(sample_ids, user_indices, item_ids, categories, target_timestamps, positions)


def load_train_targets(cache: CachedSequences, dataset_dir: Path, limit: int | None = None) -> TargetArrays:
    return _read_target_arrays(cache, dataset_dir / "train_targets.parquet", limit)


def _validate_history_positions(
    cache: CachedSequences,
    user_indices: np.ndarray,
    history_positions: np.ndarray,
    target_timestamps: np.ndarray,
    source_name: str,
) -> None:
    lengths = cache.user_offsets[user_indices + 1] - cache.user_offsets[user_indices]
    if np.any(history_positions < MIN_HISTORY_LENGTH) or np.any(history_positions > lengths):
        raise ValueError(
            f"{source_name} has a history position below {MIN_HISTORY_LENGTH} or outside user sequence boundaries."
        )
    nonempty = history_positions > 0
    indices = cache.user_offsets[user_indices[nonempty]] + history_positions[nonempty] - 1
    if np.any(cache.timestamp_sequences[indices] >= target_timestamps[nonempty]):
        raise ValueError(f"{source_name} history_end_position includes a target-second event.")


def _empty_history_batch(batch_size: int, short_len: int, long_len: int) -> dict[str, np.ndarray]:
    return {
        "short_items": np.zeros((batch_size, short_len), dtype=np.int64),
        "short_categories": np.zeros((batch_size, short_len), dtype=np.int64),
        "long_items": np.zeros((batch_size, long_len), dtype=np.int64),
        "long_categories": np.zeros((batch_size, long_len), dtype=np.int64),
        "long_timestamps": np.zeros((batch_size, long_len), dtype=np.int64),
    }


def build_history_batch(
    cache: CachedSequences,
    user_indices: np.ndarray,
    history_end_positions: np.ndarray,
    short_len: int,
    long_len: int,
) -> dict[str, np.ndarray]:
    output = _empty_history_batch(len(user_indices), short_len, long_len)
    for row, (user_index, history_end) in enumerate(zip(user_indices, history_end_positions)):
        start = int(cache.user_offsets[user_index])
        end = start + int(history_end)
        short_start = max(start, end - short_len)
        long_end = max(start, end - short_len)
        long_start = max(start, end - short_len - long_len)
        short_items = np.asarray(cache.item_sequences[short_start:end], dtype=np.int64)
        short_categories = np.asarray(cache.category_sequences[short_start:end], dtype=np.int64)
        long_items = np.asarray(cache.item_sequences[long_start:long_end], dtype=np.int64)
        long_categories = np.asarray(cache.category_sequences[long_start:long_end], dtype=np.int64)
        long_timestamps = np.asarray(cache.timestamp_sequences[long_start:long_end], dtype=np.int64)
        if len(short_items):
            output["short_items"][row, -len(short_items) :] = short_items
            output["short_categories"][row, -len(short_categories) :] = short_categories
        if len(long_items):
            output["long_items"][row, -len(long_items) :] = long_items
            output["long_categories"][row, -len(long_categories) :] = long_categories
            output["long_timestamps"][row, -len(long_timestamps) :] = long_timestamps
    return output


def _draw_unique_negatives(
    pool: WeightedAliasPool,
    count: int,
    seen_items: set[int],
    selected_items: set[int],
    rng: np.random.Generator,
) -> list[int]:
    selected: list[int] = []
    attempts = 0
    while len(selected) < count:
        remaining = count - len(selected)
        for index in pool.draw_indices(rng, max(32, remaining * 3)):
            item_id = int(pool.item_ids[index])
            attempts += 1
            if item_id in seen_items or item_id in selected_items:
                continue
            selected_items.add(item_id)
            selected.append(item_id)
            if len(selected) == count:
                break
        if attempts > max(5_000, count * 1_000):
            raise RuntimeError("Negative sampling retry limit exceeded; eligible candidate pool is too small.")
    return selected


def sample_training_negatives(
    cache: CachedSequences,
    target_item_id: int,
    target_category_id: int,
    seen_items: set[int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Return two category-priority and two global weighted negatives."""
    if target_item_id not in seen_items:
        raise ValueError("The training target must be present in the user's full sequence.")
    category_pool = cache.category_pools.get(int(target_category_id))
    if category_pool is None:
        raise ValueError(f"No train-pool category sampler for mapped category {target_category_id}.")

    category_available = len(category_pool.item_set) - len(category_pool.item_set.intersection(seen_items))
    category_count = min(CATEGORY_NEGATIVE_COUNT, category_available)
    selected_items: set[int] = set()
    category_negatives = _draw_unique_negatives(
        category_pool, category_count, seen_items, selected_items, rng
    )
    global_count = GLOBAL_NEGATIVE_COUNT + (CATEGORY_NEGATIVE_COUNT - category_count)
    # The conservative count avoids a costly global set intersection in the
    # normal Taobao case, while the exact check produces a useful error on a
    # tiny synthetic fixture.
    conservative_available = len(cache.global_pool.item_ids) - len(seen_items) - len(selected_items)
    if conservative_available < global_count:
        exact_available = len(cache.global_pool.item_set - seen_items) - len(selected_items)
        if exact_available < global_count:
            raise ValueError(
                f"Only {exact_available} global negatives are available; {global_count} are required."
            )
    global_negatives = _draw_unique_negatives(
        cache.global_pool, global_count, seen_items, selected_items, rng
    )
    negatives = np.asarray([*category_negatives, *global_negatives], dtype=np.int64)
    if len(negatives) != NEGATIVE_COUNT or len(set(negatives.tolist())) != NEGATIVE_COUNT:
        raise AssertionError("Training group did not contain exactly four unique negatives.")
    if target_item_id in negatives or any(int(item) in seen_items for item in negatives):
        raise AssertionError("Training negative violates target/seen-item exclusion.")
    return negatives


class TrainGroupDataset(Dataset[dict[str, torch.Tensor]]):
    """One date-split training target expanded to a five-candidate group."""

    def __init__(
        self,
        cache: CachedSequences,
        targets: TargetArrays,
        config: RunConfig,
    ) -> None:
        self.cache = cache
        self.targets = targets
        self.config = config
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.targets)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _seen_items(self, user_index: int) -> set[int]:
        start = int(self.cache.user_offsets[user_index])
        end = int(self.cache.user_offsets[user_index + 1])
        unique = np.unique(self.cache.item_sequences[start:end])
        return {int(item_id) for item_id in unique if int(item_id) >= FIRST_REAL_ID}

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        target_item_id = int(self.targets.item_ids[index])
        target_category_id = int(self.targets.category_ids[index])
        user_index = int(self.targets.user_indices[index])
        sample_id = int(self.targets.sample_ids[index])
        rng = np.random.default_rng(stable_seed(self.config.seed, self.epoch, sample_id))
        negatives = sample_training_negatives(
            self.cache,
            target_item_id,
            target_category_id,
            self._seen_items(user_index),
            rng,
        )
        candidate_items = np.concatenate((np.asarray([target_item_id], dtype=np.int64), negatives))
        candidate_categories = self.cache.item_category_by_id[candidate_items]
        history = build_history_batch(
            self.cache,
            self.targets.user_indices[index : index + 1],
            self.targets.history_end_positions[index : index + 1],
            self.config.short_len,
            self.config.long_len,
        )
        output = {name: torch.from_numpy(values[0]) for name, values in history.items()}
        output.update(
            {
                "candidate_items": torch.from_numpy(candidate_items),
                "candidate_categories": torch.from_numpy(candidate_categories.astype(np.int64, copy=False)),
                "target_timestamps": torch.tensor(self.targets.target_timestamps[index], dtype=torch.long),
                "labels": torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
                "history_lengths": torch.tensor(self.targets.history_end_positions[index], dtype=torch.long),
                "sample_ids": torch.tensor(sample_id, dtype=torch.long),
            }
        )
        return output


def _batch_to_tensors(
    histories: dict[str, np.ndarray],
    candidate_items: np.ndarray,
    candidate_categories: np.ndarray,
    target_timestamps: np.ndarray,
    history_lengths: np.ndarray,
    sample_ids: np.ndarray,
) -> dict[str, torch.Tensor]:
    output = {name: torch.from_numpy(values) for name, values in histories.items()}
    output.update(
        {
            "candidate_items": torch.from_numpy(candidate_items.astype(np.int64, copy=False)),
            "candidate_categories": torch.from_numpy(candidate_categories.astype(np.int64, copy=False)),
            "target_timestamps": torch.from_numpy(target_timestamps.astype(np.int64, copy=False)),
            "history_lengths": torch.from_numpy(history_lengths.astype(np.int64, copy=False)),
            "sample_ids": torch.from_numpy(sample_ids.astype(np.int64, copy=False)),
        }
    )
    return output


class EvaluationBatchIterator:
    """Read fixed 100-candidate groups from Parquet without full materialisation."""

    def __init__(
        self,
        cache: CachedSequences,
        candidate_path: Path,
        config: RunConfig,
        *,
        selected_sample_ids: Sequence[int] | None = None,
        limit: int | None = None,
    ) -> None:
        self.cache = cache
        self.candidate_path = candidate_path
        self.config = config
        self.selected_sample_ids = (
            frozenset(int(sample_id) for sample_id in selected_sample_ids)
            if selected_sample_ids is not None
            else None
        )
        self.limit = limit

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        parquet = pq.ParquetFile(self.candidate_path)
        emitted = 0
        columns = (
            "sample_id",
            "user_id",
            "target_item_id",
            "target_category_id",
            "target_timestamp",
            "history_end_position",
            "negative_item_ids",
        )
        for record_batch in parquet.iter_batches(batch_size=10_000, columns=columns):
            sample_ids = np.asarray(record_batch.column("sample_id").to_numpy(zero_copy_only=False), dtype=np.int64)
            if self.selected_sample_ids is not None:
                selected_mask = np.fromiter(
                    (int(sample_id) in self.selected_sample_ids for sample_id in sample_ids),
                    dtype=bool,
                    count=len(sample_ids),
                )
                if not np.any(selected_mask):
                    continue
            else:
                selected_mask = np.ones(len(sample_ids), dtype=bool)

            list_array = record_batch.column("negative_item_ids")
            offsets = np.asarray(list_array.offsets.to_numpy(zero_copy_only=False), dtype=np.int64)
            if not np.all(np.diff(offsets) == EVAL_NEGATIVE_COUNT):
                raise ValueError(f"{self.candidate_path.name} contains a candidate row without 99 negatives.")
            values_start, values_end = int(offsets[0]), int(offsets[-1])
            raw_negative_values = list_array.values.slice(values_start, values_end - values_start)
            negatives = np.asarray(raw_negative_values.to_numpy(zero_copy_only=False), dtype=np.int64).reshape(
                len(sample_ids), EVAL_NEGATIVE_COUNT
            )

            raw_users = np.asarray(record_batch.column("user_id").to_numpy(zero_copy_only=False), dtype=np.int64)[selected_mask]
            raw_targets = np.asarray(record_batch.column("target_item_id").to_numpy(zero_copy_only=False), dtype=np.int64)[selected_mask]
            raw_target_categories = np.asarray(
                record_batch.column("target_category_id").to_numpy(zero_copy_only=False), dtype=np.int64
            )[selected_mask]
            target_timestamps = np.asarray(record_batch.column("target_timestamp").to_numpy(zero_copy_only=False), dtype=np.int64)[selected_mask]
            history_lengths = np.asarray(record_batch.column("history_end_position").to_numpy(zero_copy_only=False), dtype=np.int64)[selected_mask]
            selected_ids = sample_ids[selected_mask]
            candidates_raw = np.concatenate((raw_targets[:, None], negatives[selected_mask]), axis=1)
            candidate_items = self.cache.map_items(candidates_raw, strict=True)
            candidate_categories = self.cache.item_category_by_id[candidate_items]
            expected_raw_categories = self.cache.category_raw_ids[
                candidate_categories[:, 0] - FIRST_REAL_ID
            ]
            if not np.array_equal(raw_target_categories, expected_raw_categories):
                raise ValueError(f"Target category mismatch in {self.candidate_path.name} against item_catalog.")
            user_indices = self.cache.map_users(raw_users)
            _validate_history_positions(
                self.cache, user_indices, history_lengths, target_timestamps, self.candidate_path.name
            )

            for start in range(0, len(selected_ids), self.config.eval_batch_size):
                if self.limit is not None and emitted >= self.limit:
                    return
                end = min(start + self.config.eval_batch_size, len(selected_ids))
                if self.limit is not None:
                    end = min(end, start + self.limit - emitted)
                histories = build_history_batch(
                    self.cache,
                    user_indices[start:end],
                    history_lengths[start:end],
                    self.config.short_len,
                    self.config.long_len,
                )
                yield _batch_to_tensors(
                    histories,
                    candidate_items[start:end],
                    candidate_categories[start:end],
                    target_timestamps[start:end],
                    history_lengths[start:end],
                    selected_ids[start:end],
                )
                emitted += end - start


def evaluation_batches(
    cache: CachedSequences,
    dataset_dir: Path,
    split: str,
    config: RunConfig,
    *,
    selected_sample_ids: Sequence[int] | None = None,
    limit: int | None = None,
) -> EvaluationBatchIterator:
    if split not in {"valid", "test"}:
        raise ValueError(f"Unknown evaluation split: {split}")
    return EvaluationBatchIterator(
        cache,
        dataset_dir / f"{split}_candidates.parquet",
        config,
        selected_sample_ids=selected_sample_ids,
        limit=limit,
    )
