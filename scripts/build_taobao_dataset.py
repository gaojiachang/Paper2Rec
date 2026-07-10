#!/usr/bin/env python3
"""Build the date-split Taobao data set used by the SIM reproduction.

This script intentionally keeps raw Taobao IDs and full histories.  It creates
time-based positive targets and fixed evaluation candidates; sequence clipping,
padding, ID mapping, hard search, and training-negative sampling belong to
later model stages.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data/processed/taobao-userbehavior/user_behavior_clean.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/taobao-userbehavior/dataset"

TIMEZONE = ZoneInfo("Asia/Shanghai")


def local_timestamp(year: int, month: int, day: int) -> int:
    return int(datetime(year, month, day, tzinfo=TIMEZONE).timestamp())


WARMUP_START = local_timestamp(2017, 11, 25)
TRAIN_START = local_timestamp(2017, 11, 26)
VALID_START = local_timestamp(2017, 12, 2)
TEST_START = local_timestamp(2017, 12, 3)
DATA_END = local_timestamp(2017, 12, 4)

DEFAULT_MIN_HISTORY = 50
DEFAULT_CATEGORY_NEGATIVES = 49
DEFAULT_GLOBAL_NEGATIVES = 50
DEFAULT_SEED = 2026

TARGET_COLUMNS = (
    "sample_id",
    "user_id",
    "target_item_id",
    "target_category_id",
    "target_timestamp",
    "history_end_position",
)

CANDIDATE_SCHEMA = pa.schema(
    [
        ("sample_id", pa.int64()),
        ("user_id", pa.int64()),
        ("target_item_id", pa.int64()),
        ("target_category_id", pa.int64()),
        ("target_timestamp", pa.int64()),
        ("history_end_position", pa.int64()),
        ("negative_item_ids", pa.list_(pa.int64())),
    ]
)


@dataclass(frozen=True)
class Target:
    sample_id: int
    user_id: int
    target_item_id: int
    target_category_id: int
    target_timestamp: int
    history_end_position: int


@dataclass
class SplitSamplingStats:
    samples: int = 0
    category_negatives: int = 0
    global_negatives: int = 0
    category_shortfall_backfilled: int = 0
    negative_samples_checked: int = 0


class WeightedAliasPool:
    """Weighted O(1) draws using the alias method.

    Rejection against a per-user forbidden set makes consecutive draws a
    weighted sample without replacement while avoiding repeated normalisation
    of a 500k-item probability vector for every evaluation sample.
    """

    def __init__(self, item_ids: np.ndarray, weights: np.ndarray) -> None:
        self.item_ids = np.asarray(item_ids, dtype=np.int64)
        weights = np.asarray(weights, dtype=np.float64)
        if len(self.item_ids) == 0:
            raise ValueError("Cannot build a sampling pool without items.")
        if len(self.item_ids) != len(weights) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("Sampling weights must be finite, positive, and aligned with item IDs.")

        self.item_set = frozenset(int(item_id) for item_id in self.item_ids)
        self.probability, self.alias = self._build_alias_table(weights)

    @staticmethod
    def _build_alias_table(weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
            if scaled[higher] < 1.0:
                small.append(higher)
            else:
                large.append(higher)

        for index in [*small, *large]:
            probability[index] = 1.0
            alias[index] = index
        return probability, alias

    def draw_indices(self, rng: np.random.Generator, size: int) -> np.ndarray:
        columns = rng.integers(0, len(self.item_ids), size=size)
        thresholds = rng.random(size)
        return np.where(
            thresholds < self.probability[columns], columns, self.alias[columns]
        )


class CandidateWriter:
    """Buffered writer for the two fixed-negative candidate files."""

    def __init__(self, path: Path, batch_size: int) -> None:
        self.path = path
        self.writer = pq.ParquetWriter(path, CANDIDATE_SCHEMA, compression="zstd")
        self.batch_size = batch_size
        self.buffer: dict[str, list[Any]] = {field.name: [] for field in CANDIDATE_SCHEMA}
        self.rows = 0

    def add(self, target: Target, negatives: list[int]) -> None:
        self.buffer["sample_id"].append(target.sample_id)
        self.buffer["user_id"].append(target.user_id)
        self.buffer["target_item_id"].append(target.target_item_id)
        self.buffer["target_category_id"].append(target.target_category_id)
        self.buffer["target_timestamp"].append(target.target_timestamp)
        self.buffer["history_end_position"].append(target.history_end_position)
        self.buffer["negative_item_ids"].append(negatives)
        self.rows += 1
        if len(self.buffer["sample_id"]) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer["sample_id"]:
            return
        self.writer.write_table(pa.Table.from_pydict(self.buffer, schema=CANDIDATE_SCHEMA))
        for values in self.buffer.values():
            values.clear()

    def close(self) -> None:
        self.flush()
        self.writer.close()


def scalar(connection: duckdb.DuckDBPyConnection, query: str, parameters: list[Any] | None = None) -> Any:
    return connection.execute(query, parameters or []).fetchone()[0]


def day_range(start: int, end: int) -> dict[str, Any]:
    return {
        "start": datetime.fromtimestamp(start, tz=TIMEZONE).isoformat(),
        "end_exclusive": datetime.fromtimestamp(end, tz=TIMEZONE).isoformat(),
        "start_timestamp": start,
        "end_timestamp_exclusive": end,
    }


def build_sql_tables(
    connection: duckdb.DuckDBPyConnection,
    input_path: Path,
    min_history_length: int,
) -> dict[str, int]:
    """Create item catalog, timestamp-aware histories, and positive targets."""
    print(f"[input] loading {input_path}", flush=True)
    connection.execute(
        """
        CREATE TABLE interactions AS
        SELECT user_id, item_id, category_id, timestamp
        FROM read_parquet(?)
        """,
        [str(input_path)],
    )
    source_rows = int(scalar(connection, "SELECT COUNT(*) FROM interactions"))

    # Aggregating per user-second before the cumulative sum is deliberate:
    # all events with timestamp == target_timestamp receive the same history
    # length, so no event in the target second leaks into its history.
    print("[history] computing strictly-earlier history positions", flush=True)
    connection.execute(
        """
        CREATE TABLE timestamp_history AS
        WITH events_per_second AS (
            SELECT user_id, timestamp, COUNT(*)::BIGINT AS events_at_timestamp
            FROM interactions
            GROUP BY user_id, timestamp
        )
        SELECT
            user_id,
            timestamp,
            CAST(
                SUM(events_at_timestamp) OVER (
                    PARTITION BY user_id
                    ORDER BY timestamp
                    ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
                ) - events_at_timestamp
                AS BIGINT
            ) AS history_end_position
        FROM events_per_second
        """
    )

    print("[catalog] deriving the train-time item pool", flush=True)
    connection.execute(
        f"""
        CREATE TABLE item_catalog AS
        SELECT
            item_id,
            MIN(category_id)::BIGINT AS category_id,
            COUNT(*)::BIGINT AS train_frequency,
            POWER(COUNT(*)::DOUBLE, 0.75) AS sampling_weight
        FROM interactions
        WHERE timestamp >= {TRAIN_START} AND timestamp < {VALID_START}
        GROUP BY item_id
        """
    )
    category_conflicts = int(
        scalar(
            connection,
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT item_id
                FROM interactions
                WHERE timestamp >= {TRAIN_START} AND timestamp < {VALID_START}
                GROUP BY item_id
                HAVING COUNT(DISTINCT category_id) > 1
            )
            """,
        )
    )
    if category_conflicts:
        raise ValueError(f"The cleaned input has {category_conflicts} train items with conflicting categories.")

    create_train_targets(connection, min_history_length)
    create_evaluation_targets(connection, "valid", VALID_START, TEST_START, min_history_length)
    create_evaluation_targets(connection, "test", TEST_START, DATA_END, min_history_length)

    stats: dict[str, int] = {
        "source_rows": source_rows,
        "timestamp_groups": int(scalar(connection, "SELECT COUNT(*) FROM timestamp_history")),
        "train_item_pool_count": int(scalar(connection, "SELECT COUNT(*) FROM item_catalog")),
        "train_category_count": int(scalar(connection, "SELECT COUNT(DISTINCT category_id) FROM item_catalog")),
    }
    for split, start, end in (
        ("train", TRAIN_START, VALID_START),
        ("valid", VALID_START, TEST_START),
        ("test", TEST_START, DATA_END),
    ):
        stats[f"{split}_eligible_events"] = int(
            scalar(
                connection,
                f"""
                SELECT COUNT(*)
                FROM interactions AS source
                INNER JOIN timestamp_history AS history USING (user_id, timestamp)
                WHERE source.timestamp >= {start}
                  AND source.timestamp < {end}
                  AND history.history_end_position >= {min_history_length}
                """,
            )
        )
    for split in ("train", "valid", "test"):
        stats[f"{split}_selected_targets"] = int(
            scalar(connection, f"SELECT COUNT(*) FROM {split}_selected")
        )
        stats[f"{split}_targets"] = int(
            scalar(connection, f"SELECT COUNT(*) FROM {split}_targets")
        )
    for split in ("valid", "test"):
        stats[f"{split}_cold_start_targets_removed"] = (
            stats[f"{split}_selected_targets"] - stats[f"{split}_targets"]
        )
        stats[f"{split}_cold_start_item_count"] = int(
            scalar(
                connection,
                f"""
                SELECT COUNT(DISTINCT selected.target_item_id)
                FROM {split}_selected AS selected
                LEFT JOIN item_catalog AS catalog
                    ON selected.target_item_id = catalog.item_id
                WHERE catalog.item_id IS NULL
                """,
            )
        )
    return stats


def create_train_targets(connection: duckdb.DuckDBPyConnection, min_history_length: int) -> None:
    """Keep each user's latest history-eligible target in each train day."""
    connection.execute(
        f"""
        CREATE TABLE train_selected AS
        WITH ranked AS (
            SELECT
                source.user_id,
                source.item_id AS target_item_id,
                source.category_id AS target_category_id,
                source.timestamp AS target_timestamp,
                history.history_end_position,
                ROW_NUMBER() OVER (
                    PARTITION BY source.user_id,
                        FLOOR((source.timestamp - {TRAIN_START}) / 86400)
                    ORDER BY source.timestamp DESC, source.item_id DESC
                ) AS target_rank
            FROM interactions AS source
            INNER JOIN timestamp_history AS history USING (user_id, timestamp)
            WHERE source.timestamp >= {TRAIN_START}
              AND source.timestamp < {VALID_START}
              AND history.history_end_position >= {min_history_length}
        )
        SELECT
            user_id,
            target_item_id,
            target_category_id,
            target_timestamp,
            history_end_position
        FROM ranked
        WHERE target_rank = 1
        """
    )
    connection.execute(
        """
        CREATE TABLE train_targets AS
        SELECT
            CAST(ROW_NUMBER() OVER (
                ORDER BY user_id ASC, target_timestamp ASC, target_item_id ASC
            ) - 1 AS BIGINT) AS sample_id,
            user_id,
            target_item_id,
            target_category_id,
            target_timestamp,
            history_end_position
        FROM train_selected
        """
    )


def create_evaluation_targets(
    connection: duckdb.DuckDBPyConnection,
    split: str,
    start: int,
    end: int,
    min_history_length: int,
) -> None:
    """Keep the latest target, then remove targets absent from the train pool."""
    connection.execute(
        f"""
        CREATE TABLE {split}_selected AS
        WITH ranked AS (
            SELECT
                source.user_id,
                source.item_id AS target_item_id,
                source.category_id AS target_category_id,
                source.timestamp AS target_timestamp,
                history.history_end_position,
                ROW_NUMBER() OVER (
                    PARTITION BY source.user_id
                    ORDER BY source.timestamp DESC, source.item_id DESC
                ) AS target_rank
            FROM interactions AS source
            INNER JOIN timestamp_history AS history USING (user_id, timestamp)
            WHERE source.timestamp >= {start}
              AND source.timestamp < {end}
              AND history.history_end_position >= {min_history_length}
        )
        SELECT
            user_id,
            target_item_id,
            target_category_id,
            target_timestamp,
            history_end_position
        FROM ranked
        WHERE target_rank = 1
        """
    )
    connection.execute(
        f"""
        CREATE TABLE {split}_targets AS
        SELECT
            CAST(ROW_NUMBER() OVER (
                ORDER BY selected.user_id ASC,
                         selected.target_timestamp ASC,
                         selected.target_item_id ASC
            ) - 1 AS BIGINT) AS sample_id,
            selected.user_id,
            selected.target_item_id,
            selected.target_category_id,
            selected.target_timestamp,
            selected.history_end_position
        FROM {split}_selected AS selected
        INNER JOIN item_catalog AS catalog
            ON selected.target_item_id = catalog.item_id
        """
    )


def copy_sql_outputs(
    connection: duckdb.DuckDBPyConnection,
    work_dir: Path,
) -> tuple[Path, Path]:
    train_path = work_dir / "train_targets.parquet"
    catalog_path = work_dir / "item_catalog.parquet"
    connection.execute(
        """
        COPY (
            SELECT sample_id, user_id, target_item_id, target_category_id,
                   target_timestamp, history_end_position
            FROM train_targets
            ORDER BY sample_id
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        [str(train_path)],
    )
    connection.execute(
        """
        COPY (
            SELECT item_id, category_id, train_frequency, sampling_weight
            FROM item_catalog
            ORDER BY item_id
        ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
        """,
        [str(catalog_path)],
    )
    return train_path, catalog_path


def load_target_map(connection: duckdb.DuckDBPyConnection, split: str) -> dict[int, Target]:
    table = connection.execute(
        f"""
        SELECT sample_id, user_id, target_item_id, target_category_id,
               target_timestamp, history_end_position
        FROM {split}_targets
        ORDER BY user_id, target_timestamp, target_item_id
        """
    ).to_arrow_table()
    columns = [table.column(name).to_pylist() for name in TARGET_COLUMNS]
    targets: dict[int, Target] = {}
    for values in zip(*columns):
        target = Target(*(int(value) for value in values))
        if target.user_id in targets:
            raise ValueError(f"{split} has more than one target for user {target.user_id}.")
        targets[target.user_id] = target
    return targets


def build_sampling_pools(catalog_path: Path) -> tuple[WeightedAliasPool, dict[int, WeightedAliasPool]]:
    catalog = pq.read_table(catalog_path, columns=["item_id", "category_id", "sampling_weight"])
    item_ids = catalog.column("item_id").to_numpy(zero_copy_only=False)
    category_ids = catalog.column("category_id").to_numpy(zero_copy_only=False)
    weights = catalog.column("sampling_weight").to_numpy(zero_copy_only=False)
    global_pool = WeightedAliasPool(item_ids, weights)

    category_positions: dict[int, list[int]] = {}
    for position, category_id in enumerate(category_ids):
        category_positions.setdefault(int(category_id), []).append(position)
    category_pools = {
        category_id: WeightedAliasPool(item_ids[positions], weights[positions])
        for category_id, positions in category_positions.items()
    }
    return global_pool, category_pools


def available_count(pool: WeightedAliasPool, seen_items: set[int]) -> int:
    # CPython's set intersection iterates the smaller set, which is the
    # per-user history in this data set (at most a few hundred items).
    return len(pool.item_ids) - len(pool.item_set.intersection(seen_items))


def draw_without_replacement(
    pool: WeightedAliasPool,
    count: int,
    seen_items: set[int],
    already_selected: set[int],
    rng: np.random.Generator,
) -> list[int]:
    """Draw weighted, unique items while rejecting seen and selected items."""
    selected: list[int] = []
    attempts = 0
    while len(selected) < count:
        remaining = count - len(selected)
        # A 2x oversample makes the common case one vectorised alias draw.
        for index in pool.draw_indices(rng, max(64, remaining * 2)):
            item_id = int(pool.item_ids[index])
            attempts += 1
            if item_id in seen_items or item_id in already_selected:
                continue
            already_selected.add(item_id)
            selected.append(item_id)
            if len(selected) == count:
                break
        if attempts > max(10_000, count * 1_000):
            raise RuntimeError(
                "Weighted negative sampling exceeded its retry limit; "
                "the requested pool may not contain enough eligible items."
            )
    return selected


def sample_negatives(
    target: Target,
    seen_items: set[int],
    global_pool: WeightedAliasPool,
    category_pools: dict[int, WeightedAliasPool],
    category_negatives: int,
    global_negatives: int,
    rng: np.random.Generator,
) -> tuple[list[int], int, int]:
    """Sample the 49 same-category and 50 global evaluation negatives."""
    if target.target_item_id not in seen_items:
        raise ValueError(
            f"Target item {target.target_item_id} is absent from user {target.user_id}'s full history."
        )
    category_pool = category_pools.get(target.target_category_id)
    if category_pool is None:
        raise ValueError(
            f"Target category {target.target_category_id} is absent from the train item pool."
        )

    same_category_count = min(category_negatives, available_count(category_pool, seen_items))
    selected: set[int] = set()
    category_items = draw_without_replacement(
        category_pool, same_category_count, seen_items, selected, rng
    )
    required_global = global_negatives + (category_negatives - same_category_count)

    # This fast conservative bound is sufficient for the Taobao train pool.
    # If a caller reduces the pool through custom arguments, use the exact
    # intersection before raising a precise error.
    conservative_available = len(global_pool.item_ids) - len(seen_items) - len(selected)
    if conservative_available < required_global:
        exact_available = available_count(global_pool, seen_items) - len(selected)
        if exact_available < required_global:
            raise ValueError(
                f"User {target.user_id} has only {exact_available} eligible global negatives; "
                f"{required_global} are required."
            )
    global_items = draw_without_replacement(
        global_pool, required_global, seen_items, selected, rng
    )
    negatives = [*category_items, *global_items]
    expected_count = category_negatives + global_negatives
    if (
        len(negatives) != expected_count
        or len(set(negatives)) != expected_count
        or target.target_item_id in negatives
        or not set(negatives).isdisjoint(seen_items)
        or any(item_id not in global_pool.item_set for item_id in negatives)
    ):
        raise AssertionError(f"Negative-sampling validation failed for user {target.user_id}.")
    return negatives, same_category_count, required_global


def iter_user_seen_items(input_path: Path, batch_size: int) -> Iterator[tuple[int, set[int]]]:
    """Stream all user-item sets from the input's promised user-time-item order."""
    parquet = pq.ParquetFile(input_path)
    current_user: int | None = None
    seen_items: set[int] = set()

    for batch in parquet.iter_batches(batch_size=batch_size, columns=["user_id", "item_id"]):
        user_ids = batch.column("user_id").to_numpy(zero_copy_only=False)
        item_ids = batch.column("item_id").to_numpy(zero_copy_only=False)
        boundaries = np.append(np.flatnonzero(user_ids[1:] != user_ids[:-1]) + 1, len(user_ids))
        start = 0
        for end in boundaries:
            user_id = int(user_ids[start])
            if current_user is None:
                current_user = user_id
            elif user_id != current_user:
                if user_id < current_user:
                    raise ValueError("Input is not sorted by ascending user_id as required.")
                yield current_user, seen_items
                current_user = user_id
                seen_items = set()
            seen_items.update(int(item_id) for item_id in item_ids[start:end])
            start = int(end)

    if current_user is not None:
        yield current_user, seen_items


def write_evaluation_candidates(
    input_path: Path,
    work_dir: Path,
    valid_targets: dict[int, Target],
    test_targets: dict[int, Target],
    global_pool: WeightedAliasPool,
    category_pools: dict[int, WeightedAliasPool],
    category_negatives: int,
    global_negatives: int,
    seed: int,
    batch_size: int,
) -> tuple[Path, Path, dict[str, SplitSamplingStats]]:
    """Create both candidate files in one full-data scan for complete seen sets."""
    valid_path = work_dir / "valid_candidates.parquet"
    test_path = work_dir / "test_candidates.parquet"
    writers = {
        "valid": CandidateWriter(valid_path, batch_size),
        "test": CandidateWriter(test_path, batch_size),
    }
    target_maps = {"valid": valid_targets, "test": test_targets}
    split_stats = {"valid": SplitSamplingStats(), "test": SplitSamplingStats()}
    rng = np.random.default_rng(seed)
    completed = 0

    try:
        for user_id, seen_items in iter_user_seen_items(input_path, batch_size):
            # The split order is fixed and recorded in dataset_config.json;
            # together with user order it makes a single seed reproducible.
            for split in ("valid", "test"):
                target = target_maps[split].pop(user_id, None)
                if target is None:
                    continue
                negatives, category_count, global_count = sample_negatives(
                    target,
                    seen_items,
                    global_pool,
                    category_pools,
                    category_negatives,
                    global_negatives,
                    rng,
                )
                writers[split].add(target, negatives)
                stats = split_stats[split]
                stats.samples += 1
                stats.category_negatives += category_count
                stats.global_negatives += global_count
                stats.category_shortfall_backfilled += category_negatives - category_count
                stats.negative_samples_checked += 1
                completed += 1
                if completed % 50_000 == 0:
                    print(f"[negatives] generated {completed} evaluation samples", flush=True)
    finally:
        for writer in writers.values():
            writer.close()

    missing_users = {
        split: len(targets) for split, targets in target_maps.items() if targets
    }
    if missing_users:
        raise ValueError(f"Targets reference users absent from the cleaned input: {missing_users}")
    if writers["valid"].rows != split_stats["valid"].samples or writers["test"].rows != split_stats["test"].samples:
        raise AssertionError("Candidate writer row counts do not match sampling counts.")
    return valid_path, test_path, split_stats


def validate_sql_targets(
    connection: duckdb.DuckDBPyConnection,
    min_history_length: int,
) -> dict[str, Any]:
    """Validate timestamp-safe history positions and evaluation target rules."""
    validation: dict[str, Any] = {}
    for split in ("train", "valid", "test"):
        violations = int(
            scalar(
                connection,
                f"""
                SELECT COUNT(*)
                FROM {split}_targets AS target
                LEFT JOIN timestamp_history AS history
                    ON target.user_id = history.user_id
                   AND target.target_timestamp = history.timestamp
                WHERE history.history_end_position IS NULL
                   OR target.history_end_position != history.history_end_position
                   OR target.history_end_position < {min_history_length}
                """,
            )
        )
        if violations:
            raise AssertionError(f"{split} has {violations} invalid history positions.")
        validation[f"{split}_history_position_violations"] = violations

    for split in ("valid", "test"):
        duplicate_users = int(
            scalar(
                connection,
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT user_id
                    FROM {split}_targets
                    GROUP BY user_id
                    HAVING COUNT(*) > 1
                )
                """,
            )
        )
        target_pool_violations = int(
            scalar(
                connection,
                f"""
                SELECT COUNT(*)
                FROM {split}_targets AS target
                LEFT JOIN item_catalog AS catalog
                    ON target.target_item_id = catalog.item_id
                WHERE catalog.item_id IS NULL
                """,
            )
        )
        if duplicate_users or target_pool_violations:
            raise AssertionError(
                f"{split} target validation failed: duplicate_users={duplicate_users}, "
                f"target_pool_violations={target_pool_violations}."
            )
        validation[f"{split}_duplicate_users"] = duplicate_users
        validation[f"{split}_target_pool_violations"] = target_pool_violations
    return validation


def validate_candidate_files(
    connection: duckdb.DuckDBPyConnection,
    candidate_paths: dict[str, Path],
    expected_counts: dict[str, int],
    negative_count: int,
) -> dict[str, Any]:
    """Check persisted candidate row counts, lengths, target pool, and schema."""
    validation: dict[str, Any] = {}
    for split, path in candidate_paths.items():
        schema_names = tuple(pq.ParquetFile(path).schema_arrow.names)
        if schema_names != (*TARGET_COLUMNS, "negative_item_ids"):
            raise AssertionError(f"Unexpected {split} candidate schema: {schema_names}")
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS row_count,
                COALESCE(MIN(ARRAY_LENGTH(negative_item_ids)), 0) AS min_negative_count,
                COALESCE(MAX(ARRAY_LENGTH(negative_item_ids)), 0) AS max_negative_count,
                COUNT(*) FILTER (
                    WHERE target_item_id NOT IN (SELECT item_id FROM item_catalog)
                ) AS target_pool_violations
            FROM read_parquet(?)
            """,
            [str(path)],
        ).fetchone()
        row_count, min_count, max_count, target_pool_violations = (int(value) for value in row)
        duplicate_users = int(
            scalar(
                connection,
                """
                SELECT COUNT(*)
                FROM (
                    SELECT user_id
                    FROM read_parquet(?)
                    GROUP BY user_id
                    HAVING COUNT(*) > 1
                )
                """,
                [str(path)],
            )
        )
        invalid_negative_lengths = row_count > 0 and (
            min_count != negative_count or max_count != negative_count
        )
        if (
            row_count != expected_counts[split]
            or invalid_negative_lengths
            or target_pool_violations
            or duplicate_users
        ):
            raise AssertionError(
                f"{split} candidate file validation failed: rows={row_count}, "
                f"negative_lengths=({min_count}, {max_count}), "
                f"target_pool_violations={target_pool_violations}, "
                f"duplicate_users={duplicate_users}."
            )
        validation[split] = {
            "rows": row_count,
            "negative_count_min": min_count,
            "negative_count_max": max_count,
            "duplicate_users": duplicate_users,
            "target_pool_violations": target_pool_violations,
        }
    return validation


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def build_config(
    min_history_length: int,
    category_negatives: int,
    global_negatives: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "source_file": "data/processed/taobao-userbehavior/user_behavior_clean.parquet",
        "timezone": "Asia/Shanghai",
        "time_ranges": {
            "warmup": {**day_range(WARMUP_START, TRAIN_START), "purpose": "history_only"},
            "train": day_range(TRAIN_START, VALID_START),
            "valid": day_range(VALID_START, TEST_START),
            "test": day_range(TEST_START, DATA_END),
        },
        "history": {
            "min_length": min_history_length,
            "strictly_before_target_timestamp": True,
            "same_timestamp_events_in_history": False,
            "history_end_position": {
                "definition": "Number of this user's cleaned interactions with timestamp strictly earlier than target_timestamp.",
                "indexing": "0-based exclusive slice end; reconstruct with user_events[:history_end_position].",
            },
        },
        "positive_selection": {
            "train": "Keep the last history-eligible event for every user and calendar day.",
            "valid_test": "Keep the last history-eligible event for every user and split day, then discard it if its item is absent from the train item pool; do not backfill an earlier event.",
            "tie_breaker": "timestamp descending, then item_id descending when selecting the last event.",
        },
        "negative_sampling": {
            "seed": seed,
            "candidate_iteration_order": "user_id ascending; valid then test for each user",
            "same_category_negatives": category_negatives,
            "global_negatives": global_negatives,
            "without_replacement": True,
            "exclude_user_full_history": True,
            "item_pool": "items observed in the train time range",
            "weight": "train_frequency ** 0.75; sampling_weight stores the unnormalised value",
            "category_shortfall": "Fill the missing same-category quota from the global weighted pool.",
        },
        "id_mapping": "not performed",
        "sequence_truncation": "not performed",
    }


def build_dataset_stats(
    sql_stats: dict[str, int],
    sampling_stats: dict[str, SplitSamplingStats],
    validation: dict[str, Any],
    seed: int,
    category_negatives: int,
    global_negatives: int,
) -> dict[str, Any]:
    splits: dict[str, Any] = {
        "train": {
            "eligible_target_events": sql_stats["train_eligible_events"],
            "selected_targets": sql_stats["train_selected_targets"],
            "samples": sql_stats["train_targets"],
        }
    }
    for split in ("valid", "test"):
        sample_stats = sampling_stats[split]
        splits[split] = {
            "eligible_target_events": sql_stats[f"{split}_eligible_events"],
            "selected_last_targets": sql_stats[f"{split}_selected_targets"],
            "cold_start_targets_removed": sql_stats[f"{split}_cold_start_targets_removed"],
            "cold_start_item_count": sql_stats[f"{split}_cold_start_item_count"],
            "samples": sql_stats[f"{split}_targets"],
            "negative_sampling": asdict(sample_stats),
        }
    return {
        "source_rows": sql_stats["source_rows"],
        "timestamp_groups": sql_stats["timestamp_groups"],
        "item_catalog": {
            "train_item_pool_count": sql_stats["train_item_pool_count"],
            "category_count": sql_stats["train_category_count"],
        },
        "splits": splits,
        "negative_sampling": {
            "seed": seed,
            "same_category_negatives_per_sample": category_negatives,
            "global_negatives_per_sample": global_negatives,
            "total_negatives_per_sample": category_negatives + global_negatives,
        },
        "validation": validation,
    }


def publish_outputs(work_paths: dict[str, Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    # The default work directory is a child of output_dir, making these atomic
    # renames.  For a custom directory on another filesystem, shutil.move
    # safely falls back to a copy followed by replacement.
    for filename, source in work_paths.items():
        destination = output_dir / filename
        if source.stat().st_dev == output_dir.stat().st_dev:
            os.replace(source, destination)
        else:
            temporary = output_dir / f".{filename}.{os.getpid()}.tmp"
            try:
                shutil.copy2(source, temporary)
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the date-split Taobao SIM target and candidate data set."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Temporary working directory (default: <output-dir>/.build_taobao_dataset_tmp).",
    )
    parser.add_argument("--min-history-length", type=int, default=DEFAULT_MIN_HISTORY)
    parser.add_argument("--category-negatives", type=int, default=DEFAULT_CATEGORY_NEGATIVES)
    parser.add_argument("--global-negatives", type=int, default=DEFAULT_GLOBAL_NEGATIVES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument("--keep-work-dir", action="store_true")
    args = parser.parse_args()
    args.input = args.input.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir is not None
        else args.output_dir / ".build_taobao_dataset_tmp"
    )
    return args


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise FileNotFoundError(f"Clean input does not exist: {args.input}")
    if args.min_history_length < 1:
        raise ValueError("--min-history-length must be positive")
    if args.category_negatives < 0 or args.global_negatives < 0:
        raise ValueError("Negative counts must be non-negative")
    if args.category_negatives + args.global_negatives < 1:
        raise ValueError("At least one evaluation negative is required")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")

    protected = {Path("/").resolve(), REPO_ROOT.resolve(), args.output_dir.resolve()}
    if args.work_dir.resolve() in protected:
        raise ValueError("--work-dir must be a dedicated directory, not the repository or output directory")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True)
    duckdb_temp_dir = args.work_dir / "duckdb_temp"
    duckdb_temp_dir.mkdir()

    connection: duckdb.DuckDBPyConnection | None = None
    completed = False
    try:
        connection = duckdb.connect(database=":memory:")
        connection.execute("SET temp_directory = ?", [str(duckdb_temp_dir)])
        sql_stats = build_sql_tables(connection, args.input, args.min_history_length)
        print(
            "[targets] "
            f"train={sql_stats['train_targets']} "
            f"valid={sql_stats['valid_targets']} "
            f"test={sql_stats['test_targets']}",
            flush=True,
        )

        train_path, catalog_path = copy_sql_outputs(connection, args.work_dir)
        global_pool, category_pools = build_sampling_pools(catalog_path)
        valid_targets = load_target_map(connection, "valid")
        test_targets = load_target_map(connection, "test")
        print(
            f"[negatives] sampling {len(valid_targets)} valid and {len(test_targets)} test candidates",
            flush=True,
        )
        valid_path, test_path, sampling_stats = write_evaluation_candidates(
            args.input,
            args.work_dir,
            valid_targets,
            test_targets,
            global_pool,
            category_pools,
            args.category_negatives,
            args.global_negatives,
            args.seed,
            args.batch_size,
        )

        validation = validate_sql_targets(connection, args.min_history_length)
        validation["candidate_files"] = validate_candidate_files(
            connection,
            {"valid": valid_path, "test": test_path},
            {"valid": sql_stats["valid_targets"], "test": sql_stats["test_targets"]},
            args.category_negatives + args.global_negatives,
        )
        validation["negative_sampling"] = {
            split: {
                "negative_samples_checked": stats.negative_samples_checked,
                "negative_item_membership_checked_against_full_history": True,
                "negative_uniqueness_checked": True,
                "positive_item_exclusion_checked": True,
                "train_item_pool_membership_checked": True,
            }
            for split, stats in sampling_stats.items()
        }

        config_path = args.work_dir / "dataset_config.json"
        stats_path = args.work_dir / "dataset_stats.json"
        write_json(
            config_path,
            build_config(
                args.min_history_length,
                args.category_negatives,
                args.global_negatives,
                args.seed,
            ),
        )
        write_json(
            stats_path,
            build_dataset_stats(
                sql_stats,
                sampling_stats,
                validation,
                args.seed,
                args.category_negatives,
                args.global_negatives,
            ),
        )

        publish_outputs(
            {
                "train_targets.parquet": train_path,
                "valid_candidates.parquet": valid_path,
                "test_candidates.parquet": test_path,
                "item_catalog.parquet": catalog_path,
                "dataset_config.json": config_path,
                "dataset_stats.json": stats_path,
            },
            args.output_dir,
        )
        completed = True
        print(f"[done] data set: {args.output_dir}", flush=True)
    finally:
        if connection is not None:
            connection.close()
        if completed and not args.keep_work_dir:
            shutil.rmtree(args.work_dir, ignore_errors=True)
        elif not completed:
            print(f"[error] intermediate files retained at: {args.work_dir}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
