#!/usr/bin/env python3
"""Clean the Taobao UserBehavior data for the SIM reproduction.

The raw file has about one hundred million rows, so validation is performed in
a streaming pass.  The remaining relational operations use DuckDB and spill to
the selected work directory instead of requiring the complete data set in RAM.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = REPO_ROOT / "data/raw/taobao-userbehavior/UserBehavior.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data/processed/taobao-userbehavior"

# The published Taobao range is expressed in China Standard Time.  The raw
# timestamps are Unix timestamps, hence these bounds include 2017-11-25
# 00:00:00+08:00 and exclude 2017-12-04 00:00:00+08:00.
DATA_TIMEZONE = ZoneInfo("Asia/Shanghai")
START_TIMESTAMP = int(datetime(2017, 11, 25, tzinfo=DATA_TIMEZONE).timestamp())
END_TIMESTAMP = int(datetime(2017, 12, 4, tzinfo=DATA_TIMEZONE).timestamp())

STAGING_SCHEMA = pa.schema(
    [
        ("user_id", pa.int64()),
        ("item_id", pa.int64()),
        ("category_id", pa.int64()),
        ("behavior_type", pa.string()),
        ("timestamp", pa.int64()),
    ]
)


@dataclass
class InputStats:
    raw_rows: int = 0
    pv_rows: int = 0
    invalid_rows_removed: int = 0
    out_of_range_rows_removed: int = 0


def positive_int(value: str) -> int | None:
    """Return a positive base-10 integer, or ``None`` for invalid input."""
    if not value or not value.isdecimal():
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def flush_batch(writer: pq.ParquetWriter, batch: dict[str, list[Any]]) -> None:
    if not batch["user_id"]:
        return
    writer.write_table(pa.Table.from_pydict(batch, schema=STAGING_SCHEMA))
    for values in batch.values():
        values.clear()


def stream_validate_raw(
    input_path: Path, staging_path: Path, batch_size: int
) -> InputStats:
    """Validate raw records and write valid, in-range PV records to Parquet."""
    stats = InputStats()
    batch: dict[str, list[Any]] = {field.name: [] for field in STAGING_SCHEMA}

    with input_path.open("r", encoding="utf-8", newline="") as source, pq.ParquetWriter(
        staging_path, STAGING_SCHEMA, compression="zstd", write_statistics=True
    ) as writer:
        reader = csv.reader(source)
        for row in reader:
            stats.raw_rows += 1
            if len(row) != 5 or any(not value.strip() for value in row):
                stats.invalid_rows_removed += 1
                continue

            user_id = positive_int(row[0])
            item_id = positive_int(row[1])
            category_id = positive_int(row[2])
            timestamp = positive_int(row[4])
            if None in (user_id, item_id, category_id, timestamp):
                stats.invalid_rows_removed += 1
                continue

            if row[3] != "pv":
                continue

            stats.pv_rows += 1
            if not START_TIMESTAMP <= timestamp < END_TIMESTAMP:
                stats.out_of_range_rows_removed += 1
                continue

            batch["user_id"].append(user_id)
            batch["item_id"].append(item_id)
            batch["category_id"].append(category_id)
            batch["behavior_type"].append("pv")
            batch["timestamp"].append(timestamp)
            if len(batch["user_id"]) >= batch_size:
                flush_batch(writer, batch)

        flush_batch(writer, batch)

    return stats


def scalar(connection: duckdb.DuckDBPyConnection, query: str, parameters: Iterable[Any] = ()) -> Any:
    return connection.execute(query, parameters).fetchone()[0]


def create_resolved_table(connection: duckdb.DuckDBPyConnection, staging_path: Path) -> tuple[int, int]:
    """Deduplicate rows and map every item to its deterministic category."""
    connection.execute(
        """
        CREATE TABLE deduplicated AS
        SELECT DISTINCT user_id, item_id, category_id, behavior_type, timestamp
        FROM read_parquet(?)
        """,
        [str(staging_path)],
    )
    connection.execute(
        """
        CREATE TABLE category_counts AS
        SELECT item_id, category_id, COUNT(*) AS interaction_count
        FROM deduplicated
        GROUP BY item_id, category_id
        """
    )
    category_conflict_items = scalar(
        connection,
        """
        SELECT COUNT(*)
        FROM (
            SELECT item_id
            FROM category_counts
            GROUP BY item_id
            HAVING COUNT(*) > 1
        )
        """,
    )
    connection.execute(
        """
        CREATE TABLE canonical_categories AS
        SELECT item_id, category_id
        FROM (
            SELECT
                item_id,
                category_id,
                ROW_NUMBER() OVER (
                    PARTITION BY item_id
                    ORDER BY interaction_count DESC, category_id ASC
                ) AS category_rank
            FROM category_counts
        )
        WHERE category_rank = 1
        """
    )
    connection.execute(
        """
        CREATE TABLE resolved AS
        SELECT
            source.user_id,
            source.item_id,
            canonical.category_id,
            source.behavior_type,
            source.timestamp
        FROM deduplicated AS source
        INNER JOIN canonical_categories AS canonical USING (item_id)
        """
    )

    deduplicated_rows = scalar(connection, "SELECT COUNT(*) FROM deduplicated")
    return int(deduplicated_rows), int(category_conflict_items)


def iterative_frequency_filter(
    connection: duckdb.DuckDBPyConnection,
    min_user_interactions: int,
    min_item_interactions: int,
) -> tuple[str, int, int]:
    """Apply the specified user-then-item filters until the data is stable."""
    current_table = "resolved"
    next_table = "core_a"
    iterations = 0
    total_removed = 0

    while True:
        iterations += 1
        rows_before = int(scalar(connection, f"SELECT COUNT(*) FROM {current_table}"))
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE users_to_keep AS
            SELECT user_id
            FROM {current_table}
            GROUP BY user_id
            HAVING COUNT(*) >= ?
            """,
            [min_user_interactions],
        )
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE after_user_filter AS
            SELECT current.*
            FROM {current_table} AS current
            INNER JOIN users_to_keep USING (user_id)
            """
        )
        connection.execute(
            """
            CREATE OR REPLACE TABLE items_to_keep AS
            SELECT item_id
            FROM after_user_filter
            GROUP BY item_id
            HAVING COUNT(*) >= ?
            """,
            [min_item_interactions],
        )
        connection.execute(
            f"""
            CREATE OR REPLACE TABLE {next_table} AS
            SELECT current.*
            FROM after_user_filter AS current
            INNER JOIN items_to_keep USING (item_id)
            """
        )

        rows_after = int(scalar(connection, f"SELECT COUNT(*) FROM {next_table}"))
        removed = rows_before - rows_after
        total_removed += removed
        users_after = int(scalar(connection, f"SELECT COUNT(DISTINCT user_id) FROM {next_table}"))
        items_after = int(scalar(connection, f"SELECT COUNT(DISTINCT item_id) FROM {next_table}"))
        print(
            f"[frequency] iteration={iterations} rows={rows_after} "
            f"users={users_after} items={items_after} removed={removed}",
            flush=True,
        )

        if removed == 0:
            return next_table, iterations, total_removed

        if current_table != "resolved":
            connection.execute(f"DROP TABLE {current_table}")
        current_table = next_table
        next_table = "core_b" if next_table == "core_a" else "core_a"


def sequence_stats(connection: duckdb.DuckDBPyConnection, final_table: str) -> tuple[dict[str, float | int], dict[str, dict[str, float | int]]]:
    row = connection.execute(
        f"""
        WITH sequence_lengths AS (
            SELECT user_id, COUNT(*)::BIGINT AS sequence_length
            FROM {final_table}
            GROUP BY user_id
        )
        SELECT
            COUNT(*) AS user_count,
            MIN(sequence_length) AS min_length,
            MAX(sequence_length) AS max_length,
            AVG(sequence_length) AS mean_length,
            QUANTILE_CONT(sequence_length, 0.5) AS median_length,
            QUANTILE_CONT(sequence_length, 0.75) AS p75_length,
            QUANTILE_CONT(sequence_length, 0.9) AS p90_length,
            QUANTILE_CONT(sequence_length, 0.95) AS p95_length,
            QUANTILE_CONT(sequence_length, 0.99) AS p99_length,
            COALESCE(SUM(CASE WHEN sequence_length BETWEEN 50 AND 99 THEN 1 ELSE 0 END), 0) AS bin_50_99,
            COALESCE(SUM(CASE WHEN sequence_length BETWEEN 100 AND 199 THEN 1 ELSE 0 END), 0) AS bin_100_199,
            COALESCE(SUM(CASE WHEN sequence_length BETWEEN 200 AND 499 THEN 1 ELSE 0 END), 0) AS bin_200_499,
            COALESCE(SUM(CASE WHEN sequence_length BETWEEN 500 AND 999 THEN 1 ELSE 0 END), 0) AS bin_500_999,
            COALESCE(SUM(CASE WHEN sequence_length >= 1000 THEN 1 ELSE 0 END), 0) AS bin_1000_plus
        FROM sequence_lengths
        """
    ).fetchone()
    user_count = int(row[0])
    if user_count == 0:
        empty_lengths = {key: 0 for key in ("min", "max", "mean", "median", "p75", "p90", "p95", "p99")}
        empty_distribution = {
            label: {"users": 0, "ratio": 0.0}
            for label in ("50-99", "100-199", "200-499", "500-999", "1000+")
        }
        return empty_lengths, empty_distribution

    lengths = {
        "min": int(row[1]),
        "max": int(row[2]),
        "mean": float(row[3]),
        "median": float(row[4]),
        "p75": float(row[5]),
        "p90": float(row[6]),
        "p95": float(row[7]),
        "p99": float(row[8]),
    }
    bins = row[9:]
    labels = ("50-99", "100-199", "200-499", "500-999", "1000+")
    distribution = {
        label: {"users": int(count), "ratio": int(count) / user_count}
        for label, count in zip(labels, bins)
    }
    return lengths, distribution


def build_stats(
    connection: duckdb.DuckDBPyConnection,
    final_table: str,
    input_stats: InputStats,
    deduplicated_rows: int,
    category_conflict_items: int,
    iterations: int,
    frequency_rows_removed: int,
    min_user_interactions: int,
    min_item_interactions: int,
) -> dict[str, Any]:
    clean_rows = int(scalar(connection, f"SELECT COUNT(*) FROM {final_table}"))
    sequence_length, sequence_length_distribution = sequence_stats(connection, final_table)
    return {
        **asdict(input_stats),
        "duplicate_rows_removed": input_stats.pv_rows
        - input_stats.out_of_range_rows_removed
        - deduplicated_rows,
        "category_conflict_items": category_conflict_items,
        "frequency_filter": {
            "min_user_interactions": min_user_interactions,
            "min_item_interactions": min_item_interactions,
            "iterations": iterations,
            "rows_removed": frequency_rows_removed,
        },
        "clean_rows": clean_rows,
        "user_count": int(scalar(connection, f"SELECT COUNT(DISTINCT user_id) FROM {final_table}")),
        "item_count": int(scalar(connection, f"SELECT COUNT(DISTINCT item_id) FROM {final_table}")),
        "category_count": int(scalar(connection, f"SELECT COUNT(DISTINCT category_id) FROM {final_table}")),
        "sequence_length": sequence_length,
        "sequence_length_distribution": sequence_length_distribution,
    }


def write_outputs(
    connection: duckdb.DuckDBPyConnection,
    final_table: str,
    output_dir: Path,
    stats: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "user_behavior_clean.parquet"
    stats_path = output_dir / "clean_stats.json"
    parquet_tmp = output_dir / f".{parquet_path.name}.{os.getpid()}.tmp"
    stats_tmp = output_dir / f".{stats_path.name}.{os.getpid()}.tmp"

    try:
        connection.execute(
            f"""
            COPY (
                SELECT user_id, item_id, category_id, behavior_type, timestamp
                FROM {final_table}
                ORDER BY user_id ASC, timestamp ASC, item_id ASC
            ) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)
            """,
            [str(parquet_tmp)],
        )
        stats_tmp.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(parquet_tmp, parquet_path)
        os.replace(stats_tmp, stats_path)
    finally:
        parquet_tmp.unlink(missing_ok=True)
        stats_tmp.unlink(missing_ok=True)

    return parquet_path, stats_path


def validate_arguments(args: argparse.Namespace) -> None:
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.min_user_interactions < 1 or args.min_item_interactions < 1:
        raise ValueError("frequency thresholds must be positive")
    if not args.input.exists():
        raise FileNotFoundError(f"Raw input does not exist: {args.input}")
    if not args.input.is_file():
        raise ValueError(f"Raw input is not a file: {args.input}")

    protected_paths = {Path("/").resolve(), REPO_ROOT.resolve(), args.output_dir.resolve()}
    if args.work_dir.resolve() in protected_paths:
        raise ValueError("--work-dir must be a dedicated directory, not the repository or output directory")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean Taobao UserBehavior.csv according to design/淘宝数据清洗方案.md."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help=f"Raw CSV path (default: {DEFAULT_INPUT})")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Temporary working directory (default: <output-dir>/.preprocess_taobao_tmp).",
    )
    parser.add_argument("--batch-size", type=int, default=1_000_000, help="Rows per staging Parquet row group.")
    parser.add_argument("--min-user-interactions", type=int, default=50)
    parser.add_argument("--min-item-interactions", type=int, default=20)
    parser.add_argument("--keep-work-dir", action="store_true", help="Keep intermediate files after a successful run.")
    args = parser.parse_args()
    args.input = args.input.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.work_dir = (
        args.work_dir.expanduser().resolve()
        if args.work_dir is not None
        else args.output_dir / ".preprocess_taobao_tmp"
    )
    return args


def main() -> None:
    args = parse_args()
    validate_arguments(args)
    if args.work_dir.exists():
        shutil.rmtree(args.work_dir)
    args.work_dir.mkdir(parents=True)

    staging_path = args.work_dir / "valid_pv_in_range.parquet"
    duckdb_temp_dir = args.work_dir / "duckdb_temp"
    duckdb_temp_dir.mkdir()
    connection: duckdb.DuckDBPyConnection | None = None
    completed = False

    try:
        print(f"[input] validating {args.input}", flush=True)
        input_stats = stream_validate_raw(args.input, staging_path, args.batch_size)
        print(
            "[input] "
            + " ".join(f"{name}={value}" for name, value in asdict(input_stats).items()),
            flush=True,
        )

        connection = duckdb.connect(database=":memory:")
        connection.execute("SET temp_directory = ?", [str(duckdb_temp_dir)])
        print("[deduplicate] removing exact duplicate PV records", flush=True)
        deduplicated_rows, category_conflict_items = create_resolved_table(connection, staging_path)
        in_range_pv_rows = input_stats.pv_rows - input_stats.out_of_range_rows_removed
        print(
            f"[deduplicate] rows={deduplicated_rows} "
            f"removed={in_range_pv_rows - deduplicated_rows} "
            f"category_conflict_items={category_conflict_items}",
            flush=True,
        )

        final_table, iterations, frequency_rows_removed = iterative_frequency_filter(
            connection,
            args.min_user_interactions,
            args.min_item_interactions,
        )
        stats = build_stats(
            connection,
            final_table,
            input_stats,
            deduplicated_rows,
            category_conflict_items,
            iterations,
            frequency_rows_removed,
            args.min_user_interactions,
            args.min_item_interactions,
        )
        parquet_path, stats_path = write_outputs(connection, final_table, args.output_dir, stats)
        completed = True
        print(f"[done] clean data: {parquet_path}", flush=True)
        print(f"[done] statistics: {stats_path}", flush=True)
    finally:
        if connection is not None:
            connection.close()
        if completed and not args.keep_work_dir:
            shutil.rmtree(args.work_dir, ignore_errors=True)
        elif not completed:
            print(f"[error] intermediate files retained at: {args.work_dir}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
