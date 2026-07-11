"""Command-line configuration for the unified DIN/SIM experiments."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_DIR = REPO_ROOT / "data/processed/taobao-userbehavior/sim_din"
DEFAULT_CLEAN_PATH = REPO_ROOT / "data/processed/taobao-userbehavior/user_behavior_clean.parquet"


@dataclass(frozen=True)
class RunConfig:
    model: str
    dataset_dir: str
    clean_path: str
    cache_dir: str
    output_dir: str
    seed: int
    item_embedding_dim: int
    category_embedding_dim: int
    time_embedding_dim: int
    time_bucket_count: int
    short_len: int
    long_len: int
    hard_search_k: int
    attention_heads: int
    dropout: float
    batch_size: int
    eval_batch_size: int
    num_workers: int
    learning_rate: float
    epochs: int
    patience: int
    valid_subset_size: int
    amp: bool
    device: str | None
    rebuild_cache: bool
    fast_dev_run: bool
    fast_train_samples: int
    fast_eval_groups: int


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train candidate-aware DIN, SIM-long, or short-long fusion on Taobao."
    )
    parser.add_argument("--model", choices=("din", "sim", "ours"), required=True)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--clean-path", type=Path, default=DEFAULT_CLEAN_PATH)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--item-embedding-dim", type=int, default=32)
    parser.add_argument("--category-embedding-dim", type=int, default=16)
    parser.add_argument("--time-embedding-dim", type=int, default=8)
    parser.add_argument("--time-bucket-count", type=int, default=64)
    parser.add_argument("--short-len", type=int, default=20)
    parser.add_argument("--long-len", type=int, default=500)
    parser.add_argument("--hard-search-k", type=int, default=50)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=4096, help="Number of candidate groups per train batch.")
    parser.add_argument("--eval-batch-size", type=int, default=2048, help="Number of candidate groups per evaluation batch.")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--valid-subset-size", type=int, default=50_000)
    parser.add_argument("--device", type=str, default=None, help="Torch device, e.g. cuda:0 or cpu.")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA automatic mixed precision.")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--fast-train-samples", type=int, default=1_024)
    parser.add_argument("--fast-eval-groups", type=int, default=1_024)
    return parser.parse_args(argv)


def _positive(name: str, value: int) -> int:
    if value < 1:
        raise ValueError(f"{name} must be positive; received {value}.")
    return value


def build_config(args: argparse.Namespace) -> RunConfig:
    dataset_dir = args.dataset_dir.expanduser().resolve()
    clean_path = args.clean_path.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not clean_path.is_file():
        raise FileNotFoundError(f"Clean interaction data does not exist: {clean_path}")
    for filename in (
        "train_targets.parquet",
        "valid_candidates.parquet",
        "test_candidates.parquet",
        "item_catalog.parquet",
    ):
        if not (dataset_dir / filename).is_file():
            raise FileNotFoundError(f"Missing required data-set file: {dataset_dir / filename}")

    for name in (
        "item_embedding_dim",
        "category_embedding_dim",
        "time_embedding_dim",
        "time_bucket_count",
        "short_len",
        "long_len",
        "hard_search_k",
        "attention_heads",
        "batch_size",
        "eval_batch_size",
        "epochs",
        "patience",
        "valid_subset_size",
        "fast_train_samples",
        "fast_eval_groups",
    ):
        _positive(name, getattr(args, name))
    if args.num_workers < 0:
        raise ValueError("num_workers must be non-negative")
    if args.hard_search_k > args.long_len:
        raise ValueError("hard_search_k cannot exceed long_len")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if args.learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    output_dir = args.output_dir.expanduser().resolve() / "sim_din" / args.model / run_id
    return RunConfig(
        model=args.model,
        dataset_dir=str(dataset_dir),
        clean_path=str(clean_path),
        cache_dir=str(dataset_dir / "cache"),
        output_dir=str(output_dir),
        seed=args.seed,
        item_embedding_dim=args.item_embedding_dim,
        category_embedding_dim=args.category_embedding_dim,
        time_embedding_dim=args.time_embedding_dim,
        time_bucket_count=args.time_bucket_count,
        short_len=args.short_len,
        long_len=args.long_len,
        hard_search_k=args.hard_search_k,
        attention_heads=args.attention_heads,
        dropout=args.dropout,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        epochs=1 if args.fast_dev_run else args.epochs,
        patience=args.patience,
        valid_subset_size=args.valid_subset_size,
        amp=not args.no_amp,
        device=args.device,
        rebuild_cache=args.rebuild_cache,
        fast_dev_run=args.fast_dev_run,
        fast_train_samples=args.fast_train_samples,
        fast_eval_groups=args.fast_eval_groups,
    )
