from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path


DATASET_DEFAULTS = {
    "ml-1m": {
        "path": Path("data/processed/ml-1m/interactions_5core.tsv"),
        "max_seq_len": 200,
        "dropout": 0.2,
        "batch_size": 512,
    },
    "amazon-beauty": {
        "path": Path("data/processed/amazon-beauty/interactions_5core.tsv"),
        "max_seq_len": 50,
        "dropout": 0.5,
        "batch_size": 512,
    },
    "amazon-books": {
        "path": Path("data/processed/amazon-books/interactions_5core.tsv"),
        "max_seq_len": 50,
        "dropout": 0.5,
        "batch_size": 512,
    },
}


COMMON_DEFAULTS = {
    "output_dir": "outputs",
    "seed": 2026,
    "hidden_size": 64,
    "num_blocks": 2,
    "num_heads": 2,
    "lr": 0.001,
    "epochs": 500,
    "eval_negatives": 100,
    "eval_batch_size": 512,
    "fast_users": 1024,
    "fast_batches": 20,
}


@dataclass
class RunConfig:
    dataset: str
    data_path: str
    output_dir: str
    seed: int
    max_seq_len: int
    hidden_size: int
    num_blocks: int
    num_heads: int
    dropout: float
    batch_size: int
    lr: float
    epochs: int
    eval_negatives: int
    eval_batch_size: int
    fast_dev_run: bool
    fast_users: int
    fast_batches: int


def _arg_or_default(args: argparse.Namespace, name: str):
    value = getattr(args, name)
    if value is not None:
        return value
    return COMMON_DEFAULTS[name]


def build_config(args: argparse.Namespace) -> RunConfig:
    defaults = DATASET_DEFAULTS[args.dataset]
    max_seq_len = args.max_seq_len or defaults["max_seq_len"]
    dropout = args.dropout if args.dropout is not None else defaults["dropout"]
    batch_size = args.batch_size or defaults["batch_size"]
    epochs = 1 if args.fast_dev_run else _arg_or_default(args, "epochs")
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    output_root = _arg_or_default(args, "output_dir")
    output_dir = Path(output_root) / "sasrec" / args.dataset / run_id

    return RunConfig(
        dataset=args.dataset,
        data_path=str(defaults["path"]),
        output_dir=str(output_dir),
        seed=_arg_or_default(args, "seed"),
        max_seq_len=max_seq_len,
        hidden_size=_arg_or_default(args, "hidden_size"),
        num_blocks=_arg_or_default(args, "num_blocks"),
        num_heads=_arg_or_default(args, "num_heads"),
        dropout=dropout,
        batch_size=batch_size,
        lr=_arg_or_default(args, "lr"),
        epochs=epochs,
        eval_negatives=_arg_or_default(args, "eval_negatives"),
        eval_batch_size=_arg_or_default(args, "eval_batch_size"),
        fast_dev_run=args.fast_dev_run,
        fast_users=_arg_or_default(args, "fast_users"),
        fast_batches=_arg_or_default(args, "fast_batches"),
    )
