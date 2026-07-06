from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path


COMMON_DEFAULTS = {
    "output_dir": "outputs",
    "seed": 2026,
    "num_blocks": 2,
    "lr": 0.001,
    "eval_negatives": 100,
    "eval_batch_size": 512,
    "fast_users": 1024,
    "fast_batches": 20,
}


DATASET_DEFAULTS = {
    "ml-1m": {
        "path": Path("data/processed/ml-1m/interactions_5core.tsv"),
        "max_seq_len": 200,
        "hidden_size": 50,
        "num_heads": 1,
        "dropout": 0.2,
        "batch_size": 128,
        "adam_beta2": 0.98,
        "epochs": 201,
    },
    "amazon-beauty": {
        "path": Path("data/processed/amazon-beauty/interactions_5core.tsv"),
        "max_seq_len": 50,
        "hidden_size": 50,
        "num_heads": 1,
        "dropout": 0.5,
        "batch_size": 128,
        "adam_beta2": 0.98,
        "epochs": 201,
    },
    "amazon-books": {
        "path": Path("data/processed/amazon-books/interactions_5core.tsv"),
        "max_seq_len": 50,
        "hidden_size": 64,
        "num_heads": 2,
        "dropout": 0.5,
        "batch_size": 512,
        "adam_beta2": 0.999,
        "epochs": 500,
    },
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
    adam_beta2: float
    epochs: int
    eval_negatives: int
    eval_batch_size: int
    fast_dev_run: bool
    fast_users: int
    fast_batches: int


def _resolve(args: argparse.Namespace, defaults: dict, name: str):
    value = getattr(args, name)
    if value is not None:
        return value
    return defaults[name]


def build_config(args: argparse.Namespace) -> RunConfig:
    defaults = {**COMMON_DEFAULTS, **DATASET_DEFAULTS[args.dataset]}
    epochs = 1 if args.fast_dev_run else _resolve(args, defaults, "epochs")
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    output_root = _resolve(args, defaults, "output_dir")
    output_dir = Path(output_root) / "sasrec" / args.dataset / run_id

    return RunConfig(
        dataset=args.dataset,
        data_path=str(defaults["path"]),
        output_dir=str(output_dir),
        seed=_resolve(args, defaults, "seed"),
        max_seq_len=_resolve(args, defaults, "max_seq_len"),
        hidden_size=_resolve(args, defaults, "hidden_size"),
        num_blocks=_resolve(args, defaults, "num_blocks"),
        num_heads=_resolve(args, defaults, "num_heads"),
        dropout=_resolve(args, defaults, "dropout"),
        batch_size=_resolve(args, defaults, "batch_size"),
        lr=_resolve(args, defaults, "lr"),
        adam_beta2=_resolve(args, defaults, "adam_beta2"),
        epochs=epochs,
        eval_negatives=_resolve(args, defaults, "eval_negatives"),
        eval_batch_size=_resolve(args, defaults, "eval_batch_size"),
        fast_dev_run=args.fast_dev_run,
        fast_users=_resolve(args, defaults, "fast_users"),
        fast_batches=_resolve(args, defaults, "fast_batches"),
    )
