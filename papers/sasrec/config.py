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
    "hidden_size": 50,
    "adam_beta2": 0.98,
    "batch_size": 128,
    "epochs": 1500
}


DATASET_DEFAULTS = {
    "ml-1m": {
        "path": Path("data/processed/ml-1m/interactions_5core.tsv"),
        "max_seq_len": 200,
        "num_heads": 1,
        "dropout": 0.2,

    },
    "amazon-beauty": {
        "path": Path("data/processed/amazon-beauty/interactions_5core.tsv"),
        "max_seq_len": 50,
        "num_heads": 1,
        "dropout": 0.5,

    },
    "amazon-books": {
        "path": Path("data/processed/amazon-books/interactions_5core.tsv"),
        "max_seq_len": 50,
        "num_heads": 2,
        "dropout": 0.5,
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
    skip = {"dataset", "data_path", "output_dir", "fast_dev_run"}
    values = {
        name: _resolve(args, defaults, name)
        for name in RunConfig.__dataclass_fields__
        if name not in skip
    }
    if args.fast_dev_run:
        values["epochs"] = 1

    return RunConfig(
        dataset=args.dataset,
        data_path=str(defaults["path"]),
        output_dir=str(
            Path(_resolve(args, defaults, "output_dir"))
            / "sasrec"
            / args.dataset
            / (args.run_id or time.strftime("%Y%m%d-%H%M%S"))
        ),
        fast_dev_run=args.fast_dev_run,
        **values,
    )
