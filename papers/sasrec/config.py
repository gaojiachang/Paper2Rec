from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

COMMON_DEFAULTS = {
    "output_dir": REPO_ROOT / "outputs",
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
    "epochs": 1500,
    "patience": 50,
}


DATASET_DEFAULTS = {
    "ml-1m": {
        "source_path": REPO_ROOT / "data/processed/ml-1m/interactions_5core.tsv",
        "dataset_dir": REPO_ROOT / "data/processed/ml-1m/sasrec",
        "max_seq_len": 200,
        "num_heads": 1,
        "dropout": 0.2,

    },
    "amazon-beauty": {
        "source_path": REPO_ROOT / "data/processed/amazon-beauty/interactions_5core.tsv",
        "dataset_dir": REPO_ROOT / "data/processed/amazon-beauty/sasrec",
        "max_seq_len": 50,
        "num_heads": 1,
        "dropout": 0.5,

    },
    "amazon-books": {
        "source_path": REPO_ROOT / "data/processed/amazon-books/interactions_5core.tsv",
        "dataset_dir": REPO_ROOT / "data/processed/amazon-books/sasrec",
        "max_seq_len": 50,
        "num_heads": 2,
        "dropout": 0.5,
    },
}


@dataclass
class RunConfig:
    dataset: str
    dataset_dir: str
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
    patience: int
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
    skip = {"dataset", "dataset_dir", "output_dir", "fast_dev_run"}
    values = {
        name: _resolve(args, defaults, name)
        for name in RunConfig.__dataclass_fields__
        if name not in skip
    }
    if args.fast_dev_run:
        values["epochs"] = 1

    return RunConfig(
        dataset=args.dataset,
        dataset_dir=str(Path(defaults["dataset_dir"]).resolve()),
        output_dir=str(
            Path(_resolve(args, defaults, "output_dir"))
            / "sasrec"
            / args.dataset
            / (args.run_id or time.strftime("%Y%m%d-%H%M%S"))
        ),
        fast_dev_run=args.fast_dev_run,
        **values,
    )
