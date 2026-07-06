#!/usr/bin/env python3
"""Command-line entrypoint for the minimal SASRec baseline."""

from __future__ import annotations

import argparse
from typing import Iterable

from config import DATASET_DEFAULTS
from trainer import run


OPTION_TYPES = {
    **dict.fromkeys(("output_dir", "run_id"), str),
    **dict.fromkeys(("seed", "max_seq_len", "hidden_size", "num_blocks", "num_heads"), int),
    **dict.fromkeys(("batch_size", "epochs", "eval_negatives", "eval_batch_size"), int),
    **dict.fromkeys(("fast_users", "fast_batches"), int),
    **dict.fromkeys(("dropout", "lr", "adam_beta2"), float),
}


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/evaluate a minimal SASRec baseline.")
    parser.add_argument("--dataset", choices=sorted(DATASET_DEFAULTS), required=True)
    for name, value_type in OPTION_TYPES.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=value_type, default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
