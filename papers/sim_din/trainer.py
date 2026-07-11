#!/usr/bin/env python3
"""CLI entrypoint for the unified Taobao DIN/SIM experiments."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from .config import RunConfig, build_config, parse_args
    from .data import (
        TrainGroupDataset,
        ensure_cache,
        evaluation_batches,
        load_train_targets,
        load_valid_subset_ids,
    )
    from .evaluate import evaluate_candidate_groups, move_batch_to_device
    from .model import SimDinModel
    from .utils import AmpController, choose_device, describe_device, save_json, set_seed
except ImportError:  # pragma: no cover - direct CLI invocation.
    from config import RunConfig, build_config, parse_args
    from data import TrainGroupDataset, ensure_cache, evaluation_batches, load_train_targets, load_valid_subset_ids
    from evaluate import evaluate_candidate_groups, move_batch_to_device
    from model import SimDinModel
    from utils import AmpController, choose_device, describe_device, save_json, set_seed


def build_model_optimizer(
    config: RunConfig, num_items: int, num_categories: int, device: torch.device
) -> tuple[SimDinModel, torch.optim.Optimizer]:
    model = SimDinModel(config, num_items, num_categories).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    return model, optimizer


def save_checkpoint(
    path: Path,
    model: SimDinModel,
    optimizer: torch.optim.Optimizer,
    config: RunConfig,
    epoch: int,
    valid_metrics: dict[str, object],
    num_items: int,
    num_categories: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "epoch": epoch,
            "valid": valid_metrics,
            "num_items": num_items,
            "num_categories": num_categories,
        },
        path,
    )


def train_one_epoch(
    model: SimDinModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: AmpController,
) -> float:
    model.train()
    total_loss = 0.0
    total_groups = 0
    for cpu_batch in loader:
        batch = move_batch_to_device(cpu_batch, device)
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast():
            logits = model(batch)
            loss = F.binary_cross_entropy_with_logits(logits, batch["labels"])
        amp.backward_step(loss, optimizer)
        group_count = logits.size(0)
        total_loss += float(loss.detach().item()) * group_count
        total_groups += group_count
    return total_loss / max(total_groups, 1)


def _write_metrics(writer: SummaryWriter, prefix: str, metrics: dict[str, object], epoch: int) -> None:
    for name in ("auc", "hr@10", "ndcg@10", "mrr"):
        writer.add_scalar(f"{prefix}/{name}", float(metrics[name]), epoch)


def _evaluation_iterator(cache, config: RunConfig, split: str, selected_ids=None):
    limit = config.fast_eval_groups if config.fast_dev_run else None
    return evaluation_batches(
        cache,
        Path(config.dataset_dir),
        split,
        config,
        selected_sample_ids=selected_ids,
        limit=limit,
    )


def run(config: RunConfig) -> dict[str, object]:
    set_seed(config.seed)
    device = choose_device(config.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    amp = AmpController(device, config.amp)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    save_json(output_dir / "config.json", asdict(config))
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    try:
        print(f"model={config.model}")
        print(f"dataset_dir={config.dataset_dir}")
        print(f"output_dir={output_dir}")
        print(f"device={describe_device(device)} amp={amp.description}")
        print("[cache] validating sequence cache", flush=True)
        cache = ensure_cache(config)
        print(
            f"[cache] users={cache.num_users} interactions={len(cache.item_sequences)} "
            f"items={cache.num_item_embeddings - 2} categories={cache.num_category_embeddings - 2}",
            flush=True,
        )
        train_limit = config.fast_train_samples if config.fast_dev_run else None
        train_targets = load_train_targets(cache, Path(config.dataset_dir), limit=train_limit)
        if not len(train_targets):
            raise ValueError("No training targets were loaded.")
        train_dataset = TrainGroupDataset(cache, train_targets, config)
        loader_generator = torch.Generator().manual_seed(config.seed)
        loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0 if config.fast_dev_run else config.num_workers,
            pin_memory=device.type == "cuda",
            generator=loader_generator,
            persistent_workers=False,
        )
        valid_subset_ids = load_valid_subset_ids(cache)
        if config.fast_dev_run:
            valid_subset_ids = valid_subset_ids[: config.fast_eval_groups]
        model, optimizer = build_model_optimizer(
            config, cache.num_item_embeddings, cache.num_category_embeddings, device
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        print(f"[model] parameters={parameter_count:,} train_groups={len(train_dataset):,}", flush=True)

        best_subset_auc = float("-inf")
        best_full_auc = float("-inf")
        best_epoch = 0
        best_path = output_dir / "best.pt"
        history: list[dict[str, Any]] = []

        progress = tqdm(range(1, config.epochs + 1), desc="epochs", unit="epoch", dynamic_ncols=True)
        for epoch in progress:
            train_dataset.set_epoch(epoch)
            train_loss = train_one_epoch(model, loader, optimizer, device, amp)
            subset_metrics = evaluate_candidate_groups(
                model,
                _evaluation_iterator(cache, config, "valid", valid_subset_ids),
                device,
                amp,
            )
            subset_auc = float(subset_metrics["auc"])
            full_metrics: dict[str, object] | None = None
            confirmed = False
            if subset_auc > best_subset_auc:
                best_subset_auc = subset_auc
                full_metrics = evaluate_candidate_groups(
                    model,
                    _evaluation_iterator(cache, config, "valid"),
                    device,
                    amp,
                )
                if float(full_metrics["auc"]) > best_full_auc:
                    best_full_auc = float(full_metrics["auc"])
                    best_epoch = epoch
                    confirmed = True
                    save_checkpoint(
                        best_path,
                        model,
                        optimizer,
                        config,
                        epoch,
                        full_metrics,
                        cache.num_item_embeddings,
                        cache.num_category_embeddings,
                    )
            writer.add_scalar("train/loss", train_loss, epoch)
            _write_metrics(writer, "valid_subset", subset_metrics, epoch)
            if full_metrics is not None:
                _write_metrics(writer, "valid_full", full_metrics, epoch)
            writer.add_scalar("meta/best_full_valid_auc", best_full_auc, epoch)
            row: dict[str, Any] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_subset": subset_metrics,
                "valid_full": full_metrics,
                "confirmed_best": confirmed,
            }
            history.append(row)
            progress.set_postfix(
                loss=f"{train_loss:.4f}",
                subset_auc=f"{subset_auc:.4f}",
                best_full_auc=f"{best_full_auc:.4f}",
                refresh=False,
            )
        if not best_path.exists():
            raise RuntimeError("No checkpoint was saved; initial full validation did not run.")
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate_candidate_groups(
            model,
            _evaluation_iterator(cache, config, "test"),
            device,
            amp,
        )
        _write_metrics(writer, "test", test_metrics, best_epoch)
        summary: dict[str, object] = {
            "best_epoch": best_epoch,
            "best_valid_auc": best_full_auc,
            "test": test_metrics,
            "history": history,
        }
        save_json(output_dir / "metrics.json", summary)
        print(
            f"[done] best_epoch={best_epoch} best_valid_auc={best_full_auc:.6f} "
            f"test_auc={float(test_metrics['auc']):.6f} test_hr@10={float(test_metrics['hr@10']):.6f}",
            flush=True,
        )
        return summary
    finally:
        writer.close()


def main() -> None:
    run(build_config(parse_args()))


if __name__ == "__main__":
    main()
