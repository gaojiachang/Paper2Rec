from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import build_config
from data import SASRecTrainDataset, build_eval_examples, load_sequence_data
from evaluate import evaluate_sampled
from model import SASRec
from utils import choose_device, describe_device, save_json, set_seed


def sasrec_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    positive_ids: torch.Tensor,
) -> torch.Tensor:
    mask = positive_ids != 0
    positive_loss = F.binary_cross_entropy_with_logits(
        positive_logits[mask],
        torch.ones_like(positive_logits[mask]),
    )
    negative_loss = F.binary_cross_entropy_with_logits(
        negative_logits[mask],
        torch.zeros_like(negative_logits[mask]),
    )
    return positive_loss + negative_loss


def train_one_epoch(
    model: SASRec,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    fast_batches: int | None,
) -> float:
    model.train()
    total_loss = 0.0
    steps = 0

    for batch in loader:
        sequences = batch["sequences"].to(device)
        positive_ids = batch["positive_ids"].to(device)
        negative_ids = batch["negative_ids"].to(device)

        optimizer.zero_grad(set_to_none=True)
        positive_logits, negative_logits = model.forward_logits(
            sequences,
            positive_ids,
            negative_ids,
        )
        loss = sasrec_loss(positive_logits, negative_logits, positive_ids)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1
        if fast_batches is not None and steps >= fast_batches:
            break

    return total_loss / max(steps, 1)


def run(args: argparse.Namespace) -> None:
    config = build_config(args)
    set_seed(config.seed)
    device = choose_device()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", asdict(config))
    writer = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    try:
        print(f"dataset={config.dataset}")
        print(f"data_path={config.data_path}")
        print(f"output_dir={config.output_dir}")
        print(f"tensorboard_dir={output_dir / 'tensorboard'}")
        print(f"device={describe_device(device)}")

        data = load_sequence_data(Path(config.data_path), config.fast_dev_run, config.fast_users)
        print(
            "loaded "
            f"users={data.num_users} items={data.num_items} interactions={data.num_interactions} "
            "padding_item_id=0 first_real_item_id=1"
        )
        if data.num_users == 0:
            raise ValueError("No users with at least three interactions were loaded.")

        train_dataset = SASRecTrainDataset(
            train_sequences=data.train_sequences,
            seen_items=data.seen_items,
            num_items=data.num_items,
            max_seq_len=config.max_seq_len,
            seed=config.seed,
        )
        loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        model = SASRec(
            num_items=data.num_items,
            max_seq_len=config.max_seq_len,
            hidden_size=config.hidden_size,
            num_blocks=config.num_blocks,
            num_heads=config.num_heads,
            dropout=config.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.lr,
            betas=(0.9, config.adam_beta2),
        )

        print("building fixed sampled-eval negatives")
        valid_histories, valid_candidates = build_eval_examples(
            data,
            split="valid",
            max_seq_len=config.max_seq_len,
            num_negatives=config.eval_negatives,
            seed=config.seed,
        )
        test_histories, test_candidates = build_eval_examples(
            data,
            split="test",
            max_seq_len=config.max_seq_len,
            num_negatives=config.eval_negatives,
            seed=config.seed,
        )

        best_ndcg = -1.0
        best_epoch = 0
        metrics: list[dict[str, object]] = []
        best_path = output_dir / "best.pt"

        fast_batches = config.fast_batches if config.fast_dev_run else None
        with tqdm(
            total=config.epochs,
            desc="epochs",
            unit="epoch",
            dynamic_ncols=True,
        ) as epoch_progress:
            for epoch in range(1, config.epochs + 1):
                loss = train_one_epoch(
                    model,
                    loader,
                    optimizer,
                    device,
                    fast_batches,
                )
                valid_metrics = evaluate_sampled(
                    model,
                    valid_histories,
                    valid_candidates,
                    device,
                    config.eval_batch_size,
                    k=10,
                )
                row = {"epoch": epoch, "loss": loss, "valid": valid_metrics}
                metrics.append(row)

                if valid_metrics["ndcg@10"] > best_ndcg:
                    best_ndcg = valid_metrics["ndcg@10"]
                    best_epoch = epoch
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "config": asdict(config),
                            "num_items": data.num_items,
                            "epoch": epoch,
                            "valid": valid_metrics,
                        },
                        best_path,
                    )

                writer.add_scalar("train/loss", loss, epoch)
                writer.add_scalar("valid/hr@10", valid_metrics["hr@10"], epoch)
                writer.add_scalar("valid/ndcg@10", valid_metrics["ndcg@10"], epoch)
                writer.add_scalar("valid/auc", valid_metrics["auc"], epoch)
                writer.add_scalar("meta/best_valid_ndcg@10", best_ndcg, epoch)
                epoch_progress.set_postfix(
                    loss=f"{loss:.4f}",
                    **{
                        "valid_hr@10": f"{valid_metrics['hr@10']:.4f}",
                        "valid_ndcg@10": f"{valid_metrics['ndcg@10']:.4f}",
                        "valid_auc": f"{valid_metrics['auc']:.4f}",
                    },
                    refresh=False,
                )
                epoch_progress.update(1)

        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate_sampled(
            model,
            test_histories,
            test_candidates,
            device,
            config.eval_batch_size,
            k=10,
        )
        writer.add_scalar("test/hr@10", test_metrics["hr@10"], config.epochs)
        writer.add_scalar("test/ndcg@10", test_metrics["ndcg@10"], config.epochs)
        writer.add_scalar("test/auc", test_metrics["auc"], config.epochs)

        summary = {
            "best_epoch": best_epoch,
            "best_valid_ndcg@10": best_ndcg,
            "test": test_metrics,
            "history": metrics,
        }
        save_json(output_dir / "metrics.json", summary)
        print(
            f"best_epoch={best_epoch} "
            f"test_hr@10={test_metrics['hr@10']:.4f} "
            f"test_ndcg@10={test_metrics['ndcg@10']:.4f} "
            f"test_auc={test_metrics['auc']:.4f}"
        )
        print(f"saved best checkpoint: {best_path}")
        print(f"saved metrics: {output_dir / 'metrics.json'}")
        print(f"saved tensorboard logs: {output_dir / 'tensorboard'}")
    finally:
        writer.close()
