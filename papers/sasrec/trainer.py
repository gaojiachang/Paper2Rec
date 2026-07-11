from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    from .config import build_config
    from .data import SASRecTrainDataset, load_prepared_data
    from .evaluate import evaluate_sampled
    from .model import SASRec
    from .utils import choose_device, describe_device, save_json, set_seed
except ImportError:  # pragma: no cover - direct script invocation
    from config import build_config
    from data import SASRecTrainDataset, load_prepared_data
    from evaluate import evaluate_sampled
    from model import SASRec
    from utils import choose_device, describe_device, save_json, set_seed


VALID_METRIC_NAMES = ("auc", "hr@10", "ndcg@10", "mrr")


def sasrec_loss(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    positive_ids: torch.Tensor,
) -> torch.Tensor:
    mask = positive_ids != 0
    positive_logits = positive_logits[mask]
    negative_logits = negative_logits[mask]
    logits = torch.cat([positive_logits, negative_logits])
    labels = torch.cat([torch.ones_like(positive_logits), torch.zeros_like(negative_logits)])
    return 2 * F.binary_cross_entropy_with_logits(logits, labels)


def build_model_optimizer(config, num_items: int, device: torch.device):
    model = SASRec(
        num_items=num_items,
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
    return model, optimizer


def save_checkpoint(path: Path, model: SASRec, config, num_items: int, epoch: int, valid: dict) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "num_items": num_items,
            "epoch": epoch,
            "valid": valid,
        },
        path,
    )


def write_epoch_scalars(
    writer: SummaryWriter,
    epoch: int,
    loss: float,
    train: dict,
    valid: dict,
    best_ndcg: float,
) -> None:
    writer.add_scalar("train/loss", loss, epoch)
    for name in VALID_METRIC_NAMES:
        writer.add_scalar(f"train/{name}", train[name], epoch)
        writer.add_scalar(f"valid/{name}", valid[name], epoch)
    writer.add_scalar("meta/best_valid_ndcg@10", best_ndcg, epoch)


def write_test_scalars(writer: SummaryWriter, epoch: int, metrics: dict) -> None:
    for name, value in metrics.items():
        writer.add_scalar(f"test/{name}", value, epoch)


def train_one_epoch(
    model: SASRec,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    fast_batches: int | None,
) -> tuple[float, dict[str, float | int]]:
    model.train()
    total_loss = 0.0
    steps = 0
    total_positions = 0
    metric_sums = torch.zeros(4, dtype=torch.float64, device=device)

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

        # 每个有效训练位置只有一个正例和一个负例，据此计算在线排名指标。
        mask = positive_ids != 0
        positive_scores = positive_logits.detach()[mask].float()
        negative_scores = negative_logits.detach()[mask].float()
        greater = (negative_scores > positive_scores).to(torch.float32)
        lower = (negative_scores < positive_scores).to(torch.float32)
        ties = (negative_scores == positive_scores).to(torch.float32)
        ranks = 1.0 + greater + 0.5 * ties
        auc_values = lower + 0.5 * ties
        metric_sums += torch.stack(
            (
                auc_values.sum(),
                (ranks <= 10.0).sum(dtype=torch.float32),
                torch.where(
                    ranks <= 10.0,
                    1.0 / torch.log2(ranks + 1.0),
                    torch.zeros_like(ranks),
                ).sum(),
                (1.0 / ranks).sum(),
            )
        ).to(torch.float64)
        total_positions += positive_scores.numel()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1
        if fast_batches is not None and steps >= fast_batches:
            break

    denominator = max(total_positions, 1)
    auc_sum, hr_sum, ndcg_sum, mrr_sum = metric_sums.tolist()
    metrics: dict[str, float | int] = {
        "count": total_positions,
        "auc": auc_sum / denominator,
        "hr@10": hr_sum / denominator,
        "ndcg@10": ndcg_sum / denominator,
        "mrr": mrr_sum / denominator,
    }
    return total_loss / max(steps, 1), metrics


def update_best_valid_metrics(metrics: dict, best_metrics: dict[str, float]) -> tuple[str, ...]:
    improved = tuple(
        name for name in VALID_METRIC_NAMES if float(metrics[name]) > best_metrics[name]
    )
    for name in improved:
        best_metrics[name] = float(metrics[name])
    return improved


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
        print(f"dataset_dir={config.dataset_dir}")
        print(f"output_dir={config.output_dir}")
        print(f"tensorboard_dir={output_dir / 'tensorboard'}")
        print(f"device={describe_device(device)}")

        data = load_prepared_data(config)
        print(
            "loaded "
            f"users={data.num_users} items={data.num_items} interactions={data.num_interactions} "
            "padding_item_id=0 first_real_item_id=1"
        )
        if data.num_users == 0:
            raise ValueError("No users with at least three interactions were loaded.")

        train_dataset = SASRecTrainDataset(
            sequences=data.train_sequences,
            positive_ids=data.train_positives,
            negative_ids=data.train_negatives,
        )
        loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=device.type == "cuda",
        )

        model, optimizer = build_model_optimizer(config, data.num_items, device)

        best_ndcg = -1.0
        best_valid_metrics = {name: float("-inf") for name in VALID_METRIC_NAMES}
        epochs_without_valid_improvement = 0
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
                loss, train_metrics = train_one_epoch(
                    model,
                    loader,
                    optimizer,
                    device,
                    fast_batches,
                )
                valid_metrics = evaluate_sampled(
                    model,
                    data.valid_histories,
                    data.valid_candidates,
                    device,
                    config.eval_batch_size,
                    k=10,
                )
                improved_valid_metrics = update_best_valid_metrics(
                    valid_metrics, best_valid_metrics
                )
                if improved_valid_metrics:
                    epochs_without_valid_improvement = 0
                else:
                    epochs_without_valid_improvement += 1
                row = {
                    "epoch": epoch,
                    "loss": loss,
                    "train": train_metrics,
                    "valid": valid_metrics,
                    "improved_valid_metrics": list(improved_valid_metrics),
                    "epochs_without_valid_improvement": epochs_without_valid_improvement,
                }
                metrics.append(row)

                if valid_metrics["ndcg@10"] > best_ndcg:
                    best_ndcg = valid_metrics["ndcg@10"]
                    best_epoch = epoch
                    save_checkpoint(best_path, model, config, data.num_items, epoch, valid_metrics)

                write_epoch_scalars(
                    writer, epoch, loss, train_metrics, valid_metrics, best_ndcg
                )
                epoch_progress.set_postfix(
                    loss=f"{loss:.4f}",
                    **{
                        "valid_hr@10": f"{valid_metrics['hr@10']:.4f}",
                        "valid_ndcg@10": f"{valid_metrics['ndcg@10']:.4f}",
                        "valid_auc": f"{valid_metrics['auc']:.4f}",
                        "valid_mrr": f"{valid_metrics['mrr']:.4f}",
                        "stale": epochs_without_valid_improvement,
                    },
                    refresh=False,
                )
                epoch_progress.update(1)
                if epochs_without_valid_improvement >= config.patience:
                    print(
                        f"[early-stop] none of {', '.join(VALID_METRIC_NAMES)} improved "
                        f"for {epochs_without_valid_improvement} epochs",
                        flush=True,
                    )
                    break

        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate_sampled(
            model,
            data.test_histories,
            data.test_candidates,
            device,
            config.eval_batch_size,
            k=10,
        )
        write_test_scalars(writer, best_epoch, test_metrics)

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
