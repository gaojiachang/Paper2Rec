"""Streaming, candidate-aware sampled ranking evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

try:
    from .utils import AmpController
except ImportError:  # pragma: no cover - direct script execution.
    from utils import AmpController


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: values.to(device, non_blocking=device.type == "cuda") for name, values in batch.items()}


@dataclass
class MetricAccumulator:
    count: int = 0
    auc_sum: float = 0.0
    hr_sum: float = 0.0
    ndcg_sum: float = 0.0
    mrr_sum: float = 0.0

    def add(self, ranks: torch.Tensor, auc_values: torch.Tensor) -> None:
        if not len(ranks):
            return
        self.count += len(ranks)
        self.auc_sum += float(auc_values.sum().item())
        self.hr_sum += float((ranks <= 10.0).float().sum().item())
        self.ndcg_sum += float(torch.where(
            ranks <= 10.0,
            1.0 / torch.log2(ranks + 1.0),
            torch.zeros_like(ranks),
        ).sum().item())
        self.mrr_sum += float((1.0 / ranks).sum().item())

    def metrics(self) -> dict[str, float | int]:
        if self.count == 0:
            return {"count": 0, "auc": 0.0, "hr@10": 0.0, "ndcg@10": 0.0, "mrr": 0.0}
        return {
            "count": self.count,
            "auc": self.auc_sum / self.count,
            "hr@10": self.hr_sum / self.count,
            "ndcg@10": self.ndcg_sum / self.count,
            "mrr": self.mrr_sum / self.count,
        }


def average_tie_rank_and_auc(scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute unbiased ranks when the positive is stored at candidate index 0."""
    if scores.ndim != 2 or scores.size(1) < 2:
        raise ValueError("Expected scores shaped [groups, positive_plus_negatives].")
    positive = scores[:, :1]
    negatives = scores[:, 1:]
    greater = (negatives > positive).sum(dim=1, dtype=torch.float32)
    lower = (negatives < positive).sum(dim=1, dtype=torch.float32)
    ties = (negatives == positive).sum(dim=1, dtype=torch.float32)
    rank = 1.0 + greater + 0.5 * ties
    auc = (lower + 0.5 * ties) / negatives.size(1)
    return rank, auc


@torch.no_grad()
def evaluate_candidate_groups(
    model: torch.nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    amp: AmpController,
) -> dict[str, object]:
    """Score each candidate separately and stream aggregate metrics."""
    model.eval()
    overall = MetricAccumulator()
    for cpu_batch in batches:
        batch = move_batch_to_device(cpu_batch, device)
        with amp.autocast():
            scores = model(batch)
        ranks, auc_values = average_tie_rank_and_auc(scores.float())
        overall.add(ranks, auc_values)
    return overall.metrics()
