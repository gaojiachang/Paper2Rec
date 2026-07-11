from __future__ import annotations

import numpy as np
import torch

try:
    from .model import SASRec
except ImportError:  # pragma: no cover - direct script invocation
    from model import SASRec


@torch.no_grad()
def evaluate_sampled(
    model: SASRec,
    histories: np.ndarray,
    candidates: np.ndarray,
    device: torch.device,
    batch_size: int,
    k: int = 10,
) -> dict[str, float]:
    model.eval()
    hits = 0.0
    ndcg = 0.0
    auc = 0.0
    mrr = 0.0
    total = len(histories)

    for start in range(0, total, batch_size):
        batch_histories = torch.tensor(
            histories[start : start + batch_size], dtype=torch.long, device=device
        )
        batch_candidates = torch.tensor(
            candidates[start : start + batch_size], dtype=torch.long, device=device
        )
        scores = model.score_candidates(batch_histories, batch_candidates)
        target_scores = scores[:, :1]
        negative_scores = scores[:, 1:]
        greater = (negative_scores > target_scores).sum(dim=1, dtype=torch.float32)
        lower = (negative_scores < target_scores).sum(dim=1, dtype=torch.float32)
        ties = (negative_scores == target_scores).sum(dim=1, dtype=torch.float32)
        ranks = 1.0 + greater + 0.5 * ties
        hits += (ranks <= k).float().sum().item()
        ndcg += torch.where(
            ranks <= k,
            1.0 / torch.log2(ranks + 1.0),
            torch.zeros_like(ranks),
        ).sum().item()
        auc += ((lower + 0.5 * ties) / negative_scores.size(1)).sum().item()
        mrr += (1.0 / ranks).sum().item()

    return {
        f"hr@{k}": hits / total,
        f"ndcg@{k}": ndcg / total,
        "auc": auc / total,
        "mrr": mrr / total,
    }
