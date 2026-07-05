from __future__ import annotations

import torch

from model import SASRec


@torch.no_grad()
def evaluate_sampled(
    model: SASRec,
    histories: torch.Tensor,
    candidates: torch.Tensor,
    device: torch.device,
    batch_size: int,
    k: int = 10,
) -> dict[str, float]:
    model.eval()
    hits = 0.0
    ndcg = 0.0
    auc = 0.0
    total = histories.size(0)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_histories = histories[start:end].to(device)
        batch_candidates = candidates[start:end].to(device)
        scores = model.score_candidates(batch_histories, batch_candidates)
        target_scores = scores[:, :1]
        negative_scores = scores[:, 1:]
        ranks = (scores > target_scores).sum(dim=1)
        hits += (ranks < k).float().sum().item()
        ndcg += torch.where(
            ranks < k,
            1.0 / torch.log2(ranks.float() + 2.0),
            torch.zeros_like(ranks, dtype=torch.float),
        ).sum().item()
        auc += (target_scores > negative_scores).float().mean(dim=1).sum().item()

    return {f"hr@{k}": hits / total, f"ndcg@{k}": ndcg / total, "auc": auc / total}
