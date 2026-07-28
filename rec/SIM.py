"""SIM（Search-based Interest Model）的独立 PyTorch 实现。

模型先按候选类目从长序列检索最近 K 条行为，再用候选感知的多头注意力聚合，
不包含任何数据处理、训练或评估代码。商品和类目 ID 的 0 均为 PAD。
"""

from __future__ import annotations

import math

import torch
from torch import nn


def hard_search_last_k(
    history_items: torch.Tensor,
    history_categories: torch.Tensor,
    candidate_categories: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """为每个候选类目检索最近 ``k`` 条同类目历史。

    返回 ``(item_ids, category_ids, valid_mask)``，形状均为 ``[B, C, K]``。
    无匹配位置左侧填 0。
    """
    batch_size, history_length = history_items.shape
    candidate_count = candidate_categories.size(1)
    selected_indices, valid_mask = _last_k_indices(
        history_items, history_categories, candidate_categories, k
    )

    def gather(values: torch.Tensor) -> torch.Tensor:
        expanded = values[:, None].expand(batch_size, candidate_count, history_length)
        result = torch.gather(expanded, 2, selected_indices)
        return torch.where(valid_mask, result, torch.zeros_like(result))

    return gather(history_items), gather(history_categories), valid_mask


def _last_k_indices(
    history_items: torch.Tensor,
    history_categories: torch.Tensor,
    candidate_categories: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回按时间正序排列的历史索引及其有效掩码。"""
    _, history_length = history_items.shape
    if not 1 <= k <= history_length:
        raise ValueError("k 必须位于 [1, history_length] 内。")
    matches = history_items[:, None].ne(0) & history_categories[:, None].eq(
        candidate_categories[:, :, None]
    )
    positions = torch.arange(history_length, device=history_items.device).view(1, 1, -1)
    scores = torch.where(matches, positions, torch.full_like(positions, -1))
    selected_positions, selected_indices = scores.topk(k, dim=-1, largest=True, sorted=True)
    return selected_indices.flip(-1).clamp_min(0), selected_positions.ge(0).flip(-1)


class TargetAttention(nn.Module):
    """以候选为查询，聚合 Hard Search 后的长期行为。"""

    def __init__(self, query_dim: int, value_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if value_dim % num_heads:
            raise ValueError("value_dim 必须可被 num_heads 整除。")
        self.num_heads = num_heads
        self.head_dim = value_dim // num_heads
        self.query = nn.Linear(query_dim, value_dim)
        self.key = nn.Linear(value_dim, value_dim)
        self.value = nn.Linear(value_dim, value_dim)
        self.output = nn.Linear(value_dim, value_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, long_history: torch.Tensor, history_mask: torch.Tensor, candidates: torch.Tensor
    ) -> torch.Tensor:
        batch_size, candidate_count, sequence_length, value_dim = long_history.shape
        query = self.query(candidates).view(batch_size, candidate_count, self.num_heads, self.head_dim)
        key = self.key(long_history).view(
            batch_size, candidate_count, sequence_length, self.num_heads, self.head_dim
        )
        value = self.value(long_history).view(
            batch_size, candidate_count, sequence_length, self.num_heads, self.head_dim
        )
        scores = torch.einsum("bchd,bckhd->bchk", query, key) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~history_mask[:, :, None], -1.0e4)
        weights = torch.softmax(scores, dim=-1)
        has_history = history_mask.any(dim=-1)
        weights = torch.where(has_history[:, :, None, None], weights, torch.zeros_like(weights))
        context = torch.einsum("bchk,bckhd->bchd", self.dropout(weights), value)
        context = context.reshape(batch_size, candidate_count, value_dim)
        return self.output(context)


class SIM(nn.Module):
    """长序列兴趣建模的 SIM 点击预测器。"""

    def __init__(
        self,
        num_items: int,
        num_categories: int,
        item_embedding_dim: int = 32,
        category_embedding_dim: int = 16,
        time_embedding_dim: int = 8,
        time_bucket_count: int = 64,
        hard_search_k: int = 50,
        num_heads: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, item_embedding_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(
            num_categories, category_embedding_dim, padding_idx=0
        )
        self.time_embedding = nn.Embedding(time_bucket_count + 1, time_embedding_dim, padding_idx=0)
        self.hard_search_k = hard_search_k
        self.time_bucket_count = time_bucket_count
        self.behavior_dim = item_embedding_dim + category_embedding_dim
        self.long_dim = self.behavior_dim + time_embedding_dim
        self.attention = TargetAttention(self.behavior_dim, self.long_dim, num_heads, dropout)
        self.predictor = nn.Sequential(
            nn.Linear(self.behavior_dim + self.long_dim, 200),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(200, 80),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(80, 1),
        )

    def embed(self, items: torch.Tensor, categories: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.item_embedding(items), self.category_embedding(categories)), dim=-1)

    def time_bucket_ids(
        self,
        selected_timestamps: torch.Tensor,
        selected_mask: torch.Tensor,
        target_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        delta = (target_timestamps[:, :, None] - selected_timestamps).clamp_min(1)
        buckets = torch.floor(torch.log2(delta.to(torch.float32))).long().add_(1)
        buckets = buckets.clamp_max(self.time_bucket_count)
        return torch.where(selected_mask, buckets, torch.zeros_like(buckets))

    def forward(
        self,
        history_items: torch.Tensor,
        history_categories: torch.Tensor,
        history_timestamps: torch.Tensor,
        candidate_items: torch.Tensor,
        candidate_categories: torch.Tensor,
        target_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        single_candidate = candidate_items.ndim == 1
        if single_candidate:
            candidate_items = candidate_items.unsqueeze(1)
            candidate_categories = candidate_categories.unsqueeze(1)
            target_timestamps = target_timestamps.unsqueeze(1)
        indices, selected_mask = _last_k_indices(
            history_items, history_categories, candidate_categories, self.hard_search_k
        )
        candidate_count = candidate_items.size(1)

        def gather(source: torch.Tensor) -> torch.Tensor:
            expanded = source[:, None].expand(-1, candidate_count, -1)
            values = torch.gather(expanded, 2, indices)
            return torch.where(selected_mask, values, torch.zeros_like(values))

        selected_items = gather(history_items)
        selected_categories = gather(history_categories)
        selected_timestamps = gather(history_timestamps)
        selected_timestamps = torch.where(selected_mask, selected_timestamps, torch.zeros_like(selected_timestamps))

        candidates = self.embed(candidate_items, candidate_categories)
        long_history = torch.cat(
            (
                self.embed(selected_items, selected_categories),
                self.time_embedding(self.time_bucket_ids(selected_timestamps, selected_mask, target_timestamps)),
            ),
            dim=-1,
        )
        interest = self.attention(long_history, selected_mask, candidates)
        logits = self.predictor(torch.cat((candidates, interest), dim=-1)).squeeze(-1)
        return logits.squeeze(1) if single_candidate else logits


if __name__ == "__main__":
    model = SIM(num_items=1_000, num_categories=100, hard_search_k=3)
    logits = model(
        history_items=torch.tensor([[0, 8, 21, 43, 61]]),
        history_categories=torch.tensor([[0, 2, 7, 5, 7]]),
        history_timestamps=torch.tensor([[0, 10, 20, 30, 40]]),
        candidate_items=torch.tensor([[101, 102]]),
        candidate_categories=torch.tensor([[7, 3]]),
        target_timestamps=torch.tensor([[50, 50]]),
    )
    print("logits:", logits)
