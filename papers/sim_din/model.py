"""Candidate-aware DIN, SIM-long, and short-long fusion models."""

from __future__ import annotations

import math

import torch
from torch import nn

try:
    from .config import RunConfig
except ImportError:  # pragma: no cover - direct script execution.
    from config import RunConfig


class Dice(nn.Module):
    """Data-adaptive activation used by the classic DIN local activation unit."""

    def __init__(self, features: int, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(features, eps=epsilon)
        self.alpha = nn.Parameter(torch.zeros(features))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        original_shape = values.shape
        flattened = values.reshape(-1, original_shape[-1])
        probability = torch.sigmoid(self.norm(flattened)).reshape(original_shape)
        return probability * values + (1.0 - probability) * self.alpha * values


class DinTargetAttention(nn.Module):
    def __init__(self, behavior_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(behavior_dim * 4, 80),
            Dice(80),
            nn.Linear(80, 40),
            Dice(40),
            nn.Linear(40, 1),
        )

    def forward(
        self,
        behavior_embeddings: torch.Tensor,
        behavior_mask: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Return a candidate-specific short-interest vector for each group."""
        batch_size, sequence_length, behavior_dim = behavior_embeddings.shape
        candidate_count = candidate_embeddings.size(1)
        behaviors = behavior_embeddings[:, None, :, :].expand(
            batch_size, candidate_count, sequence_length, behavior_dim
        )
        candidates = candidate_embeddings[:, :, None, :].expand_as(behaviors)
        features = torch.cat((behaviors, candidates, behaviors - candidates, behaviors * candidates), dim=-1)
        scores = self.score(features).squeeze(-1)
        mask = behavior_mask[:, None, :]
        # ``-1e4`` avoids all--inf softmax under mixed precision; rows without
        # valid behavior are explicitly zeroed after attention.
        scores = scores.masked_fill(~mask, -1.0e4)
        weights = torch.softmax(scores, dim=-1)
        has_behavior = mask.any(dim=-1)
        weights = torch.where(has_behavior[:, :, None], weights, torch.zeros_like(weights))
        return torch.einsum("bcl,bcld->bcd", weights, behaviors)


def hard_search_last_k(
    long_items: torch.Tensor,
    long_categories: torch.Tensor,
    long_timestamps: torch.Tensor,
    candidate_categories: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorised candidate-category Last-K selection in chronological order.

    ``long_*`` are shared per group with shape ``[B, L]`` while candidates are
    ``[B, C]``.  The returned K-length sequences are left-padded and never
    include a PAD item as a genuine match.
    """
    batch_size, sequence_length = long_items.shape
    if k > sequence_length:
        raise ValueError("Hard Search K cannot exceed the provided long sequence length.")
    candidates = candidate_categories.size(1)
    valid_matches = (long_items[:, None, :] != 0) & (
        long_categories[:, None, :] == candidate_categories[:, :, None]
    )
    position_ids = torch.arange(sequence_length, device=long_items.device).view(1, 1, sequence_length)
    position_scores = torch.where(valid_matches, position_ids, torch.full_like(position_ids, -1))
    top_positions, top_indices = torch.topk(position_scores, k=k, dim=-1, largest=True, sorted=True)
    selected_valid = top_positions >= 0
    # topk yields newest first.  Reversing produces chronological values with
    # invalid entries on the left, exactly matching the requested left padding.
    top_indices = top_indices.flip(dims=(-1,))
    selected_valid = selected_valid.flip(dims=(-1,))
    safe_indices = top_indices.clamp_min(0)

    def gather(sequence: torch.Tensor) -> torch.Tensor:
        expanded = sequence[:, None, :].expand(batch_size, candidates, sequence_length)
        values = torch.gather(expanded, dim=2, index=safe_indices)
        return torch.where(selected_valid, values, torch.zeros_like(values))

    return gather(long_items), gather(long_categories), gather(long_timestamps), selected_valid


class MultiHeadTargetAttention(nn.Module):
    """Candidate query attention over Hard Search-selected long behavior."""

    def __init__(self, query_dim: int, value_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if value_dim % heads != 0:
            raise ValueError("Long behavior dimension must be divisible by attention_heads.")
        self.heads = heads
        self.head_dim = value_dim // heads
        self.query = nn.Linear(query_dim, value_dim)
        self.key = nn.Linear(value_dim, value_dim)
        self.value = nn.Linear(value_dim, value_dim)
        self.output = nn.Linear(value_dim, value_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        long_embeddings: torch.Tensor,
        long_mask: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, candidate_count, sequence_length, value_dim = long_embeddings.shape
        query = self.query(candidate_embeddings).view(
            batch_size, candidate_count, self.heads, self.head_dim
        )
        key = self.key(long_embeddings).view(
            batch_size, candidate_count, sequence_length, self.heads, self.head_dim
        )
        value = self.value(long_embeddings).view(
            batch_size, candidate_count, sequence_length, self.heads, self.head_dim
        )
        scores = torch.einsum("bchd,bckhd->bchk", query, key) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~long_mask[:, :, None, :], -1.0e4)
        weights = torch.softmax(scores, dim=-1)
        has_long_interest = long_mask.any(dim=-1)
        weights = torch.where(
            has_long_interest[:, :, None, None], weights, torch.zeros_like(weights)
        )
        weights = self.dropout(weights)
        context = torch.einsum("bchk,bckhd->bchd", weights, value).reshape(
            batch_size, candidate_count, value_dim
        )
        context = self.output(context)
        return torch.where(has_long_interest[:, :, None], context, torch.zeros_like(context))


class PredictionMLP(nn.Module):
    def __init__(self, input_dim: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 200),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(200, 80),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(80, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features).squeeze(-1)


class SimDinModel(nn.Module):
    """Unified model whose ``model`` config selects DIN, SIM-long, or fusion."""

    def __init__(self, config: RunConfig, num_items: int, num_categories: int) -> None:
        super().__init__()
        self.model_name = config.model
        self.short_len = config.short_len
        self.long_len = config.long_len
        self.hard_search_k = config.hard_search_k
        self.time_bucket_count = config.time_bucket_count
        self.item_embedding = nn.Embedding(num_items, config.item_embedding_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(num_categories, config.category_embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(config.dropout)

        self.behavior_dim = config.item_embedding_dim + config.category_embedding_dim
        self.long_behavior_dim = self.behavior_dim + config.time_embedding_dim
        if self.model_name in {"din", "ours"}:
            self.din_attention = DinTargetAttention(self.behavior_dim)
        if self.model_name in {"sim", "ours"}:
            self.time_embedding = nn.Embedding(
                config.time_bucket_count + 1, config.time_embedding_dim, padding_idx=0
            )
            self.esu_attention = MultiHeadTargetAttention(
                self.behavior_dim,
                self.long_behavior_dim,
                config.attention_heads,
                config.dropout,
            )

        input_dim = {
            "din": self.behavior_dim * 2,
            "sim": self.behavior_dim + self.long_behavior_dim,
            "ours": self.behavior_dim * 2 + self.long_behavior_dim,
        }[self.model_name]
        self.predictor = PredictionMLP(input_dim, config.dropout)

    def _behavior_embedding(self, items: torch.Tensor, categories: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.item_embedding(items), self.category_embedding(categories)), dim=-1)

    def _time_bucket_ids(
        self,
        selected_timestamps: torch.Tensor,
        selected_mask: torch.Tensor,
        target_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        delta_seconds = (target_timestamps[:, None, None] - selected_timestamps).clamp_min(1)
        buckets = torch.floor(torch.log2(delta_seconds.to(torch.float32))).to(torch.long) + 1
        buckets = buckets.clamp_max(self.time_bucket_count)
        return torch.where(selected_mask, buckets, torch.zeros_like(buckets))

    def _short_interest(
        self,
        short_items: torch.Tensor,
        short_categories: torch.Tensor,
        candidate_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        short_embeddings = self._behavior_embedding(short_items, short_categories)
        return self.din_attention(short_embeddings, short_items != 0, candidate_embeddings)

    def _long_interest(
        self,
        long_items: torch.Tensor,
        long_categories: torch.Tensor,
        long_timestamps: torch.Tensor,
        candidate_categories: torch.Tensor,
        candidate_embeddings: torch.Tensor,
        target_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        selected_items, selected_categories, selected_timestamps, selected_mask = hard_search_last_k(
            long_items,
            long_categories,
            long_timestamps,
            candidate_categories,
            self.hard_search_k,
        )
        time_ids = self._time_bucket_ids(selected_timestamps, selected_mask, target_timestamps)
        long_embeddings = torch.cat(
            (
                self.item_embedding(selected_items),
                self.category_embedding(selected_categories),
                self.time_embedding(time_ids),
            ),
            dim=-1,
        )
        return self.esu_attention(self.dropout(long_embeddings), selected_mask, candidate_embeddings)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        candidate_items = batch["candidate_items"]
        candidate_categories = batch["candidate_categories"]
        candidate_embeddings = self._behavior_embedding(candidate_items, candidate_categories)
        features: list[torch.Tensor] = [candidate_embeddings]
        if self.model_name in {"din", "ours"}:
            features.append(
                self._short_interest(
                    batch["short_items"], batch["short_categories"], candidate_embeddings
                )
            )
        if self.model_name in {"sim", "ours"}:
            features.append(
                self._long_interest(
                    batch["long_items"],
                    batch["long_categories"],
                    batch["long_timestamps"],
                    candidate_categories,
                    candidate_embeddings,
                    batch["target_timestamps"],
                )
            )
        return self.predictor(torch.cat(features, dim=-1))
