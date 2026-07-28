"""DIN（Deep Interest Network）的独立 PyTorch 实现。

输入为用户历史行为与一个或多个候选商品，不依赖数据集或训练脚本。
商品和类目 ID 均以 0 作为 PAD。
"""

from __future__ import annotations

import torch
from torch import nn


class Dice(nn.Module):
    """DIN 中的数据自适应激活函数。"""

    def __init__(self, features: int, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(features, eps=epsilon)
        self.alpha = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        x_2d = x.reshape(-1, shape[-1])
        probability = torch.sigmoid(self.norm(x_2d)).reshape(shape)
        return probability * x + (1.0 - probability) * self.alpha * x


class TargetAttention(nn.Module):
    """针对每个候选商品计算候选感知的历史兴趣。"""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(embedding_dim * 4, 80),
            Dice(80),
            nn.Linear(80, 40),
            Dice(40),
            nn.Linear(40, 1),
        )

    def forward(
        self,
        history: torch.Tensor,
        history_mask: torch.Tensor,
        candidates: torch.Tensor,
    ) -> torch.Tensor:
        """返回形状为 ``[B, C, D]`` 的候选相关兴趣向量。"""
        batch_size, history_length, embedding_dim = history.shape
        candidate_count = candidates.size(1)
        histories = history[:, None].expand(
            batch_size, candidate_count, history_length, embedding_dim
        )
        targets = candidates[:, :, None].expand_as(histories)
        features = torch.cat(
            (histories, targets, histories - targets, histories * targets), dim=-1
        )
        scores = self.score(features).squeeze(-1)
        mask = history_mask[:, None]
        weights = torch.softmax(scores.masked_fill(~mask, -1.0e4), dim=-1)
        weights = torch.where(mask.any(dim=-1, keepdim=True), weights, torch.zeros_like(weights))
        return torch.einsum("bcl,bcld->bcd", weights, histories)


class DIN(nn.Module):
    """用候选感知注意力预测点击概率的 DIN。

    ``candidate_items`` 和 ``candidate_categories`` 可为 ``[B]`` 或 ``[B, C]``；
    返回的 logit 保持相同的候选维度。
    """

    def __init__(
        self,
        num_items: int,
        num_categories: int,
        item_embedding_dim: int = 32,
        category_embedding_dim: int = 16,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items, item_embedding_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(
            num_categories, category_embedding_dim, padding_idx=0
        )
        self.embedding_dim = item_embedding_dim + category_embedding_dim
        self.attention = TargetAttention(self.embedding_dim)
        self.predictor = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, 200),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(200, 80),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(80, 1),
        )

    def embed(self, items: torch.Tensor, categories: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.item_embedding(items), self.category_embedding(categories)), dim=-1)

    def forward(
        self,
        history_items: torch.Tensor,
        history_categories: torch.Tensor,
        candidate_items: torch.Tensor,
        candidate_categories: torch.Tensor,
    ) -> torch.Tensor:
        single_candidate = candidate_items.ndim == 1
        if single_candidate:
            candidate_items = candidate_items.unsqueeze(1)
            candidate_categories = candidate_categories.unsqueeze(1)
        history = self.embed(history_items, history_categories)
        candidates = self.embed(candidate_items, candidate_categories)
        interest = self.attention(history, history_items.ne(0), candidates)
        logits = self.predictor(torch.cat((candidates, interest), dim=-1)).squeeze(-1)
        return logits.squeeze(1) if single_candidate else logits


if __name__ == "__main__":
    model = DIN(num_items=1_000, num_categories=100)
    logits = model(
        history_items=torch.tensor([[0, 0, 8, 21, 43]]),
        history_categories=torch.tensor([[0, 0, 2, 5, 7]]),
        candidate_items=torch.tensor([[101, 102]]),
        candidate_categories=torch.tensor([[7, 3]]),
    )
    print("logits:", logits)
