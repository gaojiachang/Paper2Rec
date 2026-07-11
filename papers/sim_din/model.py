"""候选感知的 DIN、SIM 长期兴趣及短长期融合模型。"""

from __future__ import annotations

import math

import torch
from torch import nn

try:
    from .config import RunConfig
except ImportError:  # pragma: no cover - 支持直接运行脚本
    from config import RunConfig


class Dice(nn.Module):
    """经典 DIN 局部激活单元使用的数据自适应激活函数。"""

    def __init__(self, features: int, epsilon: float = 1e-8) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(features, eps=epsilon)
        self.alpha = nn.Parameter(torch.zeros(features))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        original_shape = values.shape
        # BatchNorm1d 只处理二维特征，先合并前置维度，归一化后再恢复形状。
        flattened = values.reshape(-1, original_shape[-1])
        probability = torch.sigmoid(self.norm(flattened)).reshape(original_shape)
        return probability * values + (1.0 - probability) * self.alpha * values


class DinTargetAttention(nn.Module):
    """根据候选商品为每条短期行为计算独立的 DIN 注意力权重。"""

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
        """为每个候选返回对应的短期兴趣向量，输出形状为 ``[B, C, D]``。"""
        batch_size, sequence_length, behavior_dim = behavior_embeddings.shape
        candidate_count = candidate_embeddings.size(1)
        # 将历史与候选广播为 [B, C, L, D]，逐候选、逐行为计算匹配分数。
        behaviors = behavior_embeddings[:, None, :, :].expand(
            batch_size, candidate_count, sequence_length, behavior_dim
        )
        candidates = candidate_embeddings[:, :, None, :].expand_as(behaviors)
        # 拼接行为、候选、差值和逐元素乘积，形成 DIN 的局部激活特征。
        features = torch.cat((behaviors, candidates, behaviors - candidates, behaviors * candidates), dim=-1)
        scores = self.score(features).squeeze(-1)
        mask = behavior_mask[:, None, :]
        # 使用有限大负数兼容混合精度；全为填充的序列会在 softmax 后显式清零。
        scores = scores.masked_fill(~mask, -1.0e4)
        weights = torch.softmax(scores, dim=-1)
        has_behavior = mask.any(dim=-1)
        weights = torch.where(has_behavior[:, :, None], weights, torch.zeros_like(weights))
        # 沿历史长度维加权求和，得到候选相关的兴趣表示。
        return torch.einsum("bcl,bcld->bcd", weights, behaviors)


def hard_search_last_k(
    long_items: torch.Tensor,
    long_categories: torch.Tensor,
    long_timestamps: torch.Tensor,
    candidate_categories: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """按候选类别向量化检索最近 K 条行为，并按时间正序返回。

    ``long_*`` 在组内共享，形状为 ``[B, L]``；候选类别形状为 ``[B, C]``。
    返回的定长序列采用左侧填充，PAD 不会被视为有效匹配。
    """
    batch_size, sequence_length = long_items.shape
    if k > sequence_length:
        raise ValueError("Hard Search 的 K 不能超过长期序列长度。")
    candidates = candidate_categories.size(1)
    # 为每个候选构造类别匹配矩阵 [B, C, L]，同时排除 PAD。
    valid_matches = (long_items[:, None, :] != 0) & (
        long_categories[:, None, :] == candidate_categories[:, :, None]
    )
    position_ids = torch.arange(sequence_length, device=long_items.device).view(1, 1, sequence_length)
    position_scores = torch.where(valid_matches, position_ids, torch.full_like(position_ids, -1))
    top_positions, top_indices = torch.topk(position_scores, k=k, dim=-1, largest=True, sorted=True)
    selected_valid = top_positions >= 0
    # topk 默认由新到旧；翻转后恢复时间正序，无效位置自然位于左侧。
    top_indices = top_indices.flip(dims=(-1,))
    selected_valid = selected_valid.flip(dims=(-1,))
    safe_indices = top_indices.clamp_min(0)

    def gather(sequence: torch.Tensor) -> torch.Tensor:
        """按统一索引抽取属性，并将无效检索位置还原为 PAD。"""
        expanded = sequence[:, None, :].expand(batch_size, candidates, sequence_length)
        values = torch.gather(expanded, dim=2, index=safe_indices)
        return torch.where(selected_valid, values, torch.zeros_like(values))

    return gather(long_items), gather(long_categories), gather(long_timestamps), selected_valid


class MultiHeadTargetAttention(nn.Module):
    """以候选为查询，对 Hard Search 结果执行多头目标注意力。"""

    def __init__(self, query_dim: int, value_dim: int, heads: int, dropout: float) -> None:
        super().__init__()
        if value_dim % heads != 0:
            raise ValueError("长期行为维度必须能被注意力头数整除。")
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
        # Query 来自候选，Key 和 Value 来自为该候选检索出的长期行为。
        query = self.query(candidate_embeddings).view(
            batch_size, candidate_count, self.heads, self.head_dim
        )
        key = self.key(long_embeddings).view(
            batch_size, candidate_count, sequence_length, self.heads, self.head_dim
        )
        value = self.value(long_embeddings).view(
            batch_size, candidate_count, sequence_length, self.heads, self.head_dim
        )
        # 在每个注意力头内计算缩放点积，并屏蔽无效检索位置。
        scores = torch.einsum("bchd,bckhd->bchk", query, key) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~long_mask[:, :, None, :], -1.0e4)
        weights = torch.softmax(scores, dim=-1)
        has_long_interest = long_mask.any(dim=-1)
        # 某候选没有类别匹配行为时，强制其注意力与最终上下文均为零。
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
    """将候选表示与兴趣表示映射为单个点击 logit。"""

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
    """统一模型，通过 ``model`` 配置选择 DIN、SIM 长期分支或短长期融合。"""

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
        # 按实验变体只创建实际使用的兴趣分支，避免引入无效参数。
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

        # 预测层始终包含候选表示，再按模型类型拼接短期或长期兴趣。
        input_dim = {
            "din": self.behavior_dim * 2,
            "sim": self.behavior_dim + self.long_behavior_dim,
            "ours": self.behavior_dim * 2 + self.long_behavior_dim,
        }[self.model_name]
        self.predictor = PredictionMLP(input_dim, config.dropout)

    def _behavior_embedding(self, items: torch.Tensor, categories: torch.Tensor) -> torch.Tensor:
        """拼接商品与类别 embedding，形成统一的行为表示。"""
        return torch.cat((self.item_embedding(items), self.category_embedding(categories)), dim=-1)

    def _time_bucket_ids(
        self,
        selected_timestamps: torch.Tensor,
        selected_mask: torch.Tensor,
        target_timestamps: torch.Tensor,
    ) -> torch.Tensor:
        """将目标与历史的时间差映射到对数时间桶，0 保留给 PAD。"""
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
        """使用 DIN 目标注意力提取候选相关的短期兴趣。"""
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
        """先执行类别 Hard Search，再用多头注意力提取长期兴趣。"""
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
        """按所选模型组合候选、短期兴趣和长期兴趣并输出 logits。"""
        candidate_items = batch["candidate_items"]
        candidate_categories = batch["candidate_categories"]
        candidate_embeddings = self._behavior_embedding(candidate_items, candidate_categories)
        # 三种变体都直接保留候选本身的 embedding 作为预测特征。
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
