"""SASRec（Self-Attentive Sequential Recommendation）的独立 PyTorch 实现。"""

from __future__ import annotations

import math

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    """仅关注当前位置及其之前行为的多头自注意力。"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size 必须可被 num_heads 整除。")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, hidden_size = x.shape

        def reshape(projected: torch.Tensor) -> torch.Tensor:
            return projected.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        query, key, value = map(reshape, (self.q_proj(x), self.k_proj(x), self.v_proj(x)))
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_dim)
        causal_mask = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(causal_mask, -1.0e9)
        scores = scores.masked_fill(~valid_mask[:, None, None], -1.0e9)
        context = self.dropout(torch.softmax(scores, dim=-1)) @ value
        context = context.transpose(1, 2).contiguous().view(batch_size, sequence_length, hidden_size)
        return self.out_proj(context)


class SASRecBlock(nn.Module):
    """SASRec 的预归一化注意力与逐位置前馈残差块。"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, eps=1.0e-8)
        self.attention = CausalSelfAttention(hidden_size, num_heads, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_size, eps=1.0e-8)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout1(self.attention(self.norm1(x), valid_mask))
        x = x * valid_mask.unsqueeze(-1)
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        return x * valid_mask.unsqueeze(-1)


class SASRec(nn.Module):
    """编码左侧补零的行为序列，并为候选商品输出打分。"""

    def __init__(
        self,
        num_items: int,
        max_seq_len: int = 200,
        hidden_size: int = 64,
        num_blocks: int = 2,
        num_heads: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            SASRecBlock(hidden_size, num_heads, dropout) for _ in range(num_blocks)
        )
        self.final_norm = nn.LayerNorm(hidden_size, eps=1.0e-8)
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

    def encode(self, item_sequences: torch.Tensor) -> torch.Tensor:
        if item_sequences.size(1) > self.max_seq_len:
            raise ValueError("序列长度不能超过 max_seq_len。")
        valid_mask = item_sequences.ne(0)
        positions = torch.arange(item_sequences.size(1), device=item_sequences.device).unsqueeze(0)
        x = self.item_embedding(item_sequences) * math.sqrt(self.hidden_size)
        x = self.dropout(x + self.position_embedding(positions)) * valid_mask.unsqueeze(-1)
        for block in self.blocks:
            x = block(x, valid_mask)
        return self.final_norm(x) * valid_mask.unsqueeze(-1)

    def forward_logits(
        self,
        sequences: torch.Tensor,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回所有训练位置的正、负样本 logits。"""
        hidden = self.encode(sequences)
        return (
            (hidden * self.item_embedding(positive_ids)).sum(-1),
            (hidden * self.item_embedding(negative_ids)).sum(-1),
        )

    def score_candidates(self, sequences: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        """以每条序列最后一个位置的兴趣向量对 ``[B, C]`` 候选打分。"""
        last_hidden = self.encode(sequences)[:, -1]
        return torch.einsum("bd,bcd->bc", last_hidden, self.item_embedding(candidate_ids))


if __name__ == "__main__":
    model = SASRec(num_items=1_000, max_seq_len=5)
    scores = model.score_candidates(
        sequences=torch.tensor([[0, 0, 8, 21, 43]]),
        candidate_ids=torch.tensor([[101, 102, 103]]),
    )
    print("scores:", scores)
