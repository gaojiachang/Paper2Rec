"""SASRec 的因果自注意力、残差块与候选打分模型。"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """只允许当前位置关注自身及更早行为的多头自注意力。"""

    def __init__(self, hidden_size: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape

        def shape(proj: torch.Tensor) -> torch.Tensor:
            """把投影结果整理为 ``[批量, 头数, 序列长度, 头维度]``。"""
            return proj.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        q = shape(self.q_proj(x))
        k = shape(self.k_proj(x))
        v = shape(self.v_proj(x))

        # 缩放点积得到每个头的两两位置分数，形状为 [B, H, L, L]。
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 上三角因果掩码阻止当前位置看到未来，valid_mask 再排除左侧 PAD。
        causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(causal_mask, -1.0e9)
        scores = scores.masked_fill(~valid_mask[:, None, None, :], -1.0e9)

        weights = torch.softmax(scores, dim=-1)
        weights = self.dropout(weights)
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        return self.out_proj(context)


class SASRecBlock(nn.Module):
    """由预归一化自注意力和逐位置前馈网络组成的 SASRec 残差块。"""

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
        # 两个子层都使用残差连接，并在子层后重新清零 PAD 位置。
        x = x + self.dropout1(self.attention(self.norm1(x), valid_mask))
        x = x * valid_mask.unsqueeze(-1)
        x = x + self.dropout2(self.ffn(self.norm2(x)))
        x = x * valid_mask.unsqueeze(-1)
        return x


class SASRec(nn.Module):
    """编码用户行为序列，并通过向量内积预测下一个商品。"""

    def __init__(
        self,
        num_items: int,
        max_seq_len: int,
        hidden_size: int,
        num_blocks: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.item_embedding = nn.Embedding(num_items + 1, hidden_size, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([SASRecBlock(hidden_size, num_heads, dropout) for _ in range(num_blocks)])
        self.final_norm = nn.LayerNorm(hidden_size, eps=1.0e-8)
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len

    def encode(self, item_sequences: torch.Tensor) -> torch.Tensor:
        """将左侧补零的商品序列编码为逐位置隐藏状态。"""
        valid_mask = item_sequences != 0
        positions = torch.arange(item_sequences.size(1), device=item_sequences.device).unsqueeze(0)

        # 商品嵌入按隐藏维度缩放后与位置嵌入相加，PAD 全程保持为零。
        x = self.item_embedding(item_sequences) * math.sqrt(self.hidden_size)
        x = x + self.position_embedding(positions)
        x = self.dropout(x)
        x = x * valid_mask.unsqueeze(-1)

        for block in self.blocks:
            x = block(x, valid_mask)

        x = self.final_norm(x)
        x = x * valid_mask.unsqueeze(-1)
        return x

    def forward_logits(
        self,
        sequences: torch.Tensor,
        positive_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """计算每个有效训练位置对应的正样本与负样本 logits。"""
        hidden = self.encode(sequences)
        positive_emb = self.item_embedding(positive_ids)
        negative_emb = self.item_embedding(negative_ids)
        positive_logits = (hidden * positive_emb).sum(dim=-1)
        negative_logits = (hidden * negative_emb).sum(dim=-1)
        return positive_logits, negative_logits

    def score_candidates(self, sequences: torch.Tensor, candidate_ids: torch.Tensor) -> torch.Tensor:
        """使用最后一个位置的兴趣表示为一组候选商品打分。"""
        hidden = self.encode(sequences)
        # 离线历史采用左侧补零，因此最后一个位置始终对应最近一次有效行为。
        last_hidden = hidden[:, -1, :]
        candidate_emb = self.item_embedding(candidate_ids)
        return torch.einsum("bd,bkd->bk", last_hidden, candidate_emb)
