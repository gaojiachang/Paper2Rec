"""
hyformer_demo.py

一个最小可运行的 HyFormer 风格示例：
1. 非序列特征 + 多序列池化 -> 初始 Global Tokens
2. 每层执行：
   Query Decoding：Global Token 读取对应序列 K/V
   Query Boosting：Global Tokens 与 NS Tokens 做轻量 Token Mixing
3. 顶层输出 -> CTR Prediction
4. 单个 case 完成 forward + BCE loss + backward

依赖：
    pip install torch
"""

import torch
from torch import nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    """论文中的轻量序列编码方案：用前馈变换代替序列内 Self-Attention。"""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.proj(x).chunk(2, dim=-1)
        return self.out(value * F.silu(gate))


class TokenMixer(nn.Module):
    """
    RankMixer Block 的简化实现：

    1. Parameter-free Token Mixing
    2. 每个 Token 使用独立参数的 PFFN
    3. 两次 Residual + LayerNorm

    输入/输出:
        x: [B, T, D]

    要求:
        H = T
        D % T == 0
    """

    def __init__(
        self,
        num_tokens: int,
        d_model: int,
        expansion: int = 2,
    ):
        super().__init__()

        assert d_model % num_tokens == 0

        self.num_tokens = num_tokens
        self.chunk_dim = d_model // num_tokens

        self.token_mix_norm = nn.LayerNorm(d_model)
        self.pffn_norm = nn.LayerNorm(d_model)

        # 每个 Token 拥有一套独立 FFN 参数
        self.pffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expansion * d_model),
                nn.GELU(),
                nn.Linear(expansion * d_model, d_model),
            )
            for _ in range(num_tokens)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, D]
        """

        batch_size, num_tokens, d_model = x.shape

        assert num_tokens == self.num_tokens
        assert d_model == self.chunk_dim * self.num_tokens

        # --------------------------------------------------
        # 1. Multi-Head Token Mixing
        # --------------------------------------------------
        # [B, T, D]
        # -> [B, T_token, T_head, D/T]
        mixed = x.reshape(
            batch_size,
            num_tokens,
            num_tokens,
            self.chunk_dim,
        )

        # 将所有 Token 的同一个 Head 聚合到一起
        # [B, T_token, T_head, D/T]
        # -> [B, T_head, T_token, D/T]
        mixed = mixed.transpose(1, 2).contiguous()

        # 每个新 Token 包含所有原始 Token 的一个子空间
        # [B, T, T, D/T] -> [B, T, D]
        mixed = mixed.reshape(
            batch_size,
            num_tokens,
            d_model,
        )

        # Token Mixing 后的残差与归一化
        s = self.token_mix_norm(x + mixed)

        # --------------------------------------------------
        # 2. Parameter-Isolated Per-Token FFN
        # --------------------------------------------------
        pffn_outputs = []

        for token_idx, pffn in enumerate(self.pffns):
            # 每个 Token 使用自己的 FFN
            token = s[:, token_idx, :]       # [B, D]
            token_output = pffn(token)        # [B, D]
            pffn_outputs.append(token_output)

        # [T 个 [B, D]] -> [B, T, D]
        pffn_output = torch.stack(
            pffn_outputs,
            dim=1,
        )

        # PFFN 后的残差与归一化
        output = self.pffn_norm(s + pffn_output)

        return output


class HyFormerBlock(nn.Module):
    """
    一个 HyFormer Block：
        Query Decoding -> Query Boosting
    """

    def __init__(
        self,
        num_sequences: int,
        num_ns_tokens: int,
        d_model: int,
        num_heads: int,
    ):
        super().__init__()

        self.num_sequences = num_sequences
        self.num_ns_tokens = num_ns_tokens
        self.num_total_tokens = num_sequences + num_ns_tokens

        # 每条序列独立编码，保留多序列语义差异
        self.sequence_encoders = nn.ModuleList(
            [SwiGLU(d_model) for _ in range(num_sequences)]
        )

        # 每条序列都有独立的 K/V 投影和 Cross-Attention
        self.key_projections = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_sequences)]
        )
        self.value_projections = nn.ModuleList(
            [nn.Linear(d_model, d_model) for _ in range(num_sequences)]
        )
        self.cross_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=num_heads,
                    batch_first=True,
                )
                for _ in range(num_sequences)
            ]
        )

        self.query_norm = nn.LayerNorm(d_model)

        self.query_boosting = TokenMixer(
            num_tokens=self.num_total_tokens,
            d_model=d_model,
        )

    def forward(
        self,
        global_tokens: torch.Tensor,
        ns_tokens: torch.Tensor,
        sequences: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        global_tokens: [B, S, D]，S 为序列数量
        ns_tokens:     [B, M, D]
        sequences[j]:  [B, L_j, D]
        """

        decoded_tokens = []

        for j in range(self.num_sequences):
            # A. Sequence KV Encoding
            hidden = self.sequence_encoders[j](sequences[j])
            key = self.key_projections[j](hidden)
            value = self.value_projections[j](hidden)

            # B. Query Decoding
            # 每个 Global Token 只读取对应序列
            query = global_tokens[:, j : j + 1, :]
            decoded, _ = self.cross_attentions[j](
                query=query,
                key=key,
                value=value,
                need_weights=False,
            )

            # Cross-Attention 输出 + 原 Query 残差
            decoded = self.query_norm(decoded + query)
            decoded_tokens.append(decoded)

        decoded_global = torch.cat(decoded_tokens, dim=1)  # [B, S, D]

        # C. Query Boosting
        all_tokens = torch.cat([decoded_global, ns_tokens], dim=1)
        boosted = self.query_boosting(all_tokens)

        new_global = boosted[:, : self.num_sequences, :]
        new_ns = boosted[:, self.num_sequences :, :]

        return new_global, new_ns


class TinyHyFormer(nn.Module):
    def __init__(
        self,
        num_sequences: int = 3,
        num_ns_tokens: int = 5,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()

        self.num_sequences = num_sequences
        self.num_ns_tokens = num_ns_tokens
        self.d_model = d_model

        # M 个 NS Token 全部保留
        # 每条序列池化为 1 个向量
        global_info_dim = (
            num_ns_tokens + num_sequences
        ) * d_model

        self.query_generators = nn.ModuleList([
            nn.Sequential(
                nn.Linear(global_info_dim, d_model),
                nn.GELU(),
                nn.Linear(d_model, d_model),
            )
            for _ in range(num_sequences)
        ])

        self.blocks = nn.ModuleList([
            HyFormerBlock(
                num_sequences=num_sequences,
                num_ns_tokens=num_ns_tokens,
                d_model=d_model,
                num_heads=num_heads,
            )
            for _ in range(num_layers)
        ])

        self.prediction_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def generate_queries(
        self,
        ns_tokens: torch.Tensor,
        sequences: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        ns_tokens:
            [B, M, D]

        sequences[j]:
            [B, L_j, D]
        """

        batch_size = ns_tokens.size(0)

        # NS Token 不池化，直接展平拼接
        # [B, M, D] -> [B, M*D]
        ns_info = ns_tokens.reshape(batch_size, -1)

        # 每条序列进行 Mean Pooling
        # 每个结果：[B, D]
        seq_summaries = [
            sequence.mean(dim=1)
            for sequence in sequences
        ]

        # [B, M*D] + K 个 [B, D]
        # -> [B, (M+K)*D]
        global_info = torch.cat(
            [ns_info, *seq_summaries],
            dim=-1,
        )

        # 不同 FFN 生成不同序列对应的 Global Token
        queries = [
            generator(global_info).unsqueeze(1)
            for generator in self.query_generators
        ]

        # [B, K, D]
        return torch.cat(queries, dim=1)

    def forward(
        self,
        ns_tokens: torch.Tensor,
        sequences: list[torch.Tensor],
    ) -> torch.Tensor:
        global_tokens = self.generate_queries(ns_tokens, sequences)

        print("初始 Global Tokens:", tuple(global_tokens.shape))

        for layer_idx, block in enumerate(self.blocks, start=1):
            global_tokens, ns_tokens = block(
                global_tokens=global_tokens,
                ns_tokens=ns_tokens,
                sequences=sequences,
            )

            print(
                f"Block {layer_idx}:",
                "Global",
                tuple(global_tokens.shape),
                "NS",
                tuple(ns_tokens.shape),
            )

        top_tokens = torch.cat([global_tokens, ns_tokens], dim=1)
        pooled = top_tokens.mean(dim=1)
        logit = self.prediction_head(pooled)

        return logit


def main() -> None:
    torch.manual_seed(42)

    # 单个 case
    batch_size = 1
    d_model = 32

    # 为了保证 D % T == 0：
    # 3 个 Global Tokens + 5 个 NS Tokens = 8 个 Token
    num_sequences = 3
    num_ns_tokens = 5

    # 非序列特征：
    # User / Current Query / Candidate / Context / Cross Feature
    ns_tokens = torch.randn(
        batch_size,
        num_ns_tokens,
        d_model,
    )

    # 为了让 Demo 快速运行，这里缩短序列长度。
    # 真实论文场景可对应 3000 / 50 / 50。
    sequences = [
        torch.randn(batch_size, 30, d_model),  # Long-term
        torch.randn(batch_size, 10, d_model),  # Search
        torch.randn(batch_size, 10, d_model),  # Feed
    ]

    # 正样本：用户点击候选内容
    label = torch.tensor([[1.0]])

    model = TinyHyFormer(
        num_sequences=num_sequences,
        num_ns_tokens=num_ns_tokens,
        d_model=d_model,
        num_heads=4,
        num_layers=2,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Forward
    logit = model(ns_tokens, sequences)
    probability = torch.sigmoid(logit)

    # BCEWithLogitsLoss 内部自动执行 Sigmoid + BCE
    loss = F.binary_cross_entropy_with_logits(logit, label)

    print("\nLogit:", logit.detach().item())
    print("CTR Probability:", probability.detach().item())
    print("BCE Loss:", loss.detach().item())

    # Backward
    optimizer.zero_grad()
    loss.backward()

    grad_norm = model.prediction_head[-1].weight.grad.norm().item()
    print("Prediction Head Gradient Norm:", grad_norm)

    optimizer.step()
    print("Forward + Backward + Optimizer Step 完成")


if __name__ == "__main__":
    main()