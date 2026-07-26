from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """
    可兼容旧版 PyTorch 的 RMSNorm。

    normalized_shape=(q, t) 时，会对匹配矩阵最后两个维度整体归一化；
    normalized_shape=d 时，会对每个知识库槽位的 embedding 维归一化。
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x: Tensor) -> Tensor:
        if tuple(x.shape[-len(self.normalized_shape):]) != self.normalized_shape:
            raise ValueError(
                f"RMSNorm 期望末尾 Shape={self.normalized_shape}，"
                f"实际输入 Shape={tuple(x.shape)}"
            )

        dims = tuple(range(x.ndim - len(self.normalized_shape), x.ndim))
        rms = x.square().mean(dim=dims, keepdim=True)
        x = x * torch.rsqrt(rms + self.eps)

        return x * self.weight


class SegmentWiseLinear(nn.Module):
    """
    论文中的 Segment-wise Linear Projection：

        Y = L(X) = ρX

    输入：
        X: [B, L_in, d]

    参数：
        ρ: [L_out, L_in]

    输出：
        Y: [B, L_out, d]

    注意：它变换的是 Slot/Segment 维，而不是最后的 embedding 维。
    """

    def __init__(
        self,
        in_slots: int,
        out_slots: int,
        bias: bool = False,
    ) -> None:
        super().__init__()

        self.in_slots = in_slots
        self.out_slots = out_slots

        self.weight = nn.Parameter(torch.empty(out_slots, in_slots))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_slots, 1))
        else:
            self.register_parameter("bias", None)

        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"SegmentWiseLinear 输入必须为 [B, L, d]，"
                f"实际为 {tuple(x.shape)}"
            )

        if x.size(1) != self.in_slots:
            raise ValueError(
                f"期望输入 Slot 数为 {self.in_slots}，"
                f"实际为 {x.size(1)}"
            )

        # [L_out, L_in] × [B, L_in, d] -> [B, L_out, d]
        output = torch.einsum("oi,bid->bod", self.weight, x)

        if self.bias is not None:
            output = output + self.bias

        return output


def masked_softmax(
    logits: Tensor,
    mask: Optional[Tensor],
    dim: int = -1,
    eps: float = 1e-12,
) -> Tensor:
    """
    logits: [B, q, N]
    mask:   [B, N]，True 表示有效 Token
    """

    if mask is None:
        return torch.softmax(logits, dim=dim)

    if mask.ndim != 2:
        raise ValueError(
            f"mask 应为 [B, N]，实际为 {tuple(mask.shape)}"
        )

    mask = mask[:, None, :].to(torch.bool)  # [B, 1, N]

    masked_logits = logits.masked_fill(
        ~mask,
        torch.finfo(logits.dtype).min,
    )

    weights = torch.softmax(masked_logits, dim=dim)

    # 处理某个样本全部 Token 都被 Mask 的极端情况
    weights = weights * mask.to(weights.dtype)
    weights = weights / weights.sum(
        dim=dim,
        keepdim=True,
    ).clamp_min(eps)

    return weights


class SparseMLPAttention(nn.Module):
    """
    固定长度稀疏 Token 的参数化 Attention：

        score_ij = MLP([q_i, s_j])
        R_s = Softmax(score) S

    论文只说明使用 MLP(Concat(...)) 作为参数化 Kernel，
    未披露具体网络结构。这里使用两层 Pairwise MLP。
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.score_mlp = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.attention_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: Tensor,
        sparse_tokens: Tensor,
        sparse_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        query:         [B, q, d]
        sparse_tokens: [B, N_s, d]
        sparse_mask:   [B, N_s]

        返回：
            selected: [B, q, d]
            weights:  [B, q, N_s]
        """

        batch_size, query_num, embedding_dim = query.shape
        sparse_num = sparse_tokens.size(1)

        # [B, q, 1, d] -> [B, q, N_s, d]
        query_pair = query.unsqueeze(2).expand(
            batch_size,
            query_num,
            sparse_num,
            embedding_dim,
        )

        # [B, 1, N_s, d] -> [B, q, N_s, d]
        sparse_pair = sparse_tokens.unsqueeze(1).expand(
            batch_size,
            query_num,
            sparse_num,
            embedding_dim,
        )

        pair_feature = torch.cat(
            [query_pair, sparse_pair],
            dim=-1,
        )  # [B, q, N_s, 2d]

        logits = self.score_mlp(pair_feature).squeeze(-1)
        # logits: [B, q, N_s]

        weights = masked_softmax(
            logits,
            sparse_mask,
            dim=-1,
        )
        weights = self.attention_dropout(weights)

        # [B, q, N_s] × [B, N_s, d] -> [B, q, d]
        selected = torch.einsum(
            "bqn,bnd->bqd",
            weights,
            sparse_tokens,
        )

        return selected, weights


class SlimPerBlock(nn.Module):
    """
    单层 SlimPer Block：

        Step 1：Select
        Step 2：Match
        Step 3：Refine

    默认论文配置：
        K = 64
        q = 16
        t = 32
        d = 256

    输入：
        x_k:           [B, K, d]
        sparse_tokens: [B, N_s, d]
        event_tokens:  [B, N_e, d]
        dense_features:[B, dense_dim]

    输出：
        x_next:        [B, K, d]
    """

    def __init__(
        self,
        embedding_dim: int = 256,
        knowledge_slots: int = 64,
        query_slots: int = 16,
        template_slots: int = 32,
        dense_dim: int = 1024,
        sparse_attention_hidden: int = 128,
        refine_hidden: int = 512,
        attention_dropout: float = 0.0,
        refine_dropout: float = 0.0,
        gamma: Optional[float] = None,
        post_norm: bool = False,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.knowledge_slots = knowledge_slots
        self.query_slots = query_slots
        self.template_slots = template_slots
        self.dense_dim = dense_dim

        # 默认对应标准 scaled dot-product 的 1 / sqrt(d)
        self.gamma = (
            gamma
            if gamma is not None
            else 1.0 / math.sqrt(embedding_dim)
        )

        # =========================
        # Step 1: Select
        # =========================

        # Q = L_Q(X^k): [B, K, d] -> [B, q, d]
        self.query_projection = SegmentWiseLinear(
            in_slots=knowledge_slots,
            out_slots=query_slots,
        )

        self.sparse_attention = SparseMLPAttention(
            embedding_dim=embedding_dim,
            hidden_dim=sparse_attention_hidden,
            dropout=attention_dropout,
        )

        self.sequence_attention_dropout = nn.Dropout(
            attention_dropout
        )

        # =========================
        # Step 2: Match
        # =========================

        # T = L_T(X^k): [B, K, d] -> [B, t, d]
        self.template_projection = SegmentWiseLinear(
            in_slots=knowledge_slots,
            out_slots=template_slots,
        )

        # =========================
        # Step 3: Refine
        # =========================

        self.lambda_s_norm = RMSNorm(
            (query_slots, template_slots)
        )
        self.lambda_e_norm = RMSNorm(
            (query_slots, template_slots)
        )

        # Equation (10) 中的 L(X^k)
        self.refine_kb_projection = SegmentWiseLinear(
            in_slots=knowledge_slots,
            out_slots=knowledge_slots,
        )

        # Concat 后展平：
        # λ_s: q*t
        # λ_e: q*t
        # L(X^k): K*d
        # D: dense_dim
        refine_input_dim = (
            2 * query_slots * template_slots
            + knowledge_slots * embedding_dim
            + dense_dim
        )

        self.refine_mlp = nn.Sequential(
            nn.Linear(refine_input_dim, refine_hidden),
            nn.SiLU(),
            nn.Dropout(refine_dropout),
            nn.Linear(
                refine_hidden,
                knowledge_slots * embedding_dim,
            ),
        )

        # 论文公式 (10) 只明确写了残差相加；
        # Figure 中出现 Add & Norm，因此提供可选 Post RMSNorm
        self.post_norm = (
            RMSNorm(embedding_dim)
            if post_norm
            else nn.Identity()
        )

    def _validate_inputs(
        self,
        x_k: Tensor,
        sparse_tokens: Tensor,
        event_tokens: Tensor,
        dense_features: Tensor,
    ) -> None:
        batch_size = x_k.size(0)

        expected_kb_shape = (
            batch_size,
            self.knowledge_slots,
            self.embedding_dim,
        )

        if tuple(x_k.shape) != expected_kb_shape:
            raise ValueError(
                f"x_k 期望 Shape={expected_kb_shape}，"
                f"实际为 {tuple(x_k.shape)}"
            )

        for name, tokens in (
            ("sparse_tokens", sparse_tokens),
            ("event_tokens", event_tokens),
        ):
            if tokens.ndim != 3:
                raise ValueError(
                    f"{name} 应为 [B, N, d]，"
                    f"实际为 {tuple(tokens.shape)}"
                )

            if tokens.size(0) != batch_size:
                raise ValueError(f"{name} 的 Batch Size 不一致")

            if tokens.size(-1) != self.embedding_dim:
                raise ValueError(
                    f"{name} 的 embedding_dim 应为 "
                    f"{self.embedding_dim}"
                )

        dense_features = dense_features.reshape(batch_size, -1)

        if dense_features.size(1) != self.dense_dim:
            raise ValueError(
                f"dense_features 展平后应为 "
                f"[B, {self.dense_dim}]，"
                f"实际为 {tuple(dense_features.shape)}"
            )

    def forward(
        self,
        x_k: Tensor,
        sparse_tokens: Tensor,
        event_tokens: Tensor,
        dense_features: Tensor,
        sparse_mask: Optional[Tensor] = None,
        event_mask: Optional[Tensor] = None,
        return_aux: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
        """
        sparse_mask:
            [B, N_s]，True 表示有效稀疏 Token

        event_mask:
            [B, N_e]，True 表示有效历史事件
        """

        self._validate_inputs(
            x_k,
            sparse_tokens,
            event_tokens,
            dense_features,
        )

        batch_size = x_k.size(0)
        dense_features = dense_features.reshape(batch_size, -1)

        # ==================================================
        # Step 1: Select
        # ==================================================

        # Q = L_Q(X^k)
        # [B, K, d] -> [B, q, d]
        query = self.query_projection(x_k)

        # Sequence Attention:
        # A_e = Softmax(gamma * Q E^T)
        # [B, q, d] × [B, d, N_e] -> [B, q, N_e]
        event_logits = torch.einsum(
            "bqd,bnd->bqn",
            query,
            event_tokens,
        )
        event_logits = event_logits * self.gamma

        event_attention = masked_softmax(
            event_logits,
            event_mask,
            dim=-1,
        )
        event_attention = self.sequence_attention_dropout(
            event_attention
        )

        # R_e = A_e E
        # [B, q, N_e] × [B, N_e, d] -> [B, q, d]
        selected_event = torch.einsum(
            "bqn,bnd->bqd",
            event_attention,
            event_tokens,
        )

        # Sparse MLP Attention
        # R_s = Phi_s(Q, S) S
        selected_sparse, sparse_attention = self.sparse_attention(
            query=query,
            sparse_tokens=sparse_tokens,
            sparse_mask=sparse_mask,
        )

        # ==================================================
        # Step 2: Match
        # ==================================================

        # T = L_T(X^k)
        # [B, K, d] -> [B, t, d]
        templates = self.template_projection(x_k)

        # lambda_e = R_e T^T
        # [B, q, d] × [B, d, t] -> [B, q, t]
        lambda_event = torch.einsum(
            "bqd,btd->bqt",
            selected_event,
            templates,
        )

        # lambda_s = R_s T^T
        lambda_sparse = torch.einsum(
            "bqd,btd->bqt",
            selected_sparse,
            templates,
        )

        # ==================================================
        # Step 3: Refine
        # ==================================================

        lambda_sparse_norm = self.lambda_s_norm(
            lambda_sparse
        )
        lambda_event_norm = self.lambda_e_norm(
            lambda_event
        )

        projected_kb = self.refine_kb_projection(x_k)

        refine_input = torch.cat(
            [
                lambda_sparse_norm.flatten(start_dim=1),
                lambda_event_norm.flatten(start_dim=1),
                projected_kb.flatten(start_dim=1),
                dense_features,
            ],
            dim=-1,
        )

        # ΔX^k = MLP_mu(...)
        delta_x = self.refine_mlp(refine_input)
        delta_x = delta_x.reshape(
            batch_size,
            self.knowledge_slots,
            self.embedding_dim,
        )

        # X^{k+1} = X^k + ΔX^k
        x_next = x_k + delta_x
        x_next = self.post_norm(x_next)

        if not return_aux:
            return x_next

        auxiliary = {
            "query": query,
            "templates": templates,
            "event_attention": event_attention,
            "sparse_attention": sparse_attention,
            "selected_event": selected_event,
            "selected_sparse": selected_sparse,
            "lambda_event": lambda_event,
            "lambda_sparse": lambda_sparse,
            "delta_x": delta_x,
        }

        return x_next, auxiliary


if __name__ == "__main__":
    torch.manual_seed(42)

    # 论文典型 Shape
    batch_size = 2
    knowledge_slots = 64
    query_slots = 16
    template_slots = 32
    embedding_dim = 256

    sparse_num = 32
    event_num = 5000
    dense_dim = 1024

    block = SlimPerBlock(
        embedding_dim=embedding_dim,
        knowledge_slots=knowledge_slots,
        query_slots=query_slots,
        template_slots=template_slots,
        dense_dim=dense_dim,
        sparse_attention_hidden=128,
        refine_hidden=512,
        attention_dropout=0.1,
        refine_dropout=0.1,
    )

    x_k = torch.randn(
        batch_size,
        knowledge_slots,
        embedding_dim,
        requires_grad=True,
    )

    sparse_tokens = torch.randn(
        batch_size,
        sparse_num,
        embedding_dim,
    )

    event_tokens = torch.randn(
        batch_size,
        event_num,
        embedding_dim,
    )

    dense_features = torch.randn(
        batch_size,
        dense_dim,
    )

    # True 表示有效 Token
    sparse_mask = torch.ones(
        batch_size,
        sparse_num,
        dtype=torch.bool,
    )

    event_mask = torch.ones(
        batch_size,
        event_num,
        dtype=torch.bool,
    )

    # 示例：第二个用户最后 500 条历史是 Padding
    event_mask[1, -500:] = False

    x_next, auxiliary = block(
        x_k=x_k,
        sparse_tokens=sparse_tokens,
        event_tokens=event_tokens,
        dense_features=dense_features,
        sparse_mask=sparse_mask,
        event_mask=event_mask,
        return_aux=True,
    )

    print("X^k:              ", tuple(x_k.shape))
    print("Q:                ", tuple(auxiliary["query"].shape))
    print("A_e:              ", tuple(auxiliary["event_attention"].shape))
    print("R_e:              ", tuple(auxiliary["selected_event"].shape))
    print("A_s:              ", tuple(auxiliary["sparse_attention"].shape))
    print("R_s:              ", tuple(auxiliary["selected_sparse"].shape))
    print("T:                ", tuple(auxiliary["templates"].shape))
    print("lambda_e:         ", tuple(auxiliary["lambda_event"].shape))
    print("lambda_s:         ", tuple(auxiliary["lambda_sparse"].shape))
    print("delta_X:          ", tuple(auxiliary["delta_x"].shape))
    print("X^{k+1}:          ", tuple(x_next.shape))

    # 验证梯度能够正常反向传播
    loss = x_next.square().mean()
    loss.backward()

    print("Backward success:", x_k.grad is not None)