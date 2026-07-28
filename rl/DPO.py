from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyCausalLM(nn.Module):
    """极小的 Decoder-only Transformer 语言模型。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 32,
    ) -> None:
        super().__init__()

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=2 * d_model,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T]
        return logits: [B, T, V]
        """
        batch_size, seq_len = input_ids.shape

        positions = torch.arange(
            seq_len,
            device=input_ids.device,
        ).unsqueeze(0)  # [1, T]

        hidden = (
            self.token_embedding(input_ids)
            + self.position_embedding(positions)
        )  # [B, T, D]

        # True 表示该位置不可见，实现自回归 Causal Attention。
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                dtype=torch.bool,
                device=input_ids.device,
            ),
            diagonal=1,
        )  # [T, T]

        hidden = self.transformer(
            hidden,
            mask=causal_mask,
        )  # [B, T, D]

        logits = self.lm_head(hidden)  # [B, T, V]
        return logits


def response_log_prob(
    model: nn.Module,
    input_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    计算 P(response | prompt) 的序列对数概率。

    input_ids = [prompt tokens, response tokens]
    只累计 response 部分，不累计 prompt 部分。

    return: [B]
    """
    logits = model(input_ids)  # [B, T, V]

    # 第 t 个位置预测第 t+1 个 Token。
    next_token_logits = logits[:, :-1, :]  # [B, T-1, V]
    next_token_ids = input_ids[:, 1:]      # [B, T-1]

    token_log_probs = F.log_softmax(
        next_token_logits,
        dim=-1,
    )  # [B, T-1, V]

    target_log_probs = token_log_probs.gather(
        dim=-1,
        index=next_token_ids.unsqueeze(-1),
    ).squeeze(-1)  # [B, T-1]

    # target_log_probs[:, j] 对应原序列中位置 j+1 的 Token。
    target_positions = torch.arange(
        1,
        input_ids.size(1),
        device=input_ids.device,
    )
    response_mask = target_positions >= prompt_len  # [T-1]

    sequence_log_prob = (
        target_log_probs
        * response_mask.unsqueeze(0)
    ).sum(dim=-1)  # [B]

    return sequence_log_prob


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    reference_chosen_logp: torch.Tensor,
    reference_rejected_logp: torch.Tensor,
    beta: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    DPO 核心公式：

    advantage =
        [log πθ(y_w|x) - log πθ(y_l|x)]
        - [log πref(y_w|x) - log πref(y_l|x)]

    loss = -log σ(beta * advantage)
    """
    policy_log_ratio = (
        policy_chosen_logp
        - policy_rejected_logp
    )  # [B]

    reference_log_ratio = (
        reference_chosen_logp
        - reference_rejected_logp
    )  # [B]

    advantage = policy_log_ratio - reference_log_ratio  # [B]

    loss = -F.logsigmoid(beta * advantage).mean()

    # 仅用于观察：模型认为 chosen 优于 rejected 的相对概率。
    preference_probability = torch.sigmoid(beta * advantage)

    return loss, preference_probability


def main() -> None:
    torch.manual_seed(42)

    # ------------------------------------------------------------
    # 1. 构造一个极简偏好样本
    # ------------------------------------------------------------
    # Token 语义仅用于帮助理解：
    # 0=<pad>, 1=<bos>, 2=用户, 3=提问,
    # 4=认真, 5=回答, 6=<eos>,
    # 7=敷衍, 8=拒绝
    vocab_size = 9
    prompt_len = 3

    chosen_ids = torch.tensor(
        [[1, 2, 3, 4, 5, 6]],
        dtype=torch.long,
    )  # prompt + 优质回答

    rejected_ids = torch.tensor(
        [[1, 2, 3, 7, 8, 6]],
        dtype=torch.long,
    )  # 同一 prompt + 较差回答

    print("chosen_ids shape :", tuple(chosen_ids.shape))
    print("rejected_ids shape:", tuple(rejected_ids.shape))

    # ------------------------------------------------------------
    # 2. 创建策略模型与冻结参考模型
    # ------------------------------------------------------------
    policy = TinyCausalLM(vocab_size=vocab_size)

    # reference 是训练开始时 policy 的冻结副本。
    reference = copy.deepcopy(policy)
    reference.eval()

    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=3e-3,
    )

    # ------------------------------------------------------------
    # 3. Forward：计算四个序列对数概率
    # ------------------------------------------------------------
    policy.train()

    policy_chosen_logp = response_log_prob(
        policy,
        chosen_ids,
        prompt_len,
    )
    policy_rejected_logp = response_log_prob(
        policy,
        rejected_ids,
        prompt_len,
    )

    with torch.no_grad():
        reference_chosen_logp = response_log_prob(
            reference,
            chosen_ids,
            prompt_len,
        )
        reference_rejected_logp = response_log_prob(
            reference,
            rejected_ids,
            prompt_len,
        )

    print("\n--- Forward ---")
    print(
        "policy chosen logp :",
        policy_chosen_logp.detach().tolist(),
    )
    print(
        "policy rejected logp:",
        policy_rejected_logp.detach().tolist(),
    )
    print(
        "reference chosen logp :",
        reference_chosen_logp.tolist(),
    )
    print(
        "reference rejected logp:",
        reference_rejected_logp.tolist(),
    )

    loss, preference_probability = dpo_loss(
        policy_chosen_logp=policy_chosen_logp,
        policy_rejected_logp=policy_rejected_logp,
        reference_chosen_logp=reference_chosen_logp,
        reference_rejected_logp=reference_rejected_logp,
        beta=0.1,
    )

    print("DPO loss:", float(loss.detach()))
    print(
        "P(chosen > rejected):",
        preference_probability.detach().tolist(),
    )

    # ------------------------------------------------------------
    # 4. Backward：DPO Loss 反向传播
    # ------------------------------------------------------------
    optimizer.zero_grad()
    loss.backward()

    grad_norm = torch.sqrt(
        sum(
            parameter.grad.detach().pow(2).sum()
            for parameter in policy.parameters()
            if parameter.grad is not None
        )
    )

    print("\n--- Backward ---")
    print("policy gradient norm:", float(grad_norm))
    print(
        "reference has gradient:",
        any(
            parameter.grad is not None
            for parameter in reference.parameters()
        ),
    )

    optimizer.step()

    # ------------------------------------------------------------
    # 5. 更新后再次观察 chosen / rejected 的相对差距
    # ------------------------------------------------------------
    policy.eval()

    with torch.no_grad():
        new_chosen_logp = response_log_prob(
            policy,
            chosen_ids,
            prompt_len,
        )
        new_rejected_logp = response_log_prob(
            policy,
            rejected_ids,
            prompt_len,
        )

        old_margin = (
            policy_chosen_logp
            - policy_rejected_logp
        )
        new_margin = (
            new_chosen_logp
            - new_rejected_logp
        )

    print("\n--- After One Optimizer Step ---")
    print("old chosen-rejected margin:", old_margin.tolist())
    print("new chosen-rejected margin:", new_margin.tolist())
    print(
        "margin increased:",
        bool((new_margin > old_margin).item()),
    )


if __name__ == "__main__":
    main()