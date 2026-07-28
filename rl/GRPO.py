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
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=num_layers,
        )
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

        return self.lm_head(hidden)  # [B, T, V]


def response_token_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """
    计算回答部分每个 Token 的 log probability。

    input_ids: [G, T]
    return: [G, R]
        G = 同一 Prompt 下的回答数量
        R = 回答 Token 数量
    """
    logits = model(input_ids)  # [G, T, V]

    # 位置 t 的 logits 用来预测位置 t+1 的 Token。
    next_token_logits = logits[:, :-1, :]  # [G, T-1, V]
    next_token_ids = input_ids[:, 1:]      # [G, T-1]

    all_token_log_probs = F.log_softmax(
        next_token_logits,
        dim=-1,
    ).gather(
        dim=-1,
        index=next_token_ids.unsqueeze(-1),
    ).squeeze(-1)  # [G, T-1]

    # 回答从原序列位置 prompt_len 开始，
    # 对应 shifted 序列中的索引 prompt_len - 1。
    return all_token_log_probs[:, prompt_len - 1:]  # [G, R]


def reward_function(
    input_ids: torch.Tensor,
    prompt_len: int,
    vocab_size: int,
) -> torch.Tensor:
    """
    一个不可导的玩具奖励函数，模拟 Reward Model / Rule Reward。

    Token 语义：
    4=有帮助，奖励 +1.0
    5=有细节，奖励 +0.5
    7=敷衍，奖励 -0.5
    8=无理由拒绝，奖励 -1.0
    """
    token_rewards = torch.zeros(
        vocab_size,
        dtype=torch.float32,
        device=input_ids.device,
    )
    token_rewards[4] = 1.0
    token_rewards[5] = 0.5
    token_rewards[7] = -0.5
    token_rewards[8] = -1.0

    response_ids = input_ids[:, prompt_len:]  # [G, R]
    return token_rewards[response_ids].sum(dim=-1)  # [G]


def normalize_group_rewards(
    rewards: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    GRPO 的关键：不用 Value Model，而是在同组回答中构造相对优势。

    A_i = (r_i - mean(r)) / std(r)
    """
    return (
        rewards - rewards.mean()
    ) / (
        rewards.std(unbiased=False) + eps
    )


def grpo_loss(
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Token-level GRPO Loss。

    ratio = πθ(token) / πold(token)

    policy objective =
        min(ratio * A, clip(ratio) * A)

    sampled-token KL estimator =
        πref / πθ - log(πref / πθ) - 1
    """
    advantages = advantages.detach().unsqueeze(-1)  # [G, 1]

    ratio = torch.exp(
        policy_log_probs - old_log_probs
    )  # [G, R]

    unclipped_objective = ratio * advantages
    clipped_objective = torch.clamp(
        ratio,
        1.0 - clip_eps,
        1.0 + clip_eps,
    ) * advantages

    policy_objective = torch.minimum(
        unclipped_objective,
        clipped_objective,
    )  # [G, R]

    ref_policy_log_ratio = (
        reference_log_probs - policy_log_probs
    )

    kl = (
        torch.exp(ref_policy_log_ratio)
        - ref_policy_log_ratio
        - 1.0
    )  # [G, R]

    loss = -(
        policy_objective - kl_beta * kl
    ).mean()

    stats = {
        "mean_ratio": ratio.mean().detach(),
        "mean_kl": kl.mean().detach(),
        "mean_policy_objective": policy_objective.mean().detach(),
    }
    return loss, stats


def sequence_log_probs(
    token_log_probs: torch.Tensor,
) -> torch.Tensor:
    """将回答 Token 的 log probability 求和为序列 log probability。"""
    return token_log_probs.sum(dim=-1)


def main() -> None:
    torch.manual_seed(42)

    # ------------------------------------------------------------
    # 1. 同一个 Prompt 下的一组回答
    # ------------------------------------------------------------
    # Token:
    # 0=<pad>, 1=<bos>, 2=用户, 3=问题,
    # 4=有帮助, 5=有细节, 6=<eos>,
    # 7=敷衍, 8=无理由拒绝
    vocab_size = 9
    prompt_len = 3

    group_input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6],  # 有帮助 + 有细节
            [1, 2, 3, 4, 7, 6],  # 有帮助 + 敷衍
            [1, 2, 3, 7, 5, 6],  # 敷衍 + 有细节
            [1, 2, 3, 7, 8, 6],  # 敷衍 + 无理由拒绝
        ],
        dtype=torch.long,
    )  # [G=4, T=6]

    print("group_input_ids shape:", tuple(group_input_ids.shape))

    # ------------------------------------------------------------
    # 2. Policy、Old Policy、Reference
    # ------------------------------------------------------------
    policy = TinyCausalLM(vocab_size=vocab_size)

    # Old Policy：采样这组回答时使用的旧策略，用于 importance ratio。
    old_policy = copy.deepcopy(policy)
    old_policy.eval()

    # Reference：通常是 SFT 模型的冻结副本，用于限制策略漂移。
    reference = copy.deepcopy(policy)
    reference.eval()

    for model in (old_policy, reference):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=3e-3,
    )

    # ------------------------------------------------------------
    # 3. Reward -> Group Relative Advantage
    # ------------------------------------------------------------
    rewards = reward_function(
        group_input_ids,
        prompt_len,
        vocab_size,
    )  # [G]

    advantages = normalize_group_rewards(rewards)  # [G]

    print("\n--- Reward / Advantage ---")
    print("rewards   :", rewards.tolist())
    print("advantages:", advantages.tolist())
    print("advantage mean:", float(advantages.mean()))

    # ------------------------------------------------------------
    # 4. Forward：三个模型计算回答 Token 概率
    # ------------------------------------------------------------
    policy.train()

    policy_log_probs = response_token_log_probs(
        policy,
        group_input_ids,
        prompt_len,
    )  # [G, R]

    with torch.no_grad():
        old_log_probs = response_token_log_probs(
            old_policy,
            group_input_ids,
            prompt_len,
        )
        reference_log_probs = response_token_log_probs(
            reference,
            group_input_ids,
            prompt_len,
        )

    print("\n--- Forward Shapes ---")
    print("policy_log_probs   :", tuple(policy_log_probs.shape))
    print("old_log_probs      :", tuple(old_log_probs.shape))
    print("reference_log_probs:", tuple(reference_log_probs.shape))

    before_sequence_log_probs = sequence_log_probs(
        policy_log_probs
    ).detach()

    loss, stats = grpo_loss(
        policy_log_probs=policy_log_probs,
        old_log_probs=old_log_probs,
        reference_log_probs=reference_log_probs,
        advantages=advantages,
        clip_eps=0.2,
        kl_beta=0.04,
    )

    print("\n--- GRPO Loss ---")
    print("loss                 :", float(loss.detach()))
    print("mean ratio           :", float(stats["mean_ratio"]))
    print("mean KL              :", float(stats["mean_kl"]))
    print(
        "mean policy objective:",
        float(stats["mean_policy_objective"]),
    )

    # ------------------------------------------------------------
    # 5. Backward：只更新 Policy
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
        "old policy has gradient:",
        any(
            parameter.grad is not None
            for parameter in old_policy.parameters()
        ),
    )
    print(
        "reference has gradient:",
        any(
            parameter.grad is not None
            for parameter in reference.parameters()
        ),
    )

    optimizer.step()

    # ------------------------------------------------------------
    # 6. 更新后观察：高奖励回答相对低奖励回答更受偏好
    # ------------------------------------------------------------
    policy.eval()

    with torch.no_grad():
        after_sequence_log_probs = sequence_log_probs(
            response_token_log_probs(
                policy,
                group_input_ids,
                prompt_len,
            )
        )

    best_index = int(rewards.argmax())
    worst_index = int(rewards.argmin())

    before_margin = (
        before_sequence_log_probs[best_index]
        - before_sequence_log_probs[worst_index]
    )
    after_margin = (
        after_sequence_log_probs[best_index]
        - after_sequence_log_probs[worst_index]
    )

    print("\n--- After One Optimizer Step ---")
    print(
        "before sequence log probs:",
        before_sequence_log_probs.tolist(),
    )
    print(
        "after sequence log probs :",
        after_sequence_log_probs.tolist(),
    )
    print("best-worst margin before:", float(before_margin))
    print("best-worst margin after :", float(after_margin))
    print(
        "margin increased:",
        bool(after_margin > before_margin),
    )


if __name__ == "__main__":
    main()