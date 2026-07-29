from __future__ import annotations

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class TinyPolicyValueLM(nn.Module):
    """极小的 Decoder-only LM，同时带 Policy Head 和 Value Head。"""

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
        self.value_head = nn.Linear(d_model, 1)

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        input_ids: [B, T]

        return:
            logits: [B, T, V]
            values: [B, T]
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

        logits = self.lm_head(hidden)                  # [B, T, V]
        values = self.value_head(hidden).squeeze(-1)  # [B, T]

        return logits, values


class TinyRewardModel(nn.Module):
    """极小 Reward Model：对整段 Prompt + Response 输出一个标量分数。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 32,
    ) -> None:
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scorer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input_ids: [B, T]
        return reward: [B]
        """
        hidden = self.embedding(input_ids)  # [B, T, D]
        pooled = hidden.mean(dim=1)         # [B, D]
        return self.scorer(pooled).squeeze(-1)


@torch.no_grad()
def generate_response(
    model: TinyPolicyValueLM,
    prompt_ids: torch.Tensor,
    response_len: int,
) -> torch.Tensor:
    """
    用 Old Policy 自回归采样固定长度回答。

    prompt_ids: [B, P]
    return full_ids: [B, P + R]
    """
    full_ids = prompt_ids.clone()

    for _ in range(response_len):
        logits, _ = model(full_ids)
        next_token_dist = Categorical(
            logits=logits[:, -1, :],
        )
        next_token = next_token_dist.sample()  # [B]
        full_ids = torch.cat(
            [full_ids, next_token.unsqueeze(-1)],
            dim=-1,
        )

    return full_ids


def response_statistics(
    model: TinyPolicyValueLM,
    full_ids: torch.Tensor,
    prompt_len: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    取得回答部分的：
    1. 已生成 Token 的 log probability
    2. 完整词表 logits
    3. 每个动作位置对应的 Value

    return:
        token_log_probs: [B, R]
        response_logits: [B, R, V]
        response_values: [B, R]
    """
    logits, values = model(full_ids)  # [B, T, V], [B, T]

    # 位置 t 的输出预测位置 t+1 的 Token。
    action_logits = logits[:, :-1, :]  # [B, T-1, V]
    action_targets = full_ids[:, 1:]   # [B, T-1]
    action_values = values[:, :-1]     # [B, T-1]

    # 回答第一个 Token 位于原序列 prompt_len，
    # 因此由位置 prompt_len-1 的 logits 预测。
    response_logits = action_logits[:, prompt_len - 1:, :]  # [B, R, V]
    response_targets = action_targets[:, prompt_len - 1:]    # [B, R]
    response_values = action_values[:, prompt_len - 1:]      # [B, R]

    token_log_probs = F.log_softmax(
        response_logits,
        dim=-1,
    ).gather(
        dim=-1,
        index=response_targets.unsqueeze(-1),
    ).squeeze(-1)  # [B, R]

    return token_log_probs, response_logits, response_values


def discounted_returns(
    token_rewards: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    对回答 Token 从后向前计算 Return。

    token_rewards: [B, R]
    return: [B, R]
    """
    returns = torch.zeros_like(token_rewards)
    running_return = torch.zeros(
        token_rewards.size(0),
        device=token_rewards.device,
    )

    for t in reversed(range(token_rewards.size(1))):
        running_return = (
            token_rewards[:, t]
            + gamma * running_return
        )
        returns[:, t] = running_return

    return returns


def exact_reference_kl(
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
) -> torch.Tensor:
    """
    计算回答位置上的精确 Token 分布 KL：

    KL(pi_theta || pi_ref)
    """
    policy_log_probs = F.log_softmax(
        policy_logits,
        dim=-1,
    )
    reference_log_probs = F.log_softmax(
        reference_logits,
        dim=-1,
    )
    policy_probs = policy_log_probs.exp()

    token_kl = (
        policy_probs
        * (policy_log_probs - reference_log_probs)
    ).sum(dim=-1)  # [B, R]

    return token_kl.mean()


def ppo_rlhf_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    new_values: torch.Tensor,
    returns: torch.Tensor,
    policy_logits: torch.Tensor,
    reference_logits: torch.Tensor,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    kl_beta: float = 0.04,
    entropy_coef: float = 0.01,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    LLM-RLHF PPO Loss：

    total =
        PPO Clip Policy Loss
        + value_coef * Value Loss
        + kl_beta * KL(policy || reference)
        - entropy_coef * Entropy
    """
    old_log_probs = old_log_probs.detach()
    advantages = advantages.detach()
    returns = returns.detach()
    reference_logits = reference_logits.detach()

    # PPO importance ratio：当前策略 / 生成数据的旧策略
    ratio = torch.exp(
        new_log_probs - old_log_probs
    )  # [B, R]

    unclipped_objective = ratio * advantages
    clipped_objective = torch.clamp(
        ratio,
        1.0 - clip_eps,
        1.0 + clip_eps,
    ) * advantages

    policy_loss = -torch.minimum(
        unclipped_objective,
        clipped_objective,
    ).mean()

    value_loss = F.mse_loss(
        new_values,
        returns,
    )

    reference_kl = exact_reference_kl(
        policy_logits,
        reference_logits,
    )

    entropy = Categorical(
        logits=policy_logits,
    ).entropy().mean()

    total_loss = (
        policy_loss
        + value_coef * value_loss
        + kl_beta * reference_kl
        - entropy_coef * entropy
    )

    stats = {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "reference_kl": reference_kl.detach(),
        "entropy": entropy.detach(),
        "mean_ratio": ratio.mean().detach(),
    }
    return total_loss, stats


def grad_norm(parameters) -> torch.Tensor:
    squared_norm = sum(
        parameter.grad.detach().pow(2).sum()
        for parameter in parameters
        if parameter.grad is not None
    )
    return torch.sqrt(squared_norm)


def main() -> None:
    torch.manual_seed(42)

    vocab_size = 12
    prompt_ids = torch.tensor(
        [[1, 2, 3]],
        dtype=torch.long,
    )  # [B=1, P=3]

    prompt_len = prompt_ids.size(1)
    response_len = 4

    # ------------------------------------------------------------
    # 1. 创建 Reference、Policy、Old Policy、Reward Model
    # ------------------------------------------------------------
    sft_model = TinyPolicyValueLM(vocab_size=vocab_size)

    # Reference：RLHF 开始前的 SFT 模型，整个训练过程长期冻结。
    reference = copy.deepcopy(sft_model)

    # Policy：从 SFT 模型初始化，后续被 PPO 更新。
    policy = copy.deepcopy(sft_model)

    # 模拟已经进行过少量 RL 更新，使 Policy 与 Reference 略有差异，
    # 这样本次示例中的 Reference KL 不会恰好为 0。
    with torch.no_grad():
        for parameter in policy.lm_head.parameters():
            parameter.add_(
                0.02 * torch.randn_like(parameter)
            )

    # Old Policy：当前 Policy 的短期快照，用于本轮 Rollout。
    old_policy = copy.deepcopy(policy)

    # Reward Model：训练好的奖励模型在 PPO 阶段保持冻结。
    reward_model = TinyRewardModel(vocab_size=vocab_size)

    for frozen_model in (
        reference,
        old_policy,
        reward_model,
    ):
        frozen_model.eval()
        for parameter in frozen_model.parameters():
            parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=3e-3,
    )

    # ------------------------------------------------------------
    # 2. Rollout：Old Policy 根据 Prompt 生成回答
    # ------------------------------------------------------------
    full_ids = generate_response(
        old_policy,
        prompt_ids,
        response_len,
    )  # [B, P+R]

    print("--- Rollout ---")
    print("prompt ids :", prompt_ids.tolist())
    print("full ids   :", full_ids.tolist())
    print("response   :", full_ids[:, prompt_len:].tolist())
    print("full shape :", tuple(full_ids.shape))

    # ------------------------------------------------------------
    # 3. Reward Model 对完整回答打分
    # ------------------------------------------------------------
    with torch.no_grad():
        sequence_reward = reward_model(full_ids)  # [B]

    # 只有最后一个回答 Token 获得终局 Reward。
    token_rewards = torch.zeros(
        full_ids.size(0),
        response_len,
    )
    token_rewards[:, -1] = sequence_reward

    returns = discounted_returns(
        token_rewards,
        gamma=1.0,
    )  # [B, R]

    print("\n--- Reward / Return ---")
    print("sequence reward:", sequence_reward.tolist())
    print("token rewards  :", token_rewards.tolist())
    print("returns        :", returns.tolist())

    # ------------------------------------------------------------
    # 4. Old Policy 与 Reference 计算固定训练目标
    # ------------------------------------------------------------
    with torch.no_grad():
        old_log_probs, _, old_values = response_statistics(
            old_policy,
            full_ids,
            prompt_len,
        )

        _, reference_logits, _ = response_statistics(
            reference,
            full_ids,
            prompt_len,
        )

    # 最简单的 Advantage：Return - Old Value
    advantages = returns - old_values

    print("\n--- Old Policy / Advantage ---")
    print("old log probs:", old_log_probs.tolist())
    print("old values   :", old_values.tolist())
    print("advantages   :", advantages.tolist())

    # ------------------------------------------------------------
    # 5. Forward：当前 Policy 重新评估同一条回答
    # ------------------------------------------------------------
    policy.train()

    new_log_probs, policy_logits, new_values = response_statistics(
        policy,
        full_ids,
        prompt_len,
    )

    loss, stats = ppo_rlhf_loss(
        new_log_probs=new_log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        new_values=new_values,
        returns=returns,
        policy_logits=policy_logits,
        reference_logits=reference_logits,
        clip_eps=0.2,
        value_coef=0.5,
        kl_beta=0.04,
        entropy_coef=0.01,
    )

    print("\n--- Forward Shapes ---")
    print("policy logits:", tuple(policy_logits.shape))
    print("new log probs:", tuple(new_log_probs.shape))
    print("new values   :", tuple(new_values.shape))

    print("\n--- PPO-RLHF Loss ---")
    print("total loss  :", float(loss.detach()))
    print("policy loss :", float(stats["policy_loss"]))
    print("value loss  :", float(stats["value_loss"]))
    print("reference KL:", float(stats["reference_kl"]))
    print("entropy     :", float(stats["entropy"]))
    print("mean ratio  :", float(stats["mean_ratio"]))

    # ------------------------------------------------------------
    # 6. Backward：只更新 Policy + Value
    # ------------------------------------------------------------
    optimizer.zero_grad()
    loss.backward()

    print("\n--- Backward ---")
    print(
        "policy LM-head grad norm:",
        float(grad_norm(policy.lm_head.parameters())),
    )
    print(
        "policy value-head grad norm:",
        float(grad_norm(policy.value_head.parameters())),
    )
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
    print(
        "reward model has gradient:",
        any(
            parameter.grad is not None
            for parameter in reward_model.parameters()
        ),
    )

    before_log_probs = new_log_probs.detach().clone()
    before_values = new_values.detach().clone()
    before_kl = stats["reference_kl"].clone()

    optimizer.step()

    # ------------------------------------------------------------
    # 7. 更新后再次观察当前回答
    # ------------------------------------------------------------
    policy.eval()

    with torch.no_grad():
        after_log_probs, after_logits, after_values = response_statistics(
            policy,
            full_ids,
            prompt_len,
        )
        after_kl = exact_reference_kl(
            after_logits,
            reference_logits,
        )

    print("\n--- After One Optimizer Step ---")
    print(
        "before token log probs:",
        before_log_probs.tolist(),
    )
    print(
        "after token log probs :",
        after_log_probs.tolist(),
    )
    print(
        "log-prob changes      :",
        (after_log_probs - before_log_probs).tolist(),
    )
    print("before values:", before_values.tolist())
    print("after values :", after_values.tolist())
    print("reference KL before:", float(before_kl))
    print("reference KL after :", float(after_kl))

    print("\n模型职责：")
    print("Policy + Value：更新")
    print("Old Policy：本轮采样快照，周期性同步")
    print("Reference：SFT 长期锚点，始终冻结")
    print("Reward Model：给回答打分，PPO 阶段冻结")


if __name__ == "__main__":
    main()