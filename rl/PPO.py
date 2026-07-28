import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, state_dim=4, hidden_dim=32, num_actions=2):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden_dim, num_actions)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, states):
        """
        states: [T, state_dim]
        logits: [T, num_actions]
        values: [T]
        """
        hidden = self.backbone(states)
        logits = self.actor(hidden)
        values = self.critic(hidden).squeeze(-1)
        return logits, values


def compute_returns(rewards, dones, gamma=0.99):
    """G_t = r_t + gamma * (1-done_t) * G_{t+1}"""
    returns = torch.zeros_like(rewards)
    running_return = torch.tensor(0.0)

    for t in reversed(range(len(rewards))):
        running_return = (
            rewards[t]
            + gamma * (1.0 - dones[t]) * running_return
        )
        returns[t] = running_return

    return returns


def ppo_loss(
    new_log_probs,
    old_log_probs,
    advantages,
    new_values,
    returns,
    entropy,
    clip_eps=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
):
    """
    ratio = pi_theta(a|s) / pi_old(a|s)

    policy objective =
        min(ratio*A, clip(ratio)*A)
    """
    old_log_probs = old_log_probs.detach()
    advantages = advantages.detach()
    returns = returns.detach()

    ratio = torch.exp(new_log_probs - old_log_probs)

    objective_1 = ratio * advantages
    objective_2 = torch.clamp(
        ratio,
        1.0 - clip_eps,
        1.0 + clip_eps,
    ) * advantages

    policy_loss = -torch.minimum(
        objective_1,
        objective_2,
    ).mean()

    value_loss = F.mse_loss(new_values, returns)
    entropy_bonus = entropy.mean()

    total_loss = (
        policy_loss
        + value_coef * value_loss
        - entropy_coef * entropy_bonus
    )

    stats = {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "entropy": entropy_bonus.detach(),
        "ratio": ratio.detach(),
    }
    return total_loss, stats


def main():
    torch.manual_seed(42)

    # 1. 当前 Policy 与冻结的 Old Policy
    policy = ActorCritic()
    old_policy = copy.deepcopy(policy)
    old_policy.eval()

    for parameter in old_policy.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)

    # 2. 一条包含 4 个时间步的轨迹
    states = torch.tensor(
        [
            [1.0, 0.0, 0.2, -0.1],
            [0.3, 0.8, -0.2, 0.1],
            [0.9, 0.1, 0.4, -0.3],
            [0.2, 1.0, -0.1, 0.2],
        ],
        dtype=torch.float32,
    )  # [T=4, state_dim=4]

    target_actions = torch.tensor([0, 1, 0, 1])
    dones = torch.tensor([0.0, 0.0, 0.0, 1.0])

    # 3. Rollout：Old Policy 采样动作
    with torch.no_grad():
        old_logits, old_values = old_policy(states)
        old_dist = Categorical(logits=old_logits)

        actions = old_dist.sample()
        old_log_probs = old_dist.log_prob(actions)

    # 环境奖励不可导，但不影响策略梯度
    rewards = torch.where(
        actions == target_actions,
        torch.tensor(1.0),
        torch.tensor(-0.5),
    )

    returns = compute_returns(rewards, dones)

    # 最简单 Advantage，也可替换为 GAE
    advantages = returns - old_values

    print("--- Rollout ---")
    print("states shape:", tuple(states.shape))
    print("actions     :", actions.tolist())
    print("rewards     :", rewards.tolist())
    print("returns     :", returns.tolist())
    print("old values  :", old_values.tolist())
    print("advantages  :", advantages.tolist())

    # 4. Forward：当前 Policy 重新评估旧轨迹
    new_logits, new_values = policy(states)
    new_dist = Categorical(logits=new_logits)

    new_log_probs = new_dist.log_prob(actions)
    entropy = new_dist.entropy()

    loss, stats = ppo_loss(
        new_log_probs,
        old_log_probs,
        advantages,
        new_values,
        returns,
        entropy,
    )

    print("\n--- Forward / Loss ---")
    print("logits shape :", tuple(new_logits.shape))
    print("values shape :", tuple(new_values.shape))
    print("mean ratio   :", float(stats["ratio"].mean()))
    print("policy loss  :", float(stats["policy_loss"]))
    print("value loss   :", float(stats["value_loss"]))
    print("entropy      :", float(stats["entropy"]))
    print("total loss   :", float(loss.detach()))

    # 5. Backward：同时更新 Actor 和 Critic
    optimizer.zero_grad()
    loss.backward()

    actor_grad_norm = torch.sqrt(sum(
        p.grad.pow(2).sum()
        for p in policy.actor.parameters()
        if p.grad is not None
    ))

    critic_grad_norm = torch.sqrt(sum(
        p.grad.pow(2).sum()
        for p in policy.critic.parameters()
        if p.grad is not None
    ))

    print("\n--- Backward ---")
    print("actor grad norm :", float(actor_grad_norm))
    print("critic grad norm:", float(critic_grad_norm))
    print(
        "old policy has gradient:",
        any(p.grad is not None for p in old_policy.parameters()),
    )

    before_log_probs = new_log_probs.detach().clone()
    before_values = new_values.detach().clone()

    optimizer.step()

    # 6. 更新后再次观察
    with torch.no_grad():
        after_logits, after_values = policy(states)
        after_dist = Categorical(logits=after_logits)
        after_log_probs = after_dist.log_prob(actions)

    print("\n--- After One Step ---")
    print("before action logp:", before_log_probs.tolist())
    print("after action logp :", after_log_probs.tolist())
    print(
        "logp changes      :",
        (after_log_probs - before_log_probs).tolist(),
    )
    print("before values     :", before_values.tolist())
    print("after values      :", after_values.tolist())


if __name__ == "__main__":
    main()