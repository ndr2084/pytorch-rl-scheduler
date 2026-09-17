"""A small, from-scratch PPO implementation (Schulman et al., 2017) with GAE
advantages, a clipped surrogate objective, and an entropy bonus.

Each decision has a different number of candidate nodes (see env.py), so
this is not a fixed-action-space PPO: the rollout buffer stores one
variable-length feature tensor per timestep, and the update loop runs the
model per stored timestep (rather than one batched [B, N, D] forward pass)
so the network's own permutation-invariant, N-agnostic forward pass (see
model.PolicyValueNet) is exercised the same way at train and update time.
That trades batched-matmul throughput for correctness/simplicity -- fine at
the scale this trains at (per-step candidate counts of a few hundred, not
the millions a production PPO implementation would need to batch over).
"""

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from rlscore.env import Observation, SchedulingEnv
from rlscore.model import PolicyValueNet


@dataclass
class RolloutBuffer:
    features: List[torch.Tensor] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    log_probs: List[float] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    dones: List[bool] = field(default_factory=list)

    def add(self, features, action, log_prob, value, reward, done):
        self.features.append(features)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.rewards)


def compute_gae(
    rewards: List[float], values: List[float], dones: List[bool], last_value: float, gamma: float, lam: float
) -> Tuple[List[float], List[float]]:
    advantages = [0.0] * len(rewards)
    gae = 0.0
    next_value = last_value
    for t in reversed(range(len(rewards))):
        next_non_terminal = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae
        next_value = values[t]
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns


def collect_rollout(
    env: SchedulingEnv,
    model: PolicyValueNet,
    n_steps: int,
    obs: Optional[Observation],
    device: str = "cpu",
) -> Tuple[RolloutBuffer, float, List[float], Optional[Observation]]:
    """Runs the current policy for n_steps env steps, resetting on episode
    end. Returns the rollout, a bootstrap value for the final state, the
    completed episodes' total rewards, and the observation to resume from
    on the next call."""
    buffer = RolloutBuffer()
    episode_returns: List[float] = []
    ep_reward = 0.0

    if obs is None:
        obs = env.reset()

    for _ in range(n_steps):
        features = obs.features.to(device)
        with torch.no_grad():
            logits, value = model(features)
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

        next_obs, reward, done, _info = env.step(int(action.item()))
        buffer.add(features.cpu(), int(action.item()), float(log_prob.item()), float(value.item()), reward, done)
        ep_reward += reward

        if done:
            episode_returns.append(ep_reward)
            ep_reward = 0.0
            next_obs = env.reset()
        obs = next_obs

    with torch.no_grad():
        _, last_value_t = model(obs.features.to(device))
    return buffer, float(last_value_t.item()), episode_returns, obs


def ppo_update(
    model: PolicyValueNet,
    optimizer: torch.optim.Optimizer,
    buffer: RolloutBuffer,
    last_value: float,
    gamma: float = 0.99,
    lam: float = 0.95,
    clip_eps: float = 0.2,
    epochs: int = 4,
    minibatch_size: int = 64,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    max_grad_norm: float = 0.5,
) -> dict:
    advantages, returns = compute_gae(buffer.rewards, buffer.values, buffer.dones, last_value, gamma, lam)
    advantages_t = torch.tensor(advantages, dtype=torch.float32)
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std(unbiased=False) + 1e-8)
    returns_t = torch.tensor(returns, dtype=torch.float32)
    old_log_probs_t = torch.tensor(buffer.log_probs, dtype=torch.float32)

    n = len(buffer)
    indices = list(range(n))
    totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
    n_minibatches = 0

    for _ in range(epochs):
        random.shuffle(indices)
        for start in range(0, n, minibatch_size):
            batch_idx = indices[start : start + minibatch_size]
            policy_losses, value_losses, entropies = [], [], []

            for i in batch_idx:
                logits, value = model(buffer.features[i])
                dist = torch.distributions.Categorical(logits=logits)
                new_log_prob = dist.log_prob(torch.tensor(buffer.actions[i]))
                entropy = dist.entropy()

                ratio = torch.exp(new_log_prob - old_log_probs_t[i])
                surr1 = ratio * advantages_t[i]
                surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_t[i]
                policy_losses.append(-torch.min(surr1, surr2))
                value_losses.append((value - returns_t[i]) ** 2)
                entropies.append(entropy)

            policy_loss = torch.stack(policy_losses).mean()
            value_loss = torch.stack(value_losses).mean()
            entropy_bonus = torch.stack(entropies).mean()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            totals["policy_loss"] += policy_loss.item()
            totals["value_loss"] += value_loss.item()
            totals["entropy"] += entropy_bonus.item()
            n_minibatches += 1

    return {k: v / max(n_minibatches, 1) for k, v in totals.items()}
