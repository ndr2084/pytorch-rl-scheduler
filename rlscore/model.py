"""PyTorch policy/value network for the RL scheduler.

Each scheduling decision has a variable number of candidate nodes (however
many passed the Filter phase for this pod), so the network scores nodes
independently through a shared trunk rather than assuming a fixed action
space: forward() takes [N, FEATURE_DIM] (all candidates for one decision)
and returns per-node logits [N]. A categorical distribution over those
logits is the policy; softmax(logits) is invariant to N by construction, so
the same network handles a 3-node cluster and a 1500-node cluster.
"""

import torch
import torch.nn as nn

from rlscore.features import FEATURE_DIM

HIDDEN_DIM = 64


class PolicyValueNet(nn.Module):
    def __init__(self, feature_dim: int = FEATURE_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_dim, 1)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor):
        """features: [N, feature_dim] -- the N candidate nodes for one decision.

        Returns (logits [N], value []): value is a single V(s) estimate for
        this decision, pooled across all candidates.
        """
        embed = self.trunk(features)  # [N, hidden_dim]
        logits = self.policy_head(embed).squeeze(-1)  # [N]
        pooled = embed.mean(dim=0)  # [hidden_dim]
        value = self.value_head(pooled).squeeze(-1)  # scalar
        return logits, value

    def score(self, features: torch.Tensor) -> torch.Tensor:
        """Maps a raw per-node logit to a [0,100] score for RLScorer.ScorePlacement
        / ScoreBatch -- independent of other candidates, matching the
        Kubernetes ScorePlugin contract (each node scored on its own).
        features: [N, feature_dim]. Returns [N] in [0,100].
        """
        embed = self.trunk(features)
        logits = self.policy_head(embed).squeeze(-1)
        return torch.sigmoid(logits) * 100.0
