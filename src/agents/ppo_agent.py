"""Actor-critic network for PPO.

Adapted from CleanRL's ppo.py / ppo_continuous_action.py (MIT License,
https://github.com/vwxyzjn/cleanrl). Separate actor and critic MLPs with
orthogonal initialization. Supports Discrete and Box (continuous) action spaces.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal


def layer_init(layer: nn.Linear, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Linear:
    """Orthogonal weight init, constant bias. Standard for PPO."""
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


def _mlp(in_dim: int, hidden_sizes: Sequence[int], out_dim: int, out_std: float) -> nn.Sequential:
    layers = []
    last = in_dim
    for h in hidden_sizes:
        layers += [layer_init(nn.Linear(last, h)), nn.Tanh()]
        last = h
    layers.append(layer_init(nn.Linear(last, out_dim), std=out_std))
    return nn.Sequential(*layers)


class PPOAgent(nn.Module):
    """Actor-critic with independent policy and value networks.

    Args:
        obs_dim: Flattened observation size.
        action_dim: Number of discrete actions, or size of the continuous action vector.
        continuous: True for a Gaussian policy over a Box action space.
        hidden_sizes: Hidden layer widths shared by actor and critic.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        continuous: bool = False,
        hidden_sizes: Sequence[int] = (64, 64),
    ):
        super().__init__()
        self.continuous = continuous
        # Small output std on the policy head keeps the initial policy near uniform;
        # std=1.0 on the value head is the usual choice.
        self.critic = _mlp(obs_dim, hidden_sizes, 1, out_std=1.0)
        self.actor = _mlp(obs_dim, hidden_sizes, action_dim, out_std=0.01)
        if continuous:
            # State-independent log std, learned.
            self.actor_logstd = nn.Parameter(torch.zeros(1, action_dim))

    @classmethod
    def from_envs(cls, envs: gym.vector.VectorEnv, **kwargs) -> "PPOAgent":
        """Build an agent sized to a Gymnasium vector env's single obs/action spaces."""
        obs_dim = int(np.prod(envs.single_observation_space.shape))
        space = envs.single_action_space
        if isinstance(space, gym.spaces.Discrete):
            return cls(obs_dim, int(space.n), continuous=False, **kwargs)
        if isinstance(space, gym.spaces.Box):
            return cls(obs_dim, int(np.prod(space.shape)), continuous=True, **kwargs)
        raise NotImplementedError(f"Unsupported action space: {space}")

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def get_action_and_value(
        self, obs: torch.Tensor, action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action (or score a given one).

        Returns:
            action, log_prob(action), entropy, value  (log_prob/entropy/value have shape [B])
        """
        out = self.actor(obs)
        if self.continuous:
            std = torch.exp(self.actor_logstd.expand_as(out))
            dist = Normal(out, std)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            entropy = dist.entropy().sum(-1)
        else:
            dist = Categorical(logits=out)
            if action is None:
                action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
        return action, log_prob, entropy, self.critic(obs).squeeze(-1)
