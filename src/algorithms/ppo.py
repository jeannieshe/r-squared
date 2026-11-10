"""Proximal Policy Optimization (clipped objective, GAE).

Adapted from CleanRL's ppo.py (MIT License, https://github.com/vwxyzjn/cleanrl),
split into config / rollout buffer / trainer so the agent and env are pluggable.

Expects a Gymnasium vector env (gymnasium >= 0.29) and an agent exposing
`get_value(obs)` and `get_action_and_value(obs, action=None)`, e.g.
`src.agents.ppo_agent.PPOAgent`.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class PPOConfig:
    total_timesteps: int = 500_000
    num_envs: int = 4
    num_steps: int = 128            # rollout length per env per iteration
    learning_rate: float = 2.5e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 4
    update_epochs: int = 4
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = True
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None  # early-stop epochs if approx KL exceeds this

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_iterations(self) -> int:
        return self.total_timesteps // self.batch_size


class RolloutBuffer:
    """Fixed-size on-policy storage of shape [num_steps, num_envs, ...]."""

    def __init__(self, num_steps: int, num_envs: int, obs_shape, action_shape, device):
        def z(*shape):
            return torch.zeros(shape, device=device)

        self.obs = z(num_steps, num_envs, *obs_shape)
        self.actions = z(num_steps, num_envs, *action_shape)
        self.logprobs = z(num_steps, num_envs)
        self.rewards = z(num_steps, num_envs)
        self.dones = z(num_steps, num_envs)
        self.values = z(num_steps, num_envs)
        self.advantages = z(num_steps, num_envs)
        self.returns = z(num_steps, num_envs)
        self.num_steps = num_steps

    def compute_returns_and_advantages(
        self, next_value: torch.Tensor, next_done: torch.Tensor, gamma: float, gae_lambda: float
    ) -> None:
        """Generalized Advantage Estimation (Schulman et al., 2016)."""
        lastgaelam = 0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            self.advantages[t] = lastgaelam
        self.returns = self.advantages + self.values

    def flatten(self, obs_shape, action_shape) -> Dict[str, torch.Tensor]:
        """Merge the step and env dims into one batch dim."""
        return {
            "obs": self.obs.reshape((-1, *obs_shape)),
            "actions": self.actions.reshape((-1, *action_shape)),
            "logprobs": self.logprobs.reshape(-1),
            "advantages": self.advantages.reshape(-1),
            "returns": self.returns.reshape(-1),
            "values": self.values.reshape(-1),
        }


def _episode_returns(infos: dict) -> List[float]:
    """Pull finished-episode returns written by gym.wrappers.RecordEpisodeStatistics."""
    if "episode" in infos:  # gymnasium >= 1.0
        mask = infos["_episode"]
        return [float(r) for r in np.asarray(infos["episode"]["r"])[mask]]
    if "final_info" in infos:  # gymnasium 0.29
        return [float(i["episode"]["r"]) for i in infos["final_info"] if i and "episode" in i]
    return []


class PPO:
    """PPO trainer.

    Args:
        agent: actor-critic module.
        envs: Gymnasium vector env with `config.num_envs` sub-envs.
        config: hyperparameters.
        device: torch device.
        log_fn: optional `log_fn(metrics: dict, global_step: int)` hook (e.g. TensorBoard/W&B).
    """

    def __init__(
        self,
        agent: nn.Module,
        envs: gym.vector.VectorEnv,
        config: PPOConfig,
        device: torch.device = torch.device("cpu"),
        log_fn: Optional[Callable[[dict, int], None]] = None,
    ):
        assert envs.num_envs == config.num_envs, "config.num_envs must match the vector env"
        self.agent = agent.to(device)
        self.envs = envs
        self.cfg = config
        self.device = device
        self.log_fn = log_fn or (lambda metrics, step: None)
        self.optimizer = optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)

        self.obs_shape = envs.single_observation_space.shape
        self.action_shape = envs.single_action_space.shape
        self.buffer = RolloutBuffer(
            config.num_steps, config.num_envs, self.obs_shape, self.action_shape, device
        )
        self.global_step = 0

    def _to_tensor(self, x) -> torch.Tensor:
        return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)

    def collect_rollouts(self, next_obs: torch.Tensor, next_done: torch.Tensor):
        """Step the envs `num_steps` times with the current policy and fill the buffer."""
        buf = self.buffer
        for step in range(self.cfg.num_steps):
            self.global_step += self.cfg.num_envs
            buf.obs[step] = next_obs
            buf.dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(next_obs)
            buf.values[step] = value
            buf.actions[step] = action
            buf.logprobs[step] = logprob

            obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
            # Note: truncation is treated as termination (no value bootstrap), as in CleanRL.
            done = np.logical_or(terminations, truncations)
            buf.rewards[step] = self._to_tensor(reward).view(-1)
            next_obs, next_done = self._to_tensor(obs), self._to_tensor(done)

            for r in _episode_returns(infos):
                self.log_fn({"charts/episodic_return": r}, self.global_step)

        with torch.no_grad():
            next_value = self.agent.get_value(next_obs).reshape(1, -1)
        buf.compute_returns_and_advantages(next_value, next_done, self.cfg.gamma, self.cfg.gae_lambda)
        return next_obs, next_done

    def update(self) -> Dict[str, float]:
        """Run `update_epochs` passes of clipped-PPO minibatch SGD over the buffer."""
        cfg = self.cfg
        b = self.buffer.flatten(self.obs_shape, self.action_shape)
        b_inds = np.arange(cfg.batch_size)
        clipfracs = []
        approx_kl = old_approx_kl = torch.tensor(0.0)

        for _ in range(cfg.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.batch_size, cfg.minibatch_size):
                mb = b_inds[start:start + cfg.minibatch_size]
                mb_actions = b["actions"][mb]
                if not self.agent.continuous:
                    mb_actions = mb_actions.long()
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(b["obs"][mb], mb_actions)
                logratio = newlogprob - b["logprobs"][mb]
                ratio = logratio.exp()

                with torch.no_grad():
                    # http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item())

                mb_adv = b["advantages"][mb]
                if cfg.norm_adv:
                    mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Clipped policy loss
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss (optionally clipped around the old value estimate)
                if cfg.clip_vloss:
                    v_unclipped = (newvalue - b["returns"][mb]) ** 2
                    v_clipped = b["values"][mb] + torch.clamp(
                        newvalue - b["values"][mb], -cfg.clip_coef, cfg.clip_coef
                    )
                    v_loss = 0.5 * torch.max(v_unclipped, (v_clipped - b["returns"][mb]) ** 2).mean()
                else:
                    v_loss = 0.5 * ((newvalue - b["returns"][mb]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + cfg.vf_coef * v_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

            if cfg.target_kl is not None and approx_kl > cfg.target_kl:
                break

        y_pred, y_true = b["values"].cpu().numpy(), b["returns"].cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        return {
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": float(np.mean(clipfracs)),
            "losses/explained_variance": float(explained_var),
        }

    def train(self, seed: Optional[int] = None) -> nn.Module:
        """Full training loop. Returns the trained agent."""
        cfg = self.cfg
        obs, _ = self.envs.reset(seed=seed)
        next_obs = self._to_tensor(obs)
        next_done = torch.zeros(cfg.num_envs, device=self.device)
        start_time = time.time()

        for iteration in range(1, cfg.num_iterations + 1):
            if cfg.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / cfg.num_iterations
                self.optimizer.param_groups[0]["lr"] = frac * cfg.learning_rate

            next_obs, next_done = self.collect_rollouts(next_obs, next_done)
            metrics = self.update()
            metrics["charts/learning_rate"] = self.optimizer.param_groups[0]["lr"]
            metrics["charts/SPS"] = int(self.global_step / (time.time() - start_time))
            self.log_fn(metrics, self.global_step)

        return self.agent
