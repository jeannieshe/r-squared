"""Baseline batch-selection policies for the active search task.

Every policy follows the `Policy` protocol in `src.policies.base`: given a model's per-candidate
predictive `mean` and `std`, the `available` mask, and a batch size `k`, return k distinct indices
of available candidates.

Build one by name with `make_policy("ucb", beta=2.0)`; `POLICIES` lists the registered names.
"""

import numpy as np

from src.policies.base import Policy


def _top_k(scores: np.ndarray, available: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest-scoring available candidates, best first."""
    idx = np.flatnonzero(available)
    if k > len(idx):
        raise ValueError(f"Asked for {k} candidates but only {len(idx)} are available")
    best = np.argpartition(-scores[idx], k - 1)[:k]
    return idx[best[np.argsort(-scores[idx][best])]]


def _require(name: str, **arrays: np.ndarray | None) -> None:
    missing = [key for key, value in arrays.items() if value is None]
    if missing:
        raise ValueError(f"{name} needs model predictions ({', '.join(missing)}) but got None")


class RandomPolicy:
    """Pick k available candidates uniformly at random. Ignores the model."""

    def select(self, mean, std, available, k, rng):
        return rng.choice(np.flatnonzero(available), size=k, replace=False)


class GreedyPolicy:
    """Pick the k available candidates with the highest predicted mean (pure exploitation)."""

    def select(self, mean, std, available, k, rng):
        _require("GreedyPolicy", mean=mean)
        return _top_k(mean, available, k)


class UCBPolicy:
    """Pick the k available candidates with the highest upper confidence bound, mean + beta * std.

    beta = 0 reduces to greedy; larger beta favours candidates the model is unsure about.
    """

    def __init__(self, beta: float = 2.0):
        if beta < 0:
            raise ValueError(f"beta must be >= 0, got {beta}")
        self.beta = beta

    def select(self, mean, std, available, k, rng):
        _require("UCBPolicy", mean=mean, std=std)
        return _top_k(mean + self.beta * std, available, k)


class ThompsonPolicy:
    """Draw one sample per candidate from N(mean, std^2) and pick the k available with the highest draws.

    This treats candidates as independent Gaussians, so it ignores correlations between them.
    """

    def select(self, mean, std, available, k, rng):
        _require("ThompsonPolicy", mean=mean, std=std)
        return _top_k(rng.normal(mean, std), available, k)


POLICIES: dict[str, type] = {
    "random": RandomPolicy,
    "greedy": GreedyPolicy,
    "ucb": UCBPolicy,
    "thompson": ThompsonPolicy,
}


def make_policy(name: str, **kwargs) -> Policy:
    """Build a registered policy by name, e.g. make_policy("ucb", beta=2.0)."""
    if name not in POLICIES:
        raise ValueError(f"Unknown policy {name!r}; choose from {sorted(POLICIES)}")
    return POLICIES[name](**kwargs)
