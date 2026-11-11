import numpy as np

from src.policies.base import Policy


class RandomPolicy(Policy):
    """Pick k untested candidates uniformly at random. Ignores the model."""
    def select(self, mean, std, available, k, rng):
        return rng.choice(np.flatnonzero(available), size=k, replace=False)