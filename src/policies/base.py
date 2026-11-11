from typing import Protocol

import numpy as np


class Policy(Protocol):
    def select(
            self,
            mean: np.ndarray | None,
            std: np.ndarray | None,
            available: np.ndarray,
            k: int,
            rng: np.random.Generator,
    ) -> np.ndarray:
        """Return k distinct indices into the pool, all where `available` is True."""
