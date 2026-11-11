import numpy as np
import pandas as pd

from envs.task import Task


class SearchEnv:
    """Pool-based active search: each round, test a batch and see its measured response."""

    def __init__(self, task: Task):
        self.task = task
        self.reset()

    def reset(self) -> np.ndarray:
        self.observed = np.zeros(len(self.task.pool), dtype=bool)
        self.round = 0
        return self.observed.copy()

    @property
    def done(self) -> bool:
        return self.round >= self.task.rounds

    def step(self, idx: np.ndarray) -> np.ndarray:
        """Test the candidates at `idx` and return their measured values."""
        if self.done:
            raise RuntimeError("Budget spent; call reset()")
        idx = np.asarray(idx)
        if len(idx) != self.task.batch_size or len(np.unique(idx)) != len(idx):
            raise ValueError(f"Need {self.task.batch_size} distinct candidates, got {len(idx)}")
        if self.observed[idx].any():
            raise ValueError("Batch contains already-tested candidates")

        self.observed[idx] = True
        self.round += 1
        return self.task.truth[idx]

    def revealed(self, response: str) -> pd.DataFrame:
        """Tested pool rows with their measured values."""
        rows = self.task.pool[self.observed].copy()
        rows[response] = self.task.truth[self.observed]
        return rows

    def recall(self) -> float:
        """Share of targets found so far."""
        return self.task.is_target[self.observed].sum() / self.task.is_target.sum()