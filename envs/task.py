from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

OUTCOMES = ["BLISS", "HSA", "COMBO_MaxE", "MaxE_STD", "N_REPLICATES"]

@dataclass(frozen=True)
class Task:
    pool: pd.DataFrame # hidden-pair combinations: CELL_LINE, PAIR, DRUG1, DRUG2, features
    truth: np.ndarray # mean Bliss excess score per candidate (or however set by the config)
    is_target: np.ndarray # boolean mask of the ideal combinations
    train: pd.DataFrame # all non-hidden combinations
    batch_size: int
    rounds: int

def top_frac_mask(
    pool: pd.DataFrame,
    truth: np.ndarray,
    frac: float, # must be in (0, 1]
    scope: Literal["global", "per_pair"], # when "per_pair", it asks which cell lines respond best to this pair. when "global", it asks which (cell line, drug pair) has the highest synergies.
) -> np.ndarray:
    """Mark the top `frac` of candidates by truth, over the whole pool or within each pair."""
    if not 0 < frac <= 1:
        raise ValueError(f"{frac} must be in (0, 1]")
    if scope == "global":
        groups = pd.Series(0, index=pool.index)
    elif scope == "per_pair":
        groups = pool["PAIR"]
    else:
        raise ValueError(f"Unknown scope: {scope!r}")

    ranks = pd.Series(truth, index=pool.index).groupby(groups).rank(method="first", ascending=False)
    k = np.ceil(frac * groups.map(groups.value_counts()))
    return (ranks <= k).to_numpy()

def make_task(combos: pd.DataFrame, hidden_pairs: list[str], cfg) -> Task:
    # Keep only test set pairs in the pool
    test = combos["PAIR"].isin(hidden_pairs)
    pool = combos[test].reset_index(drop=True)
    truth = pool[cfg.response].to_numpy()

    return Task(
        pool=pool.drop(columns=OUTCOMES),
        truth=truth,
        is_target=top_frac_mask(pool, truth, cfg.target.frac, cfg.target.scope),
        train=combos[~test].reset_index(drop=True),
        batch_size=cfg.budget.batch_size,
        rounds=cfg.budget.rounds,
    )