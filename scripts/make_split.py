"""Choose the hidden test pairs and write configs/split.yaml."""
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
N_HIDDEN = 20 # Size of the test suite
SEED = 0 # Set as 0 for reproducibility

def choose_hidden(pairs: list[str], n: int, seed: int) -> list[str]:
    """Hide n pairs, keeping every drug in at least one training pair."""
    uses = Counter(drug for p in pairs for drug in p.split(" + "))
    hidden = []
    for pair in np.random.default_rng(seed).permutation(sorted(pairs)):
        drugs = pair.split(" + ")
        if all(uses[d] > 1 for d in drugs):
            hidden.append(str(pair))
            uses.subtract(drugs)
        if len(hidden) == n:
            return sorted(hidden)
    raise ValueError(f"Only {len(hidden)} pairs can be hidden under the constraint")


def main() -> None:
    combos = pd.read_parquet(ROOT / "data/processed/gdsc2_combos.parquet")
    hidden = choose_hidden(combos["PAIR"].unique().tolist(), N_HIDDEN, SEED)
    split = {
        "seed": SEED,
        "created": date.today().isoformat(),
        "n_pairs_total": int(combos["PAIR"].nunique()),
        "hidden_pairs": hidden,
    }
    with open(ROOT / "configs/split.yaml", "w") as f:
        yaml.safe_dump(split, f, sort_keys=False)


if __name__ == "__main__":
    main()