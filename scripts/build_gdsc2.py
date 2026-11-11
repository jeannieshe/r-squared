"""Build the processed GDSC2 dataset into data/processed"""

from pathlib import Path

import pandas as pd

from data.gdsc import aggregate_replicates, load_gdsc2

DATA = Path(__file__).resolve().parents[1] / "data"

EXPECTED = {"cell_lines": 755, "pairs": 109, "drugs": 37, "replicates": 84_055, "combos": 68_816}


def check(replicates: pd.DataFrame, combos: pd.DataFrame) -> None:
    """Raise if the processed tables don't match the known GDSC2 shape."""
    found = {
        "cell_lines": replicates["CELL_LINE"].nunique(),
        "pairs": replicates["PAIR"].nunique(),
        "drugs": pd.concat([replicates["DRUG1"], replicates["DRUG2"]]).nunique(),
        "replicates": len(replicates),
        "combos": len(combos),
    }
    wrong = {k: (found[k], v) for k, v in EXPECTED.items() if found[k] != v}
    if wrong:
        raise ValueError(f"GDSC2 shape mismatch (found, expected): {wrong}")


def main() -> None:
    out = DATA / "processed"
    out.mkdir(parents=True, exist_ok=True)

    replicates = load_gdsc2(DATA / "raw")
    replicates.to_parquet(out / "gdsc2_replicates.parquet", index=False)
    aggregate_replicates(replicates).to_parquet(out / "gdsc2_combos.parquet", index=False)

    check(
        pd.read_parquet(out / "gdsc2_replicates.parquet"),
        pd.read_parquet(out / "gdsc2_combos.parquet"),
    )
    print(f"GDSC2 tables OK: {EXPECTED}")


if __name__ == "__main__":
    main()
