"""Load GDSC2 drug-combination matrix screens into one tidy table."""

from pathlib import Path

import pandas as pd

SCREENS = ("gdsc-007", "gdsc-008", "gdsc-009", "gdsc-010", "sandpiper-01")

COLUMNS = {
    "BARCODE": "PLATE",
    "CELL_LINE_NAME": "CELL_LINE",
    "TISSUE": "TISSUE",
    "CANCER_TYPE": "CANCER_TYPE",
    "lib1_name": "DRUG1",
    "lib1_target": "DRUG1_PROTEIN_TARGET",
    "lib1_pathway": "DRUG1_PATHWAY",
    "lib1_MaxE": "DRUG1_MaxE",
    "lib1_IC50_ln": "DRUG1_IC50_ln",
    "lib2_name": "DRUG2",
    "lib2_target": "DRUG2_PROTEIN_TARGET",
    "lib2_pathway": "DRUG2_PATHWAY",
    "lib2_MaxE": "DRUG2_MaxE",
    "lib2_IC50_ln": "DRUG2_IC50_ln",
    "combo_MaxE": "COMBO_MaxE",
    "Bliss_matrix": "BLISS",
    "HSA_matrix": "HSA",
}


def load_gdsc2(raw_dir: Path) -> pd.DataFrame:
    """One row per measured (cell line, drug pair) replicate."""
    frames = [
        pd.read_csv(raw_dir / f"{s}_matrix_results.csv.gz", usecols=list(COLUMNS)).assign(SCREEN=s) for s in SCREENS
    ]

    df = pd.concat(frames, ignore_index=True).rename(columns=COLUMNS)
    # Swap names to be alphabetical within the pair
    swap = df["DRUG1"] > df["DRUG2"]
    cols1 = [c for c in df.columns if c.startswith("DRUG1")]
    cols2 = [c.replace("DRUG1", "DRUG2") for c in cols1]
    df.loc[swap, cols1 + cols2] = df.loc[swap, cols2 + cols1].to_numpy()

    df["PAIR"] = df["DRUG1"] + " + " + df["DRUG2"]

    return df


def aggregate_replicates(df: pd.DataFrame) -> pd.DataFrame:
    """Mean response per (cell line, pair), with replicate count and spread."""
    return df.groupby(["CELL_LINE", "PAIR", "TISSUE", "DRUG1", "DRUG2"], as_index=False).agg(
        COMBO_MaxE=("COMBO_MaxE", "mean"),
        BLISS=("BLISS", "mean"),
        HSA=("HSA", "mean"),
        N_REPLICATES=("COMBO_MaxE", "size"),
        MaxE_STD=("COMBO_MaxE", "std"),
    )
