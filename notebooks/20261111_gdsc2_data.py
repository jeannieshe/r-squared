import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    print(mo.notebook_dir())

    raw_dir = mo.notebook_dir().parent / "data" / "raw"
    return mo, pd, raw_dir


@app.cell
def _(pd, raw_dir):
    screens = ["gdsc-007", "gdsc-008", "gdsc-009", "gdsc-010", "sandpiper-01"]
    df = pd.concat(
        [pd.read_csv(raw_dir / f"{s}_matrix_results.csv.gz").assign(screen=s) for s in screens],
        ignore_index=True,
    )
    df.shape
    return (df,)


@app.cell
def _(df):
    df # merges together all of the data from the main GDSC2 screen
    # for metadata information on what each of the columns reference, look at:
    # https://gdsc-combinations.depmap.sanger.ac.uk/downloads/Matrix_glossary.pdf

    # assay/batch relevant: BARCODE
    # cell line relevant: CELL_LINE_NAME, TISSUE, CANCER_TYPE
    # drug relevant (lib1 = drug1, lib2 = drug2): lib1_name, lib1_target, lib1_pathway, lib1_MaxE, lib1_IC50_ln
    # combo relevant: cmatrix, combo_MaxE, Bliss_matrix, HSA_matrix
    return


@app.cell
def _(df):
    df.columns
    return


@app.cell
def _(df):
    print(f"Number of unique cell lines: {df["CELL_LINE_NAME"].nunique()}")
    return


@app.cell
def _(df, pd):
    drugs = pd.concat([df["lib1_name"], df["lib2_name"]])
    print(f"Number of unique single drugs: {drugs.nunique()}")
    return


@app.cell
def _(df):
    import numpy as np

    names = np.sort(df[["lib1_name", "lib2_name"]].to_numpy(), axis=1)
    df["pair"] = names[:, 0] + " + " + names[:, 1]

    df["pair"].nunique()                     # 109
    df.groupby("screen")["pair"].nunique()   # 25, 28, 28, 27, 2
    return


@app.cell
def _(df):
    print(f"Number of unique drug combinations: {df["pair"].nunique()}")
    return


@app.cell
def _(mo):
    import subprocess
    processed_dir = mo.notebook_dir().parent / "data" / "processed"
    subprocess.run(['ls'])
    return


if __name__ == "__main__":
    app.run()
