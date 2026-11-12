"""Gating check before running full fit on GPU: does the model rank hidden-pair Bliss, and does its std track its error?"""

import pandas as pd
import wandb
from omegaconf import OmegaConf
from scipy.stats import spearmanr

from data.splits import load_hidden_pairs
from envs.task import make_task
from src.models.mlp import MLPEnsemble


def main() -> None:
    cfg = OmegaConf.load("configs/task.yaml")
    combos = pd.read_parquet("data/processed/gdsc2_combos.parquet")
    task = make_task(combos, load_hidden_pairs(), cfg)

    with wandb.init(entity="goodarzilab", project="r-squared", job_type="gating", name="mlp-gating") as run:
        model = MLPEnsemble(cfg.response, device="cuda").fit(task.train, task.train.iloc[:0])
        mean, std = model.predict(task.pool)
        err = abs(mean - task.truth)
        run.summary.update({
            "spearman_mean_truth": spearmanr(mean, task.truth).statistic,
            "spearman_std_abs_err": spearmanr(std, err).statistic,
            "std_min": std.min(),
            "std_max": std.max(),
            "std_cv": std.std() / std.mean(),   # near 0 means std is nearly the same everywhere
        })


if __name__ == "__main__":
    main()