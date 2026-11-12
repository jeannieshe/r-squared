import pandas as pd
from omegaconf import OmegaConf

from data.splits import load_hidden_pairs
from envs.task import make_task
from src.models.mlp import MLPEnsemble

cfg = OmegaConf.load("configs/task.yaml")
combos = pd.read_parquet("data/processed/gdsc2_combos.parquet")
task = make_task(combos, load_hidden_pairs(), cfg)

model = MLPEnsemble("BLISS", n_models=2, epochs=1)
mean, std = model.fit(task.train, task.train.iloc[:0]).predict(task.pool)   # nothing revealed yet
print(mean.shape, std.shape, std.min(), std.max())

revealed = task.pool.iloc[:50].assign(BLISS=task.truth[:50])                 # pretend 50 were revealed
mean, std = model.fit(task.train, revealed).predict(task.pool)
print("with revealed rows: OK")