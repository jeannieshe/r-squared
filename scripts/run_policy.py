"""Run one policy on the search task for every seed and log each episode to wandb."""
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import wandb
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from envs.search_env import SearchEnv
from envs.task import Task, make_task
from src.models.mlp import MLPEnsemble
from src.policies.baselines import make_policy

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"mlp": MLPEnsemble}

def run_episode(task: Task, policy, model, response, rng: np.random.Generator) -> list[float]:
    """Spend the whole budget; return recall after each round."""
    env = SearchEnv(task)
    curve = []
    while not env.done:
        mean = std = None
        if model is not None:
            mean, std = model.fit(task.train, env.revealed(response)).predict(task.pool)

        env.step(policy.select(mean, std, ~env.observed, task.batch_size, rng))
        curve.append(env.recall())
    return curve


@hydra.main(config_path="../configs", config_name="run", version_base=None)
def main(cfg: DictConfig) -> None:
    combos = pd.read_parquet(ROOT / "data/processed/gdsc2_combos.parquet")
    hidden = yaml.safe_load((ROOT / "configs/split.yaml").read_text())["hidden_pairs"]
    task = make_task(combos, hidden, cfg)
    policy_kwargs = OmegaConf.to_container(cfg.policy, resolve=True)
    name = policy_kwargs.pop("name")
    policy = make_policy(name, **policy_kwargs)

    rows = []
    for seed in cfg.seeds:
        model_kwargs = OmegaConf.to_container(cfg.model, resolve=True)
        model_name = model_kwargs.pop("name")

        model = MODELS[model_name](cfg.response, seed=seed, **model_kwargs) if model_name else None

        with wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            group=f"{name}+{model_name}",
            name=f"{name}+{model_name}-seed{seed}",
            config={**OmegaConf.to_container(cfg, resolve=True), "seed": seed},
        ) as run:
            curve = run_episode(task, policy, model, cfg.response, np.random.default_rng(seed))
            for rnd, recall in enumerate(curve, start=1):
                run.log({"recall": recall, "n_tested": rnd * task.batch_size}, step=rnd)
                rows.append({"policy": name, "model": model_name, "seed": seed, "round": rnd, "recall": recall})
            run.summary["final_recall"] = curve[-1]

    out = Path(HydraConfig.get().runtime.output_dir)
    pd.DataFrame(rows).to_parquet(out / "results.parquet", index=False)


if __name__ == "__main__":
    main()