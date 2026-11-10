## Overview
This codebase implements RL combination perturbation discovery that uses a learned world model (such as ReJam) in order to find optimal combo perturbations for a given transcriptome.

## Repository Structure
r-squared/
├── configs/                  # YAML/Hydra configuration files for hyperparams
│   ├── default.yaml
│   └── algos/
│       ├── ppo.yaml
│       └── sac.yaml
├── envs/                     # Custom Gym/Gymnasium environments or wrappers
│   ├── __init__.py
│   └── custom_env.py
├── src/
    ├── data/                 
│   │   ├── gdsc.py           # GDSC^2 dataset
│   │   └── splits.py         # frozen hidden-pair split
│   ├── agents/               # Policy, value networks, or custom agent classes
│   │   ├── __init__.py
│   │   └── ppo_agent.py
│   ├── algorithms/           # Core RL training loops (or clean/modular updates)
│   │   ├── __init__.py
│   │   └── ppo.py
│   └── utils/                # Replay buffers, normalization, logging hooks
│       ├── buffer.py
│       └── logger.py
├── scripts/                  # Entry points
│   ├── train.py
│   └── evaluate.py
├── experiments/              # Local experiment checkpoints (git-ignored)
├── results/                  # Local artifacts (git-ignored)
├── requirements.txt or pyproject.toml
└── README.md