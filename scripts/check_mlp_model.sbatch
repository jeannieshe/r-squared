#!/bin/bash
#SBATCH --job-name=r2-policies
#SBATCH --partition=gpu_batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=5G
#SBATCH --time=2:00:00
#SBATCH --output=results/%j_%x.out

cd ~/r-squared
uv run python -m scripts.check_mlp_model
