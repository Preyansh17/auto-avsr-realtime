#!/bin/bash
# Sharper gate-2 conditioning diagnostic. Account/partition on the sbatch command line.
#SBATCH --job-name=higgs_ft_diag
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/pa2753/higgs_ft/logs/%x_%j.out
#SBATCH --error=/scratch/pa2753/higgs_ft/logs/%x_%j.out
set -euo pipefail
export PYTHONPATH=/scratch/pa2753/higgs_site
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /scratch/pa2753/higgs_ft
echo "host=$(hostname) job=$SLURM_JOB_ID"
/scratch/pa2753/envs/nemo_asr/bin/python gate2_diag.py runs/gate2_diag_${SLURM_JOB_ID}
