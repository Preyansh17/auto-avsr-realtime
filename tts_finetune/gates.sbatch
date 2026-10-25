#!/bin/bash
# Phase 0 gates + Phase 1 tokenization/refs, one GPU job. Account is passed on the sbatch command line
# (see README: pick the account whose GPU starts soonest).
#SBATCH --job-name=higgs_ft_gates
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/pa2753/higgs_ft/logs/%x_%j.out
#SBATCH --error=/scratch/pa2753/higgs_ft/logs/%x_%j.out
set -euo pipefail
export PYTHONPATH=/scratch/pa2753/higgs_site
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
PY=/scratch/pa2753/envs/nemo_asr/bin/python
cd /scratch/pa2753/higgs_ft
echo "host=$(hostname) job=$SLURM_JOB_ID"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY prep_data.py tokenize
$PY prep_data.py refs
$PY phase0_gates.py --out runs/gates_${SLURM_JOB_ID}
