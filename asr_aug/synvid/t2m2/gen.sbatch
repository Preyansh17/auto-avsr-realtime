#!/bin/bash
#SBATCH --job-name=t2m2_gen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/pa2753/synvid/t2m2/logs/%x_%j.out
# Usage: sbatch --account=... --constraint="a100|h100|h200" --export=ALL,AE_STEPS=30000,GEN_STEPS=20000 gen.sbatch
# Resumable: AE skipped if ae.pth exists; generator resumes from gen_resume.pth.
set -euo pipefail
PY=/scratch/pa2753/synvid/envs/usr2/bin/python
G=/scratch/pa2753/synvid/t2m2/gen
cd /scratch/pa2753/synvid/t2m2/code
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f $G/pack.pt ] || $PY gen.py prep
[ -f $G/ae.pth ] || $PY gen.py ae --steps ${AE_STEPS:-30000} --batch 256 --seed 1
[ -f $G/gen.pth ] || $PY gen.py gen --steps ${GEN_STEPS:-20000} --batch 16 --seed 1 --lam ${LAM:-1.0}
echo DONE
