#!/bin/bash
#SBATCH --job-name=sv_t2m_build
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/pa2753/synvid/logs/%x_%j.out
# A11 text-to-mouth step 1: unit selection (CPU). Headless OpenCV env (cs nodes have no libGL).
# sbatch --account=torch_pr_39_general --partition=cs t2m_build.sbatch
set -eo pipefail
SV=/scratch/pa2753/synvid
$SV/envs/devcrop/bin/python $SV/code/t2m_build.py --out $SV/t2m
