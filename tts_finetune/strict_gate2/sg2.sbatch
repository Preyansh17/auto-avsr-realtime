#!/bin/bash
# Strict gate-2 tests. Submit with the account/partition that starts soonest (hedge to 2-3, scancel losers):
#   NAME=t1 CMD="t1_t4_parity.py" sbatch --account=<acct> -p <partition> --export=ALL sg2.sbatch
#   NAME=t3 CMD="t3_conditioning.py" ...;  NAME=t2_aligned CMD="t2_overfit.py --arm aligned" ...
# Output goes to strict_gate2/out/$NAME; a lock dir makes hedge-submitted duplicates exit.
#SBATCH --job-name=higgs_sg2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/pa2753/higgs_ft/strict_gate2/logs/%x_%j.out
#SBATCH --error=/scratch/pa2753/higgs_ft/strict_gate2/logs/%x_%j.out
set -euo pipefail
export PYTHONPATH=/scratch/pa2753/higgs_site
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
: "${NAME:?}" "${CMD:?}"
cd /scratch/pa2753/higgs_ft/strict_gate2
mkdir -p logs out
if ! mkdir "logs/lock_${NAME}" 2>/dev/null; then echo "another job already holds $NAME; exiting"; exit 0; fi
echo "host=$(hostname) job=$SLURM_JOB_ID name=$NAME cmd=$CMD"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
set -- $CMD
script=$1; shift
/scratch/pa2753/envs/nemo_asr/bin/python "$script" "out/$NAME" "$@"
echo "SG2_DONE $NAME"
