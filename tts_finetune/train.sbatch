#!/bin/bash
# LoRA finetune of Higgs v2 on patient bingge. Submit with the account/partition that starts soonest, e.g.
#   RUN=A_r16 EXTRA="" GIT_HASH=$(git rev-parse HEAD) sbatch --account=<acct> -p <partition> --export=ALL train.sbatch
# Resume after a timeout: same command with EXTRA="--resume".
#SBATCH --job-name=higgs_ft_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/pa2753/higgs_ft/logs/%x_%j.out
#SBATCH --error=/scratch/pa2753/higgs_ft/logs/%x_%j.out
set -euo pipefail
export PYTHONPATH=/scratch/pa2753/higgs_site
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
: "${RUN:?set RUN=<run name>}"
cd /scratch/pa2753/higgs_ft
# hedge-submitted copies of one run (several accounts/partitions) must not both train: first job wins the lock
mkdir -p logs
if ! mkdir "logs/lock_${RUN}" 2>/dev/null; then echo "another job already holds run $RUN; exiting"; exit 0; fi
trap 'rmdir "logs/lock_${RUN}"' EXIT
echo "host=$(hostname) job=$SLURM_JOB_ID run=$RUN extra=${EXTRA:-}"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/scratch/pa2753/envs/nemo_asr/bin/python train_lora.py --run "$RUN" ${EXTRA:-}
