#!/bin/bash
# Runs B_r64 and C_r16_lr5e-5 (plan Phase 2) as two concurrent processes on ONE GPU. Submit with the account/partition
# that starts soonest, e.g.
#   GIT_HASH=<hash> sbatch --account=<acct> -p <partition> --export=ALL train_packed.sbatch
# Per-run logs: logs/train_<RUN>_<jobid>.out; GPU utilisation / memory sampled to logs/gpu_util_<jobid>.csv.
# Resume after a timeout: EXTRA_B="--resume" EXTRA_C="--resume" (same command).
#SBATCH --job-name=higgs_ft_packed
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/pa2753/higgs_ft/logs/%x_%j.out
#SBATCH --error=/scratch/pa2753/higgs_ft/logs/%x_%j.out
set -eo pipefail
export PYTHONPATH=/scratch/pa2753/higgs_site
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
cd /scratch/pa2753/higgs_ft
mkdir -p logs
RUN_B=B_r64
RUN_C=C_r16_lr5e-5
# hedge-submitted copies must not both train: take both locks or neither
GOT=""
for r in "$RUN_B" "$RUN_C"; do
  if mkdir "logs/lock_$r" 2>/dev/null; then GOT="$GOT $r"; else
    echo "another job already holds run $r; exiting"
    for g in $GOT; do rmdir "logs/lock_$g"; done
    exit 0
  fi
done
cleanup() { for g in $GOT; do rmdir "logs/lock_$g" 2>/dev/null || true; done; kill $(jobs -p) 2>/dev/null || true; }
trap cleanup EXIT
echo "host=$(hostname) job=$SLURM_JOB_ID"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 15 > "logs/gpu_util_${SLURM_JOB_ID}.csv" &
PY=/scratch/pa2753/envs/nemo_asr/bin/python
$PY train_lora.py --run "$RUN_B" --r 64 --alpha 128 ${EXTRA_B:-} > "logs/train_${RUN_B}_${SLURM_JOB_ID}.out" 2>&1 &
PID_B=$!
sleep 10
$PY train_lora.py --run "$RUN_C" --lr 5e-5 ${EXTRA_C:-} > "logs/train_${RUN_C}_${SLURM_JOB_ID}.out" 2>&1 &
PID_C=$!
rc_b=0; rc_c=0
wait $PID_B || rc_b=$?
wait $PID_C || rc_c=$?
echo "B rc=$rc_b C rc=$rc_c"
[ "$rc_b" -eq 0 ] && [ "$rc_c" -eq 0 ]
