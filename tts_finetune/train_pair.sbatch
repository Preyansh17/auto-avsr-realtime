#!/bin/bash
# Two (or one) LoRA runs as concurrent processes on ONE GPU (cross-fitting folds, experiment K). Submit e.g.
#   RUN1=A_r16_cv0 ARGS1="--splits data/cv/splits_cv0.json --stop_after 12" \
#   RUN2=B_r64_cv0 ARGS2="--r 64 --alpha 128 --splits data/cv/splits_cv0.json --stop_after 8" \
#   GIT_HASH=<h> sbatch --account=<a> -p <p> --export=ALL train_pair.sbatch
# Do not pair two r64 runs on an L40S (46 GiB). Per-run logs: logs/train_<RUN>_<jobid>.out; GPU samples: logs/gpu_util_<jobid>.csv.
# Resume after a timeout: add --resume to ARGS1 / ARGS2.
#SBATCH --job-name=higgs_ft_pair
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/pa2753/higgs_ft/logs/%x_%j.out
#SBATCH --error=/scratch/pa2753/higgs_ft/logs/%x_%j.out
set -eo pipefail
export PYTHONPATH=/scratch/pa2753/higgs_site
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
: "${RUN1:?set RUN1}" "${ARGS1:?set ARGS1}"
cd /scratch/pa2753/higgs_ft
mkdir -p logs
RUNS="$RUN1"; [ -n "${RUN2:-}" ] && RUNS="$RUN1 $RUN2"
GOT=""
for r in $RUNS; do   # hedge-submitted copies must not both train: take every lock or none
  if mkdir "logs/lock_$r" 2>/dev/null; then GOT="$GOT $r"; else
    echo "another job already holds run $r; exiting"; for g in $GOT; do rmdir "logs/lock_$g"; done; exit 0
  fi
done
cleanup() { for g in $GOT; do rmdir "logs/lock_$g" 2>/dev/null || true; done; kill $(jobs -p) 2>/dev/null || true; }
trap cleanup EXIT
echo "host=$(hostname) job=$SLURM_JOB_ID runs=$RUNS"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv -l 15 > "logs/gpu_util_${SLURM_JOB_ID}.csv" &
PY=/scratch/pa2753/envs/nemo_asr/bin/python
$PY train_lora.py --run "$RUN1" $ARGS1 > "logs/train_${RUN1}_${SLURM_JOB_ID}.out" 2>&1 &
PID1=$!
PID2=""
if [ -n "${RUN2:-}" ]; then
  sleep 10
  $PY train_lora.py --run "$RUN2" ${ARGS2:-} > "logs/train_${RUN2}_${SLURM_JOB_ID}.out" 2>&1 &
  PID2=$!
fi
rc1=0; rc2=0
wait $PID1 || rc1=$?
[ -n "$PID2" ] && { wait $PID2 || rc2=$?; }
echo "rc1=$rc1 rc2=$rc2"
[ "$rc1" -eq 0 ] && [ "$rc2" -eq 0 ]
