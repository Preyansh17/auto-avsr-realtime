#!/bin/bash
# Item 8 SGEM TTA (sgem_whisper.py; asr_aug/tta/README.md). One GPU runs every (run, scope) pair of RUNS x SCOPES as
# its own process (batch-1 decoding alone leaves the GPU idle, and HPC cancels low-utilization jobs). Resumable.
#   sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint=b200 --export=ALL,RUNS="E2_seed1 E2_seed2" tta.sbatch
# Env: pdis (torch 2.8 cu128, transformers 4.46), which has B200 kernels; whisper_asr (torch 2.5) does not. Step 0 (no
# TTA) is decoded in the same env, so the comparison is within one env.
#   smoke: LIMIT=3 SETS="s1_val s2_oov" TAG=tta_smoke (writes <run>/<TAG>/<scope>)
#SBATCH --job-name=tta
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/pa2753/tta/logs/%x_%j.out
#SBATCH --open-mode=append
set -euo pipefail
: "${RUNS:?}"
SCOPES=${SCOPES:-"ln feat"}; LIMIT=${LIMIT:-0}; SETS=${SETS:-}; TAG=${TAG:-tta}
R=/scratch/pa2753/stack/runs; PY=${PY:-/scratch/pa2753/envs/pdis/bin/python}; L=/scratch/pa2753/tta/logs
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
echo "RUNS=$RUNS SCOPES=$SCOPES LIMIT=$LIMIT host=$(hostname) job=${SLURM_JOB_ID:-} restart=${SLURM_RESTART_COUNT:-0} $(date)"
cd /home/pa2753/auto-avsr-realtime
( while sleep 300; do echo "[gpu] $(date +%H:%M) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; done ) &
MON=$!
pids=()
for run in $RUNS; do
  for sc in $SCOPES; do
    $PY /scratch/pa2753/tta/code/sgem_whisper.py --model $R/$run/best --scope $sc --out-dir $R/$run/$TAG/$sc \
      --limit $LIMIT ${SETS:+--sets $SETS} >> $L/tta_${SLURM_JOB_ID:-x}_${run}_${sc}.log 2>&1 &
    pids+=($!)
  done
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
kill $MON 2>/dev/null || true
grep -h "TTA_DONE\|Error\|done .* min" $L/tta_${SLURM_JOB_ID:-x}_*.log | sort | uniq -c
[ $rc = 0 ]
echo "done $(date)"
