#!/bin/bash
# G2 feature prep (g2_prep.py; asr_aug/gen/README.md amendment 3): NW workers on one GPU, worker k takes dirs
# (JOB*NW+k)::(NJOB*NW); then --merge (PARTIAL=1: only ready dirs, for the training smoke). Dirs whose wavs are
# not all present are skipped; the final run must follow all g2_cf jobs. ONLY_REAL=1 PARTIAL=1: real dirs only (smoke).
#   sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint=b200 g2_prep.sbatch
#SBATCH --job-name=g2_prep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/pa2753/gen/logs/%x_%j.out
#SBATCH --open-mode=append
set -euo pipefail
JOB=${JOB:-0}; NJOB=${NJOB:-1}; NW=${NW:-4}
G=/scratch/pa2753/gen; PY=/scratch/pa2753/envs/pdis/bin/python
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / NW > 0 ? ${SLURM_CPUS_PER_TASK:-8} / NW : 1 ))
echo "JOB=$JOB/$NJOB NW=$NW host=$(hostname) job=${SLURM_JOB_ID:-} restart=${SLURM_RESTART_COUNT:-0} $(date)"
( while sleep 300; do echo "[gpu] $(date +%H:%M) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; done ) &
MON=$!
pids=()
for k in $(seq 0 $((NW - 1))); do
  $PY $G/code/g2_prep.py --shard $((JOB * NW + k)) --nshard $((NJOB * NW)) ${ONLY_REAL:+--only-real} >> $G/logs/g2_prep_${SLURM_JOB_ID:-x}_w$k.log 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
kill $MON 2>/dev/null || true
[ $rc = 0 ]
[ $NJOB = 1 ] && $PY $G/code/g2_prep.py --merge ${PARTIAL:+--partial} ${ONLY_REAL:+--only-real}
echo "done $(date)"
