#!/bin/bash
# G2 Seed-VC counterfactuals (g2_cf.py; asr_aug/gen/README.md amendment 3). Job JOB of NJOB runs NW workers on one GPU;
# worker k converts global shard JOB*NW+k of NJOB*NW. Resumable per clip.
#   sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint=b200 --export=ALL,JOB=0,NJOB=4 g2_cf.sbatch
#   MAXI=25 for a smoke (each worker converts at most MAXI clips).
#SBATCH --job-name=g2_cf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/pa2753/gen/logs/%x_%j.out
#SBATCH --open-mode=append
set -euo pipefail
JOB=${JOB:-0}; NJOB=${NJOB:-1}; NW=${NW:-4}; MAXI=${MAXI:-0}
G=/scratch/pa2753/gen; PY=/scratch/pa2753/envs/pdis/bin/python
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / NW > 0 ? ${SLURM_CPUS_PER_TASK:-8} / NW : 1 ))
echo "JOB=$JOB/$NJOB NW=$NW host=$(hostname) job=${SLURM_JOB_ID:-} restart=${SLURM_RESTART_COUNT:-0} $(date)"
cd $G/code/Controllable-Dysarthric-Speech-Synthesis/third_party/seed-vc
( while sleep 300; do echo "[gpu] $(date +%H:%M) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; done ) &
MON=$!
pids=()
for k in $(seq 0 $((NW - 1))); do
  $PY $G/code/g2_cf.py --shard $((JOB * NW + k)) --nshard $((NJOB * NW)) --max-items $MAXI >> $G/logs/g2_cf_${SLURM_JOB_ID:-x}_w$k.log 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
kill $MON 2>/dev/null || true
grep -h "GEN_DONE\|s/clip" $G/logs/g2_cf_${SLURM_JOB_ID:-x}_w*.log | tail -8
[ $rc = 0 ]
echo "done $(date)"
