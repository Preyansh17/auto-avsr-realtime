#!/bin/bash
# G2 generation (gen_g2.py; asr_aug/gen/README.md amendment 3) for one voice, NW sharded workers on one GPU.
#   sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint=b200 --export=ALL,VOICE=old|s1|s2 g2_gen.sbatch
#   LIMIT=20 OUT=/scratch/pa2753/gen/root/g2_smoke for a smoke. Resumes per clip.
#SBATCH --job-name=g2_gen
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/pa2753/gen/logs/%x_%j.out
#SBATCH --open-mode=append
set -euo pipefail
: "${VOICE:?}"
G=/scratch/pa2753/gen; PY=/scratch/pa2753/envs/pdis/bin/python; NW=${NW:-4}; LIMIT=${LIMIT:-0}; OUT=${OUT:-$G/root/g2}
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=$(( ${SLURM_CPUS_PER_TASK:-8} / NW > 0 ? ${SLURM_CPUS_PER_TASK:-8} / NW : 1 ))
echo "VOICE=$VOICE NW=$NW out=$OUT host=$(hostname) job=${SLURM_JOB_ID:-} restart=${SLURM_RESTART_COUNT:-0} $(date)"
( while sleep 300; do echo "[gpu] $(date +%H:%M) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; done ) &
MON=$!
pids=()
for k in $(seq 0 $((NW - 1))); do
  $PY $G/code/gen_g2.py --voice $VOICE --run $G/g2/run --out-root $OUT --limit $LIMIT --shard $k --nshard $NW \
    >> $G/logs/g2_gen_${SLURM_JOB_ID:-x}_w$k.log 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
kill $MON 2>/dev/null || true
[ $rc = 0 ]
echo "done $(date)"
