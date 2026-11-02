#!/bin/bash
# Score models on Tianyu's semg2 package sets: zero-shot bases, or finished fine-tunes (PLAN.md 3.3 item 0, v2).
#   sbatch --account=<A> -p l40s_public --export=ALL,JOBS="<model>|<tag>|<base> ..." teval.sbatch
#SBATCH --job-name=teval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/pa2753/unified/logs/%x_%j.out
set -euo pipefail
: "${JOBS:?}"
C=/scratch/pa2753/unified/cache; mkdir -p $C
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor XDG_CACHE_HOME=$C/xdg NUMBA_CACHE_DIR=$C/numba
JL=/scratch/pa2753/envs/nemo31/lib/python3.12/site-packages/nvidia/cu13/lib/libnvJitLink.so.13
export LD_PRELOAD="$JL${LD_PRELOAD:+:$LD_PRELOAD}"
for J in $JOBS; do IFS='|' read -r M T B <<<"$J"; bash /scratch/pa2753/unified/code/teval.sh "$M" "$T" "$B" /scratch/pa2753/unified/teval; done
echo "done $(date)"
