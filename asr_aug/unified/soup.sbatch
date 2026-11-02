#!/bin/bash
# Uniform 3-seed soups of finished v2 arms, then Tianyu-set scoring with teval.sh (PLAN.md 3.3 item 0, v3 arm S).
#   sbatch --account=<A> --constraint=h100 --export=ALL,ARMS="nemotron_en_v2r nemotron_en_v2h unified_v2r unified_v2h" soup.sbatch
#SBATCH --job-name=soup
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/pa2753/unified/logs/%x_%j.out
#SBATCH --open-mode=append
set -euo pipefail
: "${ARMS:?}"
PY=/scratch/pa2753/envs/nemo31/bin/python
C=/scratch/pa2753/unified/cache; mkdir -p $C
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor XDG_CACHE_HOME=$C/xdg NUMBA_CACHE_DIR=$C/numba
JL=/scratch/pa2753/envs/nemo31/lib/python3.12/site-packages/nvidia/cu13/lib/libnvJitLink.so.13
export LD_PRELOAD="$JL${LD_PRELOAD:+:$LD_PRELOAD}"
R=/scratch/pa2753/unified/ft/runs
for A in $ARMS; do
  OUT=$R/${A}_soup; mkdir -p $OUT
  [ -s $OUT/soup.nemo ] || $PY /scratch/pa2753/unified/code/soup.py --out $OUT/soup.nemo $R/${A}_seed{1,2,3}/nemotron_best.nemo
  bash /scratch/pa2753/unified/code/teval.sh $OUT/soup.nemo ${A}_soup ${A%%_v2*} $OUT/teval  # finished modes are skipped
  # SEEDS_TOO=1: also score the ingredients in any mode added to teval.sh after they finished (e.g. C0.08 R0)
  [ "${SEEDS_TOO:-0}" = 1 ] && for S in 1 2 3; do
    bash /scratch/pa2753/unified/code/teval.sh $R/${A}_seed$S/nemotron_best.nemo ${A}_seed$S ${A%%_v2*} $R/${A}_seed$S/teval; done
done
echo "done $(date)"
