#!/bin/bash
# Score nemotron_en models at the 560 ms cache-aware setting (att 70,6: 480 ms look-ahead, 7-frame chunk) that teval.sh
# skips (it runs only Tianyu's two phone modes, 70,1 and 70,13). Decode only; finished sets are skipped.
#   sbatch --account=<A> --constraint="a100|l40s|h100" --export=ALL,ARMS="nemotron_en_v2r nemotron_en_v2h" cache706.sbatch
#SBATCH --job-name=cache706
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
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
E=/scratch/pa2753/unified/code/tianyu_eval.py
for A in $ARMS; do
  $PY $E --model $R/${A}_soup/soup.nemo --tag ${A}_soup --out $R/${A}_soup/teval --mode cache --att 70,6
  for S in 1 2 3; do
    $PY $E --model $R/${A}_seed$S/nemotron_best.nemo --tag ${A}_seed$S --out $R/${A}_seed$S/teval --mode cache --att 70,6
  done
done
echo "done $(date)"
