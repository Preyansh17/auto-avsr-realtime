#!/bin/bash
# PLAN.md 3.3 item 0, Step 5 (decode only, nemotron_en). ARM=S6: 6-seed soup (N2r 1-3 + N2h 1-3), scored offline and
# cache-aware at 70,1 / 70,6 / 70,13. ARM=PB: phrase boosting at 70,1 on the N2r soup and seeds, lists x alphas.
#   sbatch --account=<A> --constraint="a100|l40s|h100" --export=ALL,ARM=S6 step5.sbatch
#SBATCH --job-name=step5
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/pa2753/unified/logs/%x_%j.out
#SBATCH --open-mode=append
set -euo pipefail
: "${ARM:?}"
PY=/scratch/pa2753/envs/nemo31/bin/python
C=/scratch/pa2753/unified/cache; mkdir -p $C
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor XDG_CACHE_HOME=$C/xdg NUMBA_CACHE_DIR=$C/numba
JL=/scratch/pa2753/envs/nemo31/lib/python3.12/site-packages/nvidia/cu13/lib/libnvJitLink.so.13
export LD_PRELOAD="$JL${LD_PRELOAD:+:$LD_PRELOAD}"
R=/scratch/pa2753/unified/ft/runs
CODE=/scratch/pa2753/unified/code
E=$CODE/tianyu_eval.py
if [ "$ARM" = S6 ]; then
  OUT=$R/nemotron_en_v2rh_soup6; mkdir -p $OUT
  [ -s $OUT/soup.nemo ] || $PY $CODE/soup.py --out $OUT/soup.nemo $R/nemotron_en_v2{r,h}_seed{1,2,3}/nemotron_best.nemo
  bash $CODE/teval.sh $OUT/soup.nemo nemotron_en_v2rh_soup6 nemotron_en $OUT/teval  # offline, 70,1, 70,13
  $PY $E --model $OUT/soup.nemo --tag nemotron_en_v2rh_soup6 --out $OUT/teval --mode cache --att 70,6
elif [ "$ARM" = PB ]; then
  L=/scratch/pa2753/unified/boost; mkdir -p $L
  [ -s $L/words.txt ] || $PY $CODE/make_boost_lists.py --train /scratch/pa2753/unified/ft/manifests_v2r_check/train.json --out $L
  for M in soup seed1 seed2 seed3; do
    [ $M = soup ] && N=$R/nemotron_en_v2r_soup/soup.nemo || N=$R/nemotron_en_v2r_$M/nemotron_best.nemo
    for B in clean words; do for AL in 0.5 1 2 4; do
      $PY $E --model $N --tag nemotron_en_v2r_$M --out $R/nemotron_en_v2r_$M/teval --mode cache --att 70,1 \
        --boost $L/$B.txt --boost-alpha $AL
    done; done
  done
fi
echo "done $(date)"
