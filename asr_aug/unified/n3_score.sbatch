#!/bin/bash
# PLAN.md 3.3 item 0, Step 6: soup N3 (3 seeds) and S9 (N2r 1-3 + N2h 1-3 + N3 1-3), then score the soups and the N3
# seeds on Tianyu's sets: offline + cache-aware 70,1 / 70,6 / 70,13. Decode only; finished sets are skipped.
#   sbatch --account=<A> --constraint="a100|l40s|h100" --dependency=afterok:<n3 job> n3_score.sbatch
#SBATCH --job-name=n3_score
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/pa2753/unified/logs/%x_%j.out
#SBATCH --open-mode=append
set -euo pipefail
PY=/scratch/pa2753/envs/nemo31/bin/python
C=/scratch/pa2753/unified/cache; mkdir -p $C
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
  TRITON_CACHE_DIR=$C/triton TORCHINDUCTOR_CACHE_DIR=$C/inductor XDG_CACHE_HOME=$C/xdg NUMBA_CACHE_DIR=$C/numba
JL=/scratch/pa2753/envs/nemo31/lib/python3.12/site-packages/nvidia/cu13/lib/libnvJitLink.so.13
export LD_PRELOAD="$JL${LD_PRELOAD:+:$LD_PRELOAD}"
R=/scratch/pa2753/unified/ft/runs
N3=/scratch/pa2753/n3/runs/nem_syn_2b_sc_o189_n189_tts50_seed
CODE=/scratch/pa2753/unified/code
score() {  # <model> <tag> <out>
  bash $CODE/teval.sh $1 $2 nemotron_en $3  # offline, 70,1, 70,13
  $PY $CODE/tianyu_eval.py --model $1 --tag $2 --out $3 --mode cache --att 70,6
}
O=$R/nemotron_en_n3_soup; mkdir -p $O
[ -s $O/soup.nemo ] || $PY $CODE/soup.py --out $O/soup.nemo ${N3}{7,1,42}/upstream_output/nemotron_best.nemo
score $O/soup.nemo nemotron_en_n3_soup $O/teval
O9=$R/nemotron_en_s9_soup; mkdir -p $O9
[ -s $O9/soup.nemo ] || $PY $CODE/soup.py --out $O9/soup.nemo $R/nemotron_en_v2{r,h}_seed{1,2,3}/nemotron_best.nemo \
  ${N3}{7,1,42}/upstream_output/nemotron_best.nemo
score $O9/soup.nemo nemotron_en_s9_soup $O9/teval
for S in 7 1 42; do score ${N3}$S/upstream_output/nemotron_best.nemo nemotron_en_n3_seed$S $R/nemotron_en_n3_seed$S/teval; done
echo "done $(date)"
