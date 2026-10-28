#!/bin/bash
# Soup one arm's seed checkpoints (soup.py) and decode s1/s2 val/test/oov with it.
#   sbatch --account=<A> -p l40s_public --export=ALL,ARM=E2 soup.sbatch
#SBATCH --job-name=stack_soup
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/pa2753/stack/logs/%x_%j.out
set -euo pipefail
: "${ARM:?}"
PY=/scratch/pa2753/envs/whisper_asr/bin/python
export HF_HOME=/scratch/pa2753/hf_cache PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
L=/scratch/pa2753/multi_patient/labels
RUN=/scratch/pa2753/stack/runs/${ARM}_soup
$PY /scratch/pa2753/stack/code/soup.py --arm "$ARM" --seeds ${SEEDS:-1 2 3 4 5}
cd /home/pa2753/auto-avsr-realtime
for p in s1 s2; do for s in val test oov; do
  $PY -m asr_baselines.whisper_eval --model "$RUN/best" --root-dir /scratch/pa2753/stack/root \
    --label-file $L/${p}_${s}_spm1023.csv --out "$RUN/${p}_${s}_hyps.tsv" --prefix lang_task
done; done
echo "done $(date)"
