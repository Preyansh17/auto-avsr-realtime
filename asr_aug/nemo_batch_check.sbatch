#!/bin/bash
# Does batch padding change Nemotron offline decodes? (open item from the 2026-10-20 design review.) Decodes val and
# test of one checkpoint with nemotron_eval.py at batch 1 and at the default batch 8, then compares hyps line by line.
#   sbatch --account=<A> --export=ALL,MODEL=<.nemo>,MAN=<dir with {val,test}_manifest.jsonl> nemo_batch_check.sbatch
#SBATCH --job-name=nemo_batch_check
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/pa2753/nemo_lr/logs/%x_%j.out
set -euo pipefail
: "${MODEL:?}" "${MAN:?}"
PY=/scratch/pa2753/envs/nemo_asr/bin/python
JL=$(ls /usr/local/cuda/lib64/libnvJitLink.so.13 2>/dev/null || true)
[[ -n "$JL" ]] && export LD_PRELOAD="$JL${LD_PRELOAD:+:$LD_PRELOAD}"
O=/scratch/pa2753/nemo_lr/batchcheck/$(basename "$(dirname "$MODEL")")
mkdir -p "$O"
cd /home/pa2753/auto-avsr-realtime
for s in val test; do
  for b in 1 8; do
    $PY -m asr_baselines.nemotron_eval --model "$MODEL" --manifest "$MAN/${s}_manifest.jsonl" --batch-size $b \
      --out "$O/${s}_b$b.tsv" 2>&1 | grep -a "WER="
  done
  n=$(paste <(cut -f3 "$O/${s}_b1.tsv") <(cut -f3 "$O/${s}_b8.tsv") | awk -F'\t' 'NR>1 && $1!=$2' | wc -l)
  echo "BATCHCHECK $s: $n of $(($(wc -l < "$O/${s}_b1.tsv") - 1)) hyps differ between batch 1 and 8"
done
