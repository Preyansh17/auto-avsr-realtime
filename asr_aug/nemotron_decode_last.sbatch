#!/bin/bash
# Selection-free Nemotron comparison (audit 2026-10-13): decode the final-epoch model (nemotron_patient.nemo, epoch 209 /
# 42) of every R, M0.5, S run on val and test, plus the missing val decode of best.nemo for the augmented runs. R seeds 2-3
# picked best.nemo with the pre-76157a3 SaveBestWER (best reset on every resume), so their best is not comparable.
# ARMS / SEEDS override the defaults, e.g. --export=ALL,ARMS=V1 for the re-voicing arm.
#SBATCH --job-name=aug_n_decode
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/pa2753/asr_aug/logs/%x_%j.out
set -euo pipefail
JL=$(ls /usr/local/cuda/lib64/libnvJitLink.so.13 2>/dev/null || true)
[[ -n "$JL" ]] && export LD_PRELOAD="$JL${LD_PRELOAD:+:$LD_PRELOAD}"
PY=/scratch/pa2753/envs/nemo_asr/bin/python
OUTD=/scratch/pa2753/asr_aug/runs
cd /home/pa2753/auto-avsr-realtime
dec() {  # model manifest out
  [[ -s "$3" ]] && { echo "skip $3"; return; }
  $PY -m asr_baselines.nemotron_eval --model "$1" --manifest "$2" --out "$3" 2>&1 | grep -E "WER|wrote|Error|error" || true
  [[ -s "$3" ]] || { echo "FAILED $3"; exit 1; }
}
for s in ${SEEDS:-1 2 3}; do
  for arm in ${ARMS:-R M0.5 S}; do
    if [[ $arm == R ]]; then SRC=/scratch/pa2753/experiments/nemotron_asr/nemotron_fullft_epochs210_splitv1merged_seed${s}_speedperturb
    else SRC=$OUTD/nemotron_${arm}_seed$s; fi
    D=$OUTD/nemotron_${arm}_seed$s
    for sp in val test; do
      dec "$SRC/nemotron_patient.nemo" "$SRC/${sp}_manifest.jsonl" "$D/${sp}_hyps_last.tsv"
    done
    [[ $arm != R ]] && dec "$SRC/nemotron_best.nemo" "$SRC/val_manifest.jsonl" "$D/val_hyps.tsv"
  done
done
echo done
