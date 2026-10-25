#!/bin/bash
# One augmentation arm x seed with the split_v1 best Whisper recipe (slurm/whisper_splitv1_best.sbatch step 1-2:
# large-v3 full FT + SpecAugment, lr 1e-5, warmup 25, wd 0.01, 30 epochs, val-WER checkpoint selection), offline only.
#   sbatch --account=<A> -p l40s_public --time=<T> --export=ALL,ARM=M1,SEED=1 whisper_aug.sbatch
#SBATCH --job-name=aug_whisper
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/pa2753/asr_aug/logs/%x_%j.out
set -euo pipefail
# LABEL_PREFIX: legacy (default) = every run of the 2026-10 studies; lang_task = prompt-mismatch fix (2026-10-20)
: "${ARM:?}" "${SEED:?}"
PY=/scratch/pa2753/envs/whisper_asr/bin/python
export HF_HOME=/scratch/pa2753/hf_cache TMPDIR=/scratch/pa2753/.tmp PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SEED
L=/scratch/pa2753/avsr_realtime/labels/splitv1/merged
ROOT=/scratch/pa2753/asr_aug/root
RUN=/scratch/pa2753/asr_aug/runs/whisper_${ARM}${RUN_TAG:+_$RUN_TAG}_seed${SEED}   # RUN_TAG: e.g. lt for LABEL_PREFIX=lang_task reruns
mkdir -p "$RUN"
echo "ARM=$ARM SEED=$SEED host=$(hostname) job=${SLURM_JOB_ID:-} start=$(date)"
$PY /scratch/pa2753/asr_aug/code/make_mix.py --arm "$ARM" --seed "$SEED" --out "$RUN/train.csv"
cd /home/pa2753/auto-avsr-realtime
$PY -m asr_baselines.whisper_finetune --model openai/whisper-large-v3 --root-dir $ROOT \
  --train-file "$RUN/train.csv" --val-file $L/val_spm1023.csv --output-dir "$RUN" --label-prefix "${LABEL_PREFIX:-legacy}" \
  --epochs 30 --learning-rate 1e-5 --warmup-steps 25 --weight-decay 0.01 --batch-size 8 --grad-accum 2 \
  --precision bf16 --unfreeze-encoder-layers -1 --train-decoder --num-workers 4
for s in val test; do
  $PY -m asr_baselines.whisper_eval --model "$RUN/best" --root-dir $ROOT --label-file $L/${s}_spm1023.csv \
    --out "$RUN/${s}_hyps.tsv"
done
# val-free secondary (README, 2026-10-13): the final-epoch checkpoint, chosen by no val signal
LAST=$(ls -d "$RUN"/checkpoint-* | sort -t- -k2 -n | tail -1)
echo "last checkpoint: $LAST"
for s in val test; do
  $PY -m asr_baselines.whisper_eval --model "$LAST" --root-dir $ROOT --label-file $L/${s}_spm1023.csv \
    --out "$RUN/${s}_hyps_last.tsv"
done
rm -rf "$RUN"/checkpoint-*   # ~20 GB/run of optimizer state; best/ is kept
echo "done $(date)"
