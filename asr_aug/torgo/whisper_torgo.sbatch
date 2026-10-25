#!/bin/bash
# One TORGO arm x seed, Whisper large-v3 full FT + SpecAugment with the patient best hyperparameters except 15 epochs
# (TORGO train is 5.4x the patient's). Val-WER checkpoint selection, offline greedy decode of val/test for best/ and for
# the final epoch (val-free secondary).  sbatch ... --export=ALL,ARM=M0.5,SEED=4 whisper_torgo.sbatch
#SBATCH --job-name=torgo_whisper
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/pa2753/torgo_aug/logs/%x_%j.out
set -euo pipefail
# LABEL_PREFIX: legacy (default) = every run of the 2026-10 studies; lang_task = prompt-mismatch fix (2026-10-20)
: "${ARM:?}" "${SEED:?}"
PY=/scratch/pa2753/envs/whisper_asr/bin/python
export HF_HOME=/scratch/pa2753/hf_cache TMPDIR=/scratch/pa2753/.tmp PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SEED
D=/scratch/pa2753/torgo_aug/data
ROOT=/scratch/pa2753/torgo_aug/root
RUN=/scratch/pa2753/torgo_aug/runs/whisper_${ARM}_seed${SEED}
mkdir -p "$RUN"
echo "ARM=$ARM SEED=$SEED host=$(hostname) job=${SLURM_JOB_ID:-} start=$(date)"
$PY /scratch/pa2753/torgo_aug/code/make_mix.py --arm "$ARM" --seed "$SEED" --out "$RUN/train.csv"
cd /home/pa2753/auto-avsr-realtime
$PY -m asr_baselines.whisper_finetune --model openai/whisper-large-v3 --root-dir $ROOT \
  --train-file "$RUN/train.csv" --val-file $D/val_spm1023.csv --output-dir "$RUN" --label-prefix "${LABEL_PREFIX:-legacy}" \
  --epochs 15 --learning-rate 1e-5 --warmup-steps 25 --weight-decay 0.01 --batch-size 8 --grad-accum 2 \
  --precision bf16 --unfreeze-encoder-layers -1 --train-decoder --num-workers 4
for s in val test; do
  $PY -m asr_baselines.whisper_eval --model "$RUN/best" --root-dir $ROOT --label-file $D/${s}_spm1023.csv --out "$RUN/${s}_hyps.tsv"
done
LAST=$(ls -d "$RUN"/checkpoint-* | sort -t- -k2 -n | tail -1)
echo "last checkpoint: $LAST"
for s in val test; do
  $PY -m asr_baselines.whisper_eval --model "$LAST" --root-dir $ROOT --label-file $D/${s}_spm1023.csv --out "$RUN/${s}_hyps_last.tsv"
done
rm -rf "$RUN"/checkpoint-*
echo "done $(date)"
