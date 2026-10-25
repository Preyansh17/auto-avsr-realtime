#!/bin/bash
# Whisper large-v3 full FT + SpecAugment (split_v1 best recipe, as asr_aug/whisper_aug.sbatch) on one patient set.
#   sbatch --account=<A> -p <part> --time=<T> --export=ALL,SET=s2,SEED=1,LABEL_PREFIX=lang_task whisper_patient.sbatch
# SET: s1 (semg1 oct1 session), s2 (semg2), pool12 (s1+s2 train). Eval on best/ for every patient the SET covers.
#SBATCH --job-name=mp_whisper
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/pa2753/multi_patient/logs/%x_%j.out
set -euo pipefail
: "${SET:?}" "${SEED:?}"
LP=${LABEL_PREFIX:-lang_task}
PY=/scratch/pa2753/envs/whisper_asr/bin/python
export HF_HOME=/scratch/pa2753/hf_cache TMPDIR=/scratch/pa2753/.tmp PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SEED
M=/scratch/pa2753/multi_patient; L=$M/labels; ROOT=$M/root
RUN=$M/runs/whisper_${SET}_${LP}_seed${SEED}
mkdir -p "$RUN"
case $SET in s1|s2) PATS=$SET ;; pool12) PATS="s1 s2" ;; esac
echo "SET=$SET SEED=$SEED LP=$LP host=$(hostname) job=${SLURM_JOB_ID:-} start=$(date)"
cd /home/pa2753/auto-avsr-realtime
$PY -m asr_baselines.whisper_finetune --model openai/whisper-large-v3 --root-dir $ROOT \
  --train-file $L/${SET}_train_spm1023.csv --val-file $L/${SET}_val_spm1023.csv --output-dir "$RUN" --label-prefix "$LP" \
  --epochs 30 --learning-rate 1e-5 --warmup-steps 25 --weight-decay 0.01 --batch-size 8 --grad-accum 2 \
  --precision bf16 --unfreeze-encoder-layers -1 --train-decoder --num-workers 4
for p in $PATS; do for s in val test oov; do
  $PY -m asr_baselines.whisper_eval --model "$RUN/best" --root-dir $ROOT --label-file $L/${p}_${s}_spm1023.csv --out "$RUN/${p}_${s}_hyps.tsv"
done; done
rm -rf "$RUN"/checkpoint-*
echo "done $(date)"
