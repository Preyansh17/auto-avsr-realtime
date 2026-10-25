#!/bin/bash
# One old->Oct-1 semg1 arm x seed (make_mix_oct1.py). Recipe = whisper_aug.sbatch with the fixed decoder prompt
# (lang_task); val selection and all evals on the Oct 1 s1_{val,test,oov} role files.
#   sbatch --account=<A> -p l40s_public --time=<T> --export=ALL,ARM=ONS,SEED=1 whisper_oct1.sbatch
#SBATCH --job-name=oct1_whisper
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/pa2753/asr_aug/logs/%x_%j.out
set -euo pipefail
: "${ARM:?}" "${SEED:?}"
PY=/scratch/pa2753/envs/whisper_asr/bin/python
export HF_HOME=/scratch/pa2753/hf_cache TMPDIR=/scratch/pa2753/.tmp PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SEED
L=/scratch/pa2753/multi_patient/labels
ROOT=/scratch/pa2753/asr_aug/root
RUN=/scratch/pa2753/asr_aug/runs/oct1_${ARM}${FOLD:+_fold$FOLD}_seed${SEED}
mkdir -p "$RUN"
echo "ARM=$ARM SEED=$SEED host=$(hostname) job=${SLURM_JOB_ID:-} start=$(date)"
$PY /scratch/pa2753/asr_aug/code/make_mix_oct1.py --arm "$ARM" --seed "$SEED" ${FOLD:+--fold $FOLD} --out "$RUN/train.csv"
cd /home/pa2753/auto-avsr-realtime
$PY -m asr_baselines.whisper_finetune --model openai/whisper-large-v3 --root-dir $ROOT \
  --train-file "$RUN/train.csv" --val-file $L/s1_val_spm1023.csv --output-dir "$RUN" --label-prefix lang_task \
  --epochs 30 --learning-rate 1e-5 --warmup-steps 25 --weight-decay 0.01 --batch-size 8 --grad-accum 2 \
  --precision bf16 --unfreeze-encoder-layers -1 --train-decoder --num-workers 4
for s in val test oov; do
  $PY -m asr_baselines.whisper_eval --model "$RUN/best" --root-dir $ROOT --label-file $L/s1_${s}_spm1023.csv \
    --out "$RUN/${s}_hyps.tsv" --prefix lang_task
done
LAST=$(ls -d "$RUN"/checkpoint-* | sort -t- -k2 -n | tail -1)
for s in val test oov; do
  $PY -m asr_baselines.whisper_eval --model "$LAST" --root-dir $ROOT --label-file $L/s1_${s}_spm1023.csv \
    --out "$RUN/${s}_hyps_last.tsv" --prefix lang_task
done
rm -rf "$RUN"/checkpoint-*
echo "done $(date)"
