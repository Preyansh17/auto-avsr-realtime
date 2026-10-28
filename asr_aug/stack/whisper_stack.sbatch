#!/bin/bash
# One item-3 stack arm x seed (make_mix.py). Whisper large-v3, ENCODER-ONLY finetune (all encoder layers train,
# decoder frozen: --unfreeze-encoder-layers -1 without --train-decoder), lr 1e-5, 30 epochs, lang_task prefix,
# val-WER selection on s1_val. Evals: s1 and s2 val/test/oov, best and last checkpoint. best/ is kept for the soup.
#   sbatch --account=<A> -p l40s_public --time=<T> --export=ALL,ARM=E2,SEED=1 whisper_stack.sbatch
#SBATCH --job-name=stack_whisper
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/pa2753/stack/logs/%x_%j.out
set -euo pipefail
: "${ARM:?}" "${SEED:?}"
PY=/scratch/pa2753/envs/whisper_asr/bin/python
export HF_HOME=/scratch/pa2753/hf_cache TMPDIR=/scratch/pa2753/.tmp PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 SEED
L=/scratch/pa2753/multi_patient/labels
ROOT=/scratch/pa2753/stack/root
RUN=/scratch/pa2753/stack/runs/${ARM}_seed${SEED}
mkdir -p "$RUN"
echo "ARM=$ARM SEED=$SEED host=$(hostname) job=${SLURM_JOB_ID:-} start=$(date)"
$PY /scratch/pa2753/stack/code/make_mix.py --arm "$ARM" --seed "$SEED" --out "$RUN/train.csv"
cd /home/pa2753/auto-avsr-realtime
$PY -m asr_baselines.whisper_finetune --model openai/whisper-large-v3 --root-dir $ROOT \
  --train-file "$RUN/train.csv" --val-file $L/s1_val_spm1023.csv --output-dir "$RUN" --label-prefix lang_task \
  --epochs 30 --learning-rate 1e-5 --warmup-steps 25 --weight-decay 0.01 --batch-size 8 --grad-accum 2 \
  --precision bf16 --unfreeze-encoder-layers -1 --num-workers 4
LAST=$(ls -d "$RUN"/checkpoint-* | sort -t- -k2 -n | tail -1)
for p in s1 s2; do for s in val test oov; do
  $PY -m asr_baselines.whisper_eval --model "$RUN/best" --root-dir $ROOT --label-file $L/${p}_${s}_spm1023.csv \
    --out "$RUN/${p}_${s}_hyps.tsv" --prefix lang_task
  $PY -m asr_baselines.whisper_eval --model "$LAST" --root-dir $ROOT --label-file $L/${p}_${s}_spm1023.csv \
    --out "$RUN/${p}_${s}_hyps_last.tsv" --prefix lang_task
done; done
rm -rf "$RUN"/checkpoint-*
echo "done $(date)"
