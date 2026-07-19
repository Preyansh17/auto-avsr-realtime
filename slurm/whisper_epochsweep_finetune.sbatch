#!/bin/bash
#SBATCH --job-name=whisper_epochsweep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/%u/auto-avsr/output/%x_%j.out
#SBATCH --account=torch_pr_39_general
set -euo pipefail

# Stacks an epoch-count sweep on top of the confirmed warmup=25/wd=0.01/
# lr=1e-5 recipe (see whisper_wdwarmup_finetune.sbatch; the LR sweep on
# top of this closed negative, see whisper_lrsweep_finetune.sbatch).
# 30 epochs was inherited unchanged from the original recipe and never
# swept -- Nemotron and the AV Emformer both showed non-monotonic
# epoch-count behavior (peak then reverse) on this same small dataset,
# untested for Whisper full-FT.
#
# DOMAIN in {legal,legacy,merged}, SEED in {1,2,3}, EPOCHS (e.g. 20, 45)
# must be set by the caller:
#   DOMAIN=merged SEED=1 EPOCHS=20 sbatch slurm/whisper_epochsweep_finetune.sbatch

export PROJECT_ROOT="${PROJECT_ROOT:-/home/${USER}/auto-avsr-realtime}"
export WHISPER_ENV="${WHISPER_ENV:-/scratch/${USER}/envs/whisper_asr}"
export WHISPER_PY="${WHISPER_ENV}/bin/python"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export TMPDIR="${TMPDIR:-/scratch/${USER}/.tmp}"
export PYTHONNOUSERSITE=1
mkdir -p "/scratch/${USER}/auto-avsr/output"

: "${DOMAIN:?set DOMAIN=legal|legacy|merged}"
: "${SEED:?set SEED=1|2|3}"
: "${EPOCHS:?set EPOCHS=e.g. 20|45}"

SPLIT_ROOT="${SPLIT_ROOT:-/scratch/th3482/LipVideoData/patient_legal298_legacy96_split_v1}"
LABELS_OUT_DIR="${LABELS_OUT_DIR:-/scratch/${USER}/avsr_realtime/labels/splitv1}"
ROOT_DIR="${SPLIT_ROOT}"
TRAIN_FILE="${LABELS_OUT_DIR}/${DOMAIN}/train_spm1023.csv"
VAL_FILE="${LABELS_OUT_DIR}/${DOMAIN}/val_spm1023.csv"
TEST_FILE="${LABELS_OUT_DIR}/${DOMAIN}/test_spm1023.csv"
for f in "$TRAIN_FILE" "$VAL_FILE" "$TEST_FILE"; do
  [[ -s "$f" ]] || { echo "[FATAL] missing $f" >&2; exit 3; }
done

SP_MODEL="${SP_MODEL:-${PROJECT_ROOT}/spm/spm_unigram_1023.model}"
EXP_DIR="${EXP_DIR:-/scratch/${USER}/experiments/whisper_asr}"
EXP_NAME="whisper_largev3_specaug_splitv1_${DOMAIN}_epochsweep_${EPOCHS}_seed${SEED}"

echo "Starting at $(date) on $(hostname): DOMAIN=${DOMAIN} SEED=${SEED} EPOCHS=${EPOCHS} EXP_NAME=${EXP_NAME}"
echo "TRAIN_FILE=${TRAIN_FILE} ($(wc -l < "$TRAIN_FILE") rows)"
echo "VAL_FILE=${VAL_FILE} ($(wc -l < "$VAL_FILE") rows)"
echo "TEST_FILE=${TEST_FILE} ($(wc -l < "$TEST_FILE") rows)"

cd "${PROJECT_ROOT}"
export SEED

"${WHISPER_PY}" -m asr_baselines.whisper_finetune \
  --model openai/whisper-large-v3 \
  --root-dir "${ROOT_DIR}" \
  --train-file "${TRAIN_FILE}" \
  --val-file "${VAL_FILE}" \
  --sp-model-path "${SP_MODEL}" \
  --output-dir "${EXP_DIR}/${EXP_NAME}" \
  --epochs "${EPOCHS}" \
  --learning-rate 1e-5 \
  --warmup-steps 25 \
  --weight-decay 0.01 \
  --batch-size 8 \
  --grad-accum 2 \
  --precision bf16 \
  --unfreeze-encoder-layers -1 \
  --train-decoder \
  --num-workers 4

"${WHISPER_PY}" -m asr_baselines.whisper_eval \
  --model "${EXP_DIR}/${EXP_NAME}/best" \
  --root-dir "${ROOT_DIR}" \
  --label-file "${TEST_FILE}" \
  --sp-model-path "${SP_MODEL}" \
  --out "${EXP_DIR}/${EXP_NAME}/val_hyps.tsv"

echo "Finished at $(date)"
