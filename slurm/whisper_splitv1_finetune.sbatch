#!/bin/bash
#SBATCH --job-name=whisper_splitv1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32GB
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/%u/auto-avsr/output/%x_%j.out
#SBATCH --account=torch_pr_39_general
set -euo pipefail

# Full-FT Whisper large-v3 + SpecAugment (current best recipe) on an
# independently-audited three-way split (legal/legacy/merged) -- see
# results/whisper_streaming_investigation_2026-07-09.md for the split's
# provenance (split_audit.json) and why it replaced the old ad-hoc merge.
# DOMAIN in {legal,legacy,merged}, SEED in {1,2,3} must be set by the caller:
#
#   DOMAIN=legal SEED=1 sbatch slurm/whisper_splitv1_finetune.sbatch
#
# The account above (torch_pr_39_general) is a live preference, not fixed --
# switch it to torch_pr_39_tandon_advanced (or whichever) if one queue is
# congested; SLURM won't let a pending job move accounts, cancel+resubmit.
#
# SPLIT_ROOT's label CSVs must already be in this repo's spm_unigram_1023
# tokenization, NOT the older unigram5000 format some raw label sources still
# carry (they look identical -- same 4-column shape -- but decode to garbage
# token ids and crash whisper_finetune.py with a sentencepiece IndexError).
# Run scripts/regenerate_patient_labels.py first if unsure; LABELS_OUT_DIR
# below is where this project's own regenerated split_v1 labels live.

: "${DOMAIN:?set DOMAIN=legal|legacy|merged}"
: "${SEED:?set SEED=1|2|3}"

export PROJECT_ROOT="${PROJECT_ROOT:-/home/${USER}/auto-avsr-realtime}"
export WHISPER_ENV="${WHISPER_ENV:-/scratch/${USER}/envs/whisper_asr}"
export WHISPER_PY="${WHISPER_ENV}/bin/python"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export TMPDIR="${TMPDIR:-/scratch/${USER}/.tmp}"
export PYTHONNOUSERSITE=1
mkdir -p "/scratch/${USER}/auto-avsr/output"

SPLIT_ROOT="${SPLIT_ROOT:-/scratch/th3482/LipVideoData/patient_legal298_legacy96_split_v1}"
LABELS_OUT_DIR="${LABELS_OUT_DIR:-/scratch/${USER}/avsr_realtime/labels/splitv1}"
ROOT_DIR="${SPLIT_ROOT}"
TRAIN_FILE="${LABELS_OUT_DIR}/${DOMAIN}/train_spm1023.csv"
VAL_FILE="${LABELS_OUT_DIR}/${DOMAIN}/val_spm1023.csv"
TEST_FILE="${LABELS_OUT_DIR}/${DOMAIN}/test_spm1023.csv"
for f in "$TRAIN_FILE" "$VAL_FILE" "$TEST_FILE"; do
  [[ -s "$f" ]] || { echo "[FATAL] missing $f -- run regenerate_patient_labels.py first" >&2; exit 3; }
done

SP_MODEL="${SP_MODEL:-${PROJECT_ROOT}/spm/spm_unigram_1023.model}"
EXP_DIR="${EXP_DIR:-/scratch/${USER}/experiments/whisper_asr}"
EXP_NAME="whisper_largev3_specaug_splitv1_${DOMAIN}_seed${SEED}"

echo "Starting at $(date) on $(hostname): DOMAIN=${DOMAIN} SEED=${SEED} EXP_NAME=${EXP_NAME}"
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
  --epochs 30 \
  --learning-rate 1e-5 \
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
