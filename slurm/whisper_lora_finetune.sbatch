#!/bin/bash
#SBATCH --job-name=whisper_lora
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32GB
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/%u/auto-avsr/output/%x_%j.out
#SBATCH --account=torch_pr_39_general
set -euo pipefail

# LoRA finetune of Whisper large-v3 on merged split_v1, for the FIRST LoRA-vs-
# SimulStreaming comparison (all prior LoRA numbers were CarelessWhisper causal
# streaming, small/large-v2). Same data/SpecAugment/epochs as the full-FT
# wdwarmup headline. LR held at 1e-5 for a strict one-variable A/B (LoRA vs
# full-FT only; everything else matches the 8.97% headline exactly, incl.
# warmup=25 / wd=0.01). Adapter is merged into the base
# at save time, so the existing converter + streaming eval are LoRA-agnostic.
#
# DOMAIN/SEED set by caller (default merged/1).
#   DOMAIN=merged SEED=1 sbatch slurm/whisper_lora_finetune.sbatch

export PROJECT_ROOT="${PROJECT_ROOT:-/home/${USER}/auto-avsr-realtime}"
export WHISPER_ENV="${WHISPER_ENV:-/scratch/${USER}/envs/whisper_asr}"
export WHISPER_PY="${WHISPER_ENV}/bin/python"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export TMPDIR="${TMPDIR:-/scratch/${USER}/.tmp}"
export PYTHONNOUSERSITE=1
mkdir -p "/scratch/${USER}/auto-avsr/output"

DOMAIN="${DOMAIN:-merged}"
SEED="${SEED:-1}"

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
MODEL="${MODEL:-openai/whisper-large-v3}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LR="${LR:-1e-5}"   # strict A/B: match the full-FT wdwarmup headline exactly
# LR is folded into the run name so a non-default LR (e.g. LoRA's own 1e-4)
# doesn't clobber the strict-A/B (1e-5) checkpoint dir.
LR_TAG="$(echo "$LR" | tr -d '.-' )"
EXP_NAME="whisper_largev3_specaug_splitv1_${DOMAIN}_lora_r${LORA_RANK}_lr${LR_TAG}_seed${SEED}"

echo "Starting at $(date) on $(hostname): DOMAIN=${DOMAIN} SEED=${SEED} EXP_NAME=${EXP_NAME}"
echo "TRAIN_FILE=${TRAIN_FILE} ($(wc -l < "$TRAIN_FILE") rows)"

cd "${PROJECT_ROOT}"
export SEED

"${WHISPER_PY}" -m asr_baselines.whisper_finetune \
  --model "${MODEL}" \
  --root-dir "${ROOT_DIR}" \
  --train-file "${TRAIN_FILE}" \
  --val-file "${VAL_FILE}" \
  --sp-model-path "${SP_MODEL}" \
  --output-dir "${EXP_DIR}/${EXP_NAME}" \
  --epochs 30 \
  --learning-rate "${LR}" \
  --warmup-steps 25 \
  --weight-decay 0.01 \
  --batch-size 8 \
  --grad-accum 2 \
  --precision bf16 \
  --lora \
  --lora-rank "${LORA_RANK}" \
  --lora-alpha "${LORA_ALPHA}" \
  --num-workers 4

"${WHISPER_PY}" -m asr_baselines.whisper_eval \
  --model "${EXP_DIR}/${EXP_NAME}/best" \
  --root-dir "${ROOT_DIR}" \
  --label-file "${TEST_FILE}" \
  --sp-model-path "${SP_MODEL}" \
  --out "${EXP_DIR}/${EXP_NAME}/val_hyps.tsv"

echo "Finished at $(date)"
