#!/bin/bash
#SBATCH --job-name=whisper_ft
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/pa2753/auto-avsr/output/%x_%j.out
#SBATCH --error=/scratch/pa2753/auto-avsr/output/%x_%j.err
#SBATCH --mail-user=pa2753@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=torch_pr_39_general

# Finetune Whisper (small/medium) on patient data. Green recipe: train only the
# first 5 encoder layers + SpecAugment. Reuses the AV patient label CSVs.
#
# SpecAugment ablation: submit twice, once with SPECAUG=1 (default) and once
# with SPECAUG=0 -> compare WER. That is the Phase-1 "SpecAugment as a separate
# experiment" deliverable, measured through the Whisper vehicle.
#
# Runs OUTSIDE the AV singularity overlay: needs a standalone venv/conda env
# built from asr_baselines/requirements-whisper.txt at WHISPER_ENV (a bare
# filesystem path, e.g. /scratch/$USER/envs/whisper_asr -- NOT inside the
# read-only avsr.ext3 overlay, which is where the earlier attempt to build this
# env hit an inode-quota wall). PYTHONNOUSERSITE=1 is set deliberately: this
# venv's python otherwise silently resolves torchaudio/tqdm/etc from
# ~/.local/lib/pythonX.Y/site-packages if anything is pip-installed there,
# which can shadow the venv's own (compatible) versions with incompatible ones
# -- caught this exact failure mode (a stray torchaudio built for a different
# CUDA major version) before it could waste a GPU allocation.

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/pa2753/auto-avsr-realtime}"
export WHISPER_ENV="${WHISPER_ENV:-/scratch/${USER}/envs/whisper_asr}"
export WHISPER_PY="${WHISPER_ENV}/bin/python"
mkdir -p "/scratch/${USER}/auto-avsr/output"

source "${PROJECT_ROOT}/slurm/_patient_data.sh"

export WHISPER_MODEL="${WHISPER_MODEL:-openai/whisper-small}"
export SP_MODEL="${SP_MODEL:-${PROJECT_ROOT}/spm/spm_unigram_1023.model}"
export EXP_DIR="${EXP_DIR:-/scratch/pa2753/experiments/whisper_asr}"
export EXP_NAME="${EXP_NAME:-whisper_${DATASET_TAG}_j${SLURM_JOB_ID:-noslurm}}"
export EPOCHS="${EPOCHS:-30}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export PRECISION="${PRECISION:-bf16}"
export UNFREEZE_ENCODER_LAYERS="${UNFREEZE_ENCODER_LAYERS:-5}"
export SPECAUG="${SPECAUG:-1}"        # 0 -> ablation, disable SpecAugment
export NUM_WORKERS="${NUM_WORKERS:-4}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"

SPECAUG_FLAG=""
[[ "${SPECAUG}" == "0" ]] && SPECAUG_FLAG="--no-specaug"

[[ -x "${WHISPER_PY}" ]] || { echo "[FATAL] no python at ${WHISPER_PY}; set WHISPER_ENV to the whisper_asr venv" >&2; exit 4; }

echo "Starting Whisper finetune at $(date) on $(hostname)"
echo "WHISPER_ENV=${WHISPER_ENV} MODEL=${WHISPER_MODEL} EXP_NAME=${EXP_NAME} SPECAUG=${SPECAUG} UNFREEZE=${UNFREEZE_ENCODER_LAYERS}"
[[ "${DRY_RUN:-0}" == "1" ]] && { echo "DRY_RUN=1, not launching."; exit 0; }

cd "${PROJECT_ROOT}"
export PYTHONNOUSERSITE=1

[[ -s "${TRAIN_FILE}" ]] || "${WHISPER_PY}" scripts/regenerate_patient_labels.py --old-csv "${RAW_TRAIN_FILE}" --out "${TRAIN_FILE}"
[[ -s "${VAL_FILE}" ]]   || "${WHISPER_PY}" scripts/regenerate_patient_labels.py --old-csv "${RAW_VAL_FILE}" --out "${VAL_FILE}"

"${WHISPER_PY}" -m asr_baselines.whisper_finetune \
  --model "${WHISPER_MODEL}" \
  --root-dir "${ROOT_DIR}" \
  --train-file "${TRAIN_FILE}" \
  --val-file "${VAL_FILE}" \
  --sp-model-path "${SP_MODEL}" \
  --output-dir "${EXP_DIR}/${EXP_NAME}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --precision "${PRECISION}" \
  --unfreeze-encoder-layers "${UNFREEZE_ENCODER_LAYERS}" \
  --num-workers "${NUM_WORKERS}" \
  ${SPECAUG_FLAG}

"${WHISPER_PY}" -m asr_baselines.whisper_eval \
  --model "${EXP_DIR}/${EXP_NAME}/best" \
  --root-dir "${ROOT_DIR}" \
  --label-file "${VAL_FILE}" \
  --sp-model-path "${SP_MODEL}" \
  --out "${EXP_DIR}/${EXP_NAME}/val_hyps.tsv"

echo "Finished at $(date)"
