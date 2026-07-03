#!/bin/bash
#SBATCH --job-name=nemotron_ft
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/pa2753/auto-avsr/output/%x_%j.out
#SBATCH --mail-user=pa2753@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=torch_pr_39_general

# Finetune Nemotron streaming FastConformer-RNNT on patient data. Green recipe:
# train first 5 encoder layers + NeMo SpecAugment (cut freq, blown time).
# Builds NeMo manifests from the AV patient CSVs, finetunes, then evals.
#
# SpecAugment ablation: SPECAUG=1 (default) vs SPECAUG=0.
#
# Runs OUTSIDE any singularity overlay: needs a standalone venv/conda env at
# NEMO_ENV (bare filesystem path, e.g. /scratch/$USER/envs/nemo_asr -- same
# pattern as the Whisper env, built via asr_baselines/requirements-nemotron.txt
# using `conda create -p <path>` + pip install, with CONDA_PKGS_DIRS/PIP_CACHE_DIR/
# TMPDIR redirected to scratch to avoid /home inode-quota exhaustion).
#
# Getting this pipeline running required fixing 16 real bugs (missing
# sentencepiece/torchaudio/torchcodec/ffmpeg, Lhotse dataloader config schema
# drift, lightning-vs-pytorch_lightning package split, scheduler max_steps
# needing Lhotse's lazy dataset to support len() which it doesn't, and a hard
# segfault ultimately traced to the missing numba-cuda companion package --
# see git log / requirements-nemotron.txt for the full story). All fixes are
# baked into nemotron_finetune.py and requirements-nemotron.txt; a fresh env
# built from that requirements file should just work.

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/pa2753/auto-avsr-realtime}"
export NEMO_ENV="${NEMO_ENV:-/scratch/${USER}/envs/nemo_asr}"
export NEMO_PY="${NEMO_ENV}/bin/python"
mkdir -p "/scratch/${USER}/auto-avsr/output"

source "${PROJECT_ROOT}/slurm/_patient_data.sh"

export NEMO_MODEL="${NEMO_MODEL:-nvidia/nemotron-speech-streaming-en-0.6b}"
export SP_MODEL="${SP_MODEL:-${PROJECT_ROOT}/spm/spm_unigram_1023.model}"
export EXP_DIR="${EXP_DIR:-/scratch/pa2753/experiments/nemotron_asr}"
export EXP_NAME="${EXP_NAME:-nemotron_${DATASET_TAG}_j${SLURM_JOB_ID:-noslurm}}"
export EPOCHS="${EPOCHS:-30}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export PRECISION="${PRECISION:-bf16-mixed}"
export UNFREEZE_ENCODER_LAYERS="${UNFREEZE_ENCODER_LAYERS:-5}"
export TRAIN_DECODER="${TRAIN_DECODER:-0}"   # 1 -> also unfreeze decoder (with -1 encoder = full finetune)
export SPECAUG="${SPECAUG:-1}"
export SEED="${SEED:-}"           # set an int for reproducible runs
export NUM_WORKERS="${NUM_WORKERS:-2}"
export HF_HOME="${HF_HOME:-/scratch/${USER}/.huggingface}"
export TMPDIR="${TMPDIR:-/scratch/${USER}/.tmp}"
mkdir -p "${TMPDIR}"

[[ -x "${NEMO_PY}" ]] || { echo "[FATAL] no python at ${NEMO_PY}; set NEMO_ENV to the nemo_asr venv" >&2; exit 4; }

SPECAUG_FLAG=""
[[ "${SPECAUG}" == "0" ]] && SPECAUG_FLAG="--no-specaug"
TRAIN_DECODER_FLAG=""
[[ "${TRAIN_DECODER}" == "1" ]] && TRAIN_DECODER_FLAG="--train-decoder"
SEED_FLAG=""
[[ -n "${SEED}" ]] && SEED_FLAG="--seed ${SEED}"

OUT="${EXP_DIR}/${EXP_NAME}"

echo "Starting Nemotron finetune at $(date) on $(hostname)"
echo "NEMO_ENV=${NEMO_ENV} MODEL=${NEMO_MODEL} EXP_NAME=${EXP_NAME} SPECAUG=${SPECAUG} UNFREEZE=${UNFREEZE_ENCODER_LAYERS} TRAIN_DECODER=${TRAIN_DECODER} SEED=${SEED}"
[[ "${DRY_RUN:-0}" == "1" ]] && { echo "DRY_RUN=1, not launching."; exit 0; }

cd "${PROJECT_ROOT}"
export PYTHONNOUSERSITE=1
# make_nemo_manifest.py's duration fallback (torchaudio.info() doesn't exist on
# this stack -> falls back to torchaudio.load(), which routes through
# torchcodec, which dlopen()s the conda-forge-installed libavutil.so etc.).
# Those libs live under NEMO_ENV/lib; conda activate normally adds this to
# LD_LIBRARY_PATH automatically, but we call the venv python directly (no
# activate) specifically to avoid the module-load complexity, so it must be
# set explicitly -- confirmed missing this breaks manifest building on a fresh
# compute node (worked interactively on the login node only because that shell
# had leftover env state from installing ffmpeg there).
export LD_LIBRARY_PATH="${NEMO_ENV}/lib:${LD_LIBRARY_PATH:-}"

[[ -s "${TRAIN_FILE}" ]] || "${NEMO_PY}" scripts/regenerate_patient_labels.py --old-csv "${RAW_TRAIN_FILE}" --out "${TRAIN_FILE}"
[[ -s "${VAL_FILE}" ]]   || "${NEMO_PY}" scripts/regenerate_patient_labels.py --old-csv "${RAW_VAL_FILE}" --out "${VAL_FILE}"

mkdir -p "${OUT}"
"${NEMO_PY}" -m asr_baselines.make_nemo_manifest \
  --root-dir "${ROOT_DIR}" --label-file "${TRAIN_FILE}" \
  --sp-model-path "${SP_MODEL}" --out "${OUT}/train_manifest.jsonl"
"${NEMO_PY}" -m asr_baselines.make_nemo_manifest \
  --root-dir "${ROOT_DIR}" --label-file "${VAL_FILE}" \
  --sp-model-path "${SP_MODEL}" --out "${OUT}/val_manifest.jsonl"

"${NEMO_PY}" -m asr_baselines.nemotron_finetune \
  --model "${NEMO_MODEL}" \
  --train-manifest "${OUT}/train_manifest.jsonl" \
  --val-manifest "${OUT}/val_manifest.jsonl" \
  --output-dir "${OUT}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --precision "${PRECISION}" \
  --unfreeze-encoder-layers "${UNFREEZE_ENCODER_LAYERS}" \
  --num-workers "${NUM_WORKERS}" \
  ${SPECAUG_FLAG} ${TRAIN_DECODER_FLAG} ${SEED_FLAG}

"${NEMO_PY}" -m asr_baselines.nemotron_eval \
  --model "${OUT}/nemotron_patient.nemo" \
  --manifest "${OUT}/val_manifest.jsonl" \
  --out "${OUT}/val_hyps.tsv"

echo "Finished at $(date)"
