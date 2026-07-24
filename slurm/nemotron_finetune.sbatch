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
export WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
export BATCH_SIZE="${BATCH_SIZE:-4}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export PRECISION="${PRECISION:-bf16-mixed}"
export UNFREEZE_ENCODER_LAYERS="${UNFREEZE_ENCODER_LAYERS:-5}"
export TRAIN_DECODER="${TRAIN_DECODER:-0}"   # 1 -> also unfreeze decoder (with -1 encoder = full finetune)
export LORA="${LORA:-0}"          # 1 -> freeze everything, train only a bottleneck adapter (overrides UNFREEZE_ENCODER_LAYERS/TRAIN_DECODER)
export LORA_DIM="${LORA_DIM:-32}"
export LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
export SPECAUG="${SPECAUG:-1}"
export SEED="${SEED:-}"           # set an int for reproducible runs
export NUM_WORKERS="${NUM_WORKERS:-2}"
export RESUME_CKPT="${RESUME_CKPT:-}"  # PTL .ckpt path to resume optimizer/scheduler/epoch state from
export SPEED_PERTURB="${SPEED_PERTURB:-0}"
export DECODER_JOINT_LR_SCALE="${DECODER_JOINT_LR_SCALE:-1.0}"  # e.g. 0.2 -> decoder+joint LR = LEARNING_RATE*0.2; 1.0 = disabled (flat LR)
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
RESUME_FLAG=""
[[ -n "${RESUME_CKPT}" ]] && RESUME_FLAG="--resume-ckpt ${RESUME_CKPT}"
SPEED_PERTURB_FLAG=""
[[ "${SPEED_PERTURB}" == "1" ]] && SPEED_PERTURB_FLAG="--speed-perturb"
LORA_FLAG=""
[[ "${LORA}" == "1" ]] && LORA_FLAG="--lora --lora-dim ${LORA_DIM} --lora-dropout ${LORA_DROPOUT}"

OUT="${EXP_DIR}/${EXP_NAME}"

echo "Starting Nemotron finetune at $(date) on $(hostname)"
echo "NEMO_ENV=${NEMO_ENV} MODEL=${NEMO_MODEL} EXP_NAME=${EXP_NAME} SPECAUG=${SPECAUG} UNFREEZE=${UNFREEZE_ENCODER_LAYERS} TRAIN_DECODER=${TRAIN_DECODER} LORA=${LORA} LORA_DIM=${LORA_DIM} SEED=${SEED}"
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

# TEST_FILE: held-out set for the FINAL reported eval, distinct from VAL_FILE
# (used during training to pick the best-epoch checkpoint via nemotron_finetune.py's
# SaveBestWER callback). Same rationale/pattern as whisper_finetune.sbatch's
# TEST_FILE. Defaults to VAL_FILE for back-compat.
export TEST_FILE="${TEST_FILE:-${VAL_FILE}}"

mkdir -p "${OUT}"
"${NEMO_PY}" -m asr_baselines.make_nemo_manifest \
  --root-dir "${ROOT_DIR}" --label-file "${TRAIN_FILE}" \
  --sp-model-path "${SP_MODEL}" --out "${OUT}/train_manifest.jsonl"
"${NEMO_PY}" -m asr_baselines.make_nemo_manifest \
  --root-dir "${ROOT_DIR}" --label-file "${VAL_FILE}" \
  --sp-model-path "${SP_MODEL}" --out "${OUT}/val_manifest.jsonl"
if [[ "${TEST_FILE}" == "${VAL_FILE}" ]]; then
  cp "${OUT}/val_manifest.jsonl" "${OUT}/test_manifest.jsonl"
else
  "${NEMO_PY}" -m asr_baselines.make_nemo_manifest \
    --root-dir "${ROOT_DIR}" --label-file "${TEST_FILE}" \
    --sp-model-path "${SP_MODEL}" --out "${OUT}/test_manifest.jsonl"
fi

"${NEMO_PY}" -m asr_baselines.nemotron_finetune \
  --model "${NEMO_MODEL}" \
  --train-manifest "${OUT}/train_manifest.jsonl" \
  --val-manifest "${OUT}/val_manifest.jsonl" \
  --output-dir "${OUT}" \
  --epochs "${EPOCHS}" \
  --learning-rate "${LEARNING_RATE}" \
  --weight-decay "${WEIGHT_DECAY}" \
  --decoder-joint-lr-scale "${DECODER_JOINT_LR_SCALE}" \
  --batch-size "${BATCH_SIZE}" \
  --grad-accum "${GRAD_ACCUM}" \
  --precision "${PRECISION}" \
  --unfreeze-encoder-layers "${UNFREEZE_ENCODER_LAYERS}" \
  --num-workers "${NUM_WORKERS}" \
  ${SPECAUG_FLAG} ${TRAIN_DECODER_FLAG} ${SEED_FLAG} ${LORA_FLAG} ${RESUME_FLAG} ${SPEED_PERTURB_FLAG}

# Report on the checkpoint the SELECTION set (VAL_FILE) picked as best,
# evaluated on the disjoint TEST_FILE -- falls back to the last-epoch
# checkpoint if val_wer selection never engaged (see nemotron_finetune.py's
# own WARNING for that case).
REPORT_CKPT="${OUT}/nemotron_best.nemo"
[[ -s "${REPORT_CKPT}" ]] || REPORT_CKPT="${OUT}/nemotron_patient.nemo"
"${NEMO_PY}" -m asr_baselines.nemotron_eval \
  --model "${REPORT_CKPT}" \
  --manifest "${OUT}/test_manifest.jsonl" \
  --out "${OUT}/test_hyps.tsv"

echo "Finished at $(date)"
