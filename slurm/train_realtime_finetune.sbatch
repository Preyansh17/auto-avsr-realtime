#!/bin/bash
#SBATCH --job-name=avsr_rt_finetune
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="rtx8000|v100|a100"
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=pa2753@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Fine-tune the pretrained device_avsr streaming model on patient data.
# Prereqs inside the overlay env (see README): conda env "avsr_realtime"
# with torch/torchaudio 2.6.0; run scripts/download_assets.py and
# scripts/bootstrap_from_jit.py once beforehand.

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/${USER}/auto-avsr-realtime}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}}"
export ROOT_DIR="${ROOT_DIR:-${SCRATCH_ROOT}/LipVideoData/patient_25p_crops_unseen}"
export TRAIN_FILE="${TRAIN_FILE:-labels/patient_retinaface_train_transcript_lengths_seg24s_spm1023.csv}"
export VAL_FILE="${VAL_FILE:-labels/patient_retinaface_val_transcript_lengths_seg24s_spm1023.csv}"
export EXP_DIR="${EXP_DIR:-${SCRATCH_ROOT}/experiments/avsr_realtime}"
export EXP_NAME="${EXP_NAME:-patient_finetune_device}"
export SP_MODEL="${SP_MODEL:-${PROJECT_ROOT}/spm/spm_unigram_1023.model}"
export PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-${PROJECT_ROOT}/cpts/online_avsr_bootstrap.ckpt}"
export EPOCHS="${EPOCHS:-30}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-4}"
export PRECISION="${PRECISION:-bf16-mixed}"
export LEARNING_RATE="${LEARNING_RATE:-0.0001}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export OVERLAY_EXT3="${OVERLAY_EXT3:-${SCRATCH_ROOT}/avsr/overlay-15GB-500K.ext3}"
export SIF_IMAGE="${SIF_IMAGE:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"

singularity exec --nv \
  --overlay "${OVERLAY_EXT3}:ro" \
  --writable-tmpfs \
  "${SIF_IMAGE}" \
  /bin/bash -lc '
set -euo pipefail
source /ext3/env.sh
conda activate avsr_realtime
cd "${PROJECT_ROOT}"

python train.py \
  --model-source bootstrap \
  --root-dir "${ROOT_DIR}" \
  --train-file "${TRAIN_FILE}" \
  --val-file "${VAL_FILE}" \
  --exp-dir "${EXP_DIR}" \
  --exp-name "${EXP_NAME}" \
  --sp-model-path "${SP_MODEL}" \
  --pretrained-model-path "${PRETRAINED_MODEL_PATH}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --accumulate-grad-batches "${ACCUMULATE_GRAD_BATCHES}" \
  --precision "${PRECISION}" \
  --learning-rate "${LEARNING_RATE}" \
  --num-workers "${NUM_WORKERS}"
'
