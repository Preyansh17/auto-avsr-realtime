#!/bin/bash
#SBATCH --job-name=online_avsr_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="rtx8000|v100|a100"
#SBATCH --output=/scratch/th3482/auto-avsr/output/%x_%j/%x_%j.out
#SBATCH --error=/scratch/th3482/auto-avsr/output/%x_%j/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=torch_pr_39_tandon_advanced

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/th3482/auto-avsr}"
export ROOT_DIR="${ROOT_DIR:-/scratch/th3482/LipVideoData/patient_25p_crops_unseen}"
export TRAIN_FILE="${TRAIN_FILE:-patient_retinaface_train_transcript_lengths_seg24s.csv}"
export VAL_FILE="${VAL_FILE:-patient_retinaface_val_transcript_lengths_seg24s.csv}"
export TEST_FILE="${TEST_FILE:-${VAL_FILE}}"
export EXP_DIR="${EXP_DIR:-/scratch/th3482/experiments/online_avsr}"
export EXP_NAME="${EXP_NAME:-patient_online_avsr}"
export SP_MODEL="${SP_MODEL:-${PROJECT_ROOT}/spm/online_unigram1023.model}"
export GENERATE_SP_MODEL="${GENERATE_SP_MODEL:-0}"
export EPOCHS="${EPOCHS:-55}"
export NUM_NODES="${NUM_NODES:-1}"
export GPUS="${GPUS:-1}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export MAX_VIDEOS="${MAX_VIDEOS:-0}"
export LEARNING_RATE="${LEARNING_RATE:-0.0008}"
export PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-}"
export RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"
export DRY_RUN="${DRY_RUN:-0}"
export OUTPUT_DIR="${OUTPUT_DIR:-/scratch/th3482/auto-avsr/output/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
export OVERLAY_EXT3="${OVERLAY_EXT3:-/scratch/th3482/avsr/overlay-15GB-500K.ext3}"
export SIF_IMAGE="${SIF_IMAGE:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"

mkdir -p "${OUTPUT_DIR}" "${EXP_DIR}"

singularity exec --nv \
  --overlay "${OVERLAY_EXT3}:ro" \
  --writable-tmpfs \
  "${SIF_IMAGE}" \
  /bin/bash -lc '
set -euo pipefail
source /ext3/env.sh
conda activate auto_avsr
cd "${PROJECT_ROOT}"

python - <<'"'"'PY'"'"'
from online_avsr.checkpoint import preflight_environment
print(preflight_environment())
PY

args=(
  --model-source scratch
  --root-dir "${ROOT_DIR}"
  --train-file "${TRAIN_FILE}"
  --val-file "${VAL_FILE}"
  --test-file "${TEST_FILE}"
  --exp-dir "${EXP_DIR}"
  --exp-name "${EXP_NAME}"
  --sp-model-path "${SP_MODEL}"
  --epochs "${EPOCHS}"
  --num-nodes "${NUM_NODES}"
  --gpus "${GPUS}"
  --batch-size "${BATCH_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --learning-rate "${LEARNING_RATE}"
)

if [[ "${GENERATE_SP_MODEL}" == "1" ]]; then
  args+=(--generate-sp-model)
fi
if [[ "${MAX_VIDEOS}" != "0" ]]; then
  args+=(--max-videos "${MAX_VIDEOS}")
fi
if [[ -n "${PRETRAINED_MODEL_PATH}" ]]; then
  args+=(--pretrained-model-path "${PRETRAINED_MODEL_PATH}")
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  args+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  args+=(--dry-run)
fi

python train_online_patient_av.py "${args[@]}"
'
