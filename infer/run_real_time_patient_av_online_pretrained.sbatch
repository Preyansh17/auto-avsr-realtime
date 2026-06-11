#!/bin/bash
#SBATCH --job-name=online_avsr_inf
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16GB
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --constraint="rtx8000|v100|a100"
#SBATCH --output=/scratch/th3482/auto-avsr/output/%x_%j/%x_%j.out
#SBATCH --error=/scratch/th3482/auto-avsr/output/%x_%j/%x_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=torch_pr_39_tandon_advanced

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/th3482/auto-avsr}"
export PATIENT_DIR="${PATIENT_DIR:-/scratch/th3482/LipVideoData/patient_25p}"
export OUTPUT_DIR="${OUTPUT_DIR:-/scratch/th3482/auto-avsr/output/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
export SP_MODEL="${SP_MODEL:-${PROJECT_ROOT}/spm/spm_unigram_1023.model}"
export ONLINE_CHECKPOINT="${ONLINE_CHECKPOINT:-}"
export CHECKPOINT_URL="${CHECKPOINT_URL:-}"
export CHECKPOINT_SHA256="${CHECKPOINT_SHA256:-}"
export DETECTOR="${DETECTOR:-mediapipe}"
export MAX_VIDEOS="${MAX_VIDEOS:-0}"
export CHUNK_FRAMES="${CHUNK_FRAMES:-64}"
export BEAM_WIDTH="${BEAM_WIDTH:-20}"
export DRY_RUN="${DRY_RUN:-0}"
export OVERLAY_EXT3="${OVERLAY_EXT3:-/scratch/th3482/avsr/overlay-15GB-500K.ext3}"
export SIF_IMAGE="${SIF_IMAGE:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"

if [[ -z "${ONLINE_CHECKPOINT}" && -z "${CHECKPOINT_URL}" ]]; then
  echo "[FATAL] Set ONLINE_CHECKPOINT or CHECKPOINT_URL for pretrained online AVSR inference." >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"

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
  --model-source pretrained
  --sp-model-path "${SP_MODEL}"
  --patient-dir "${PATIENT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --detector "${DETECTOR}"
  --chunk-frames "${CHUNK_FRAMES}"
  --beam-width "${BEAM_WIDTH}"
)

if [[ -n "${CHECKPOINT_URL}" ]]; then
  args+=(--checkpoint-url "${CHECKPOINT_URL}")
  if [[ -n "${CHECKPOINT_SHA256}" ]]; then
    args+=(--checkpoint-sha256 "${CHECKPOINT_SHA256}")
  fi
else
  args+=(--checkpoint-path "${ONLINE_CHECKPOINT}")
fi

if [[ "${MAX_VIDEOS}" != "0" ]]; then
  args+=(--max-videos "${MAX_VIDEOS}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  args+=(--dry-run)
fi

python infer/real_time_patient_av_online.py "${args[@]}"
'
