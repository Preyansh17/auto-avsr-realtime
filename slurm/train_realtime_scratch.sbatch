#!/bin/bash
#SBATCH --job-name=avsr_rt_scratch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/pa2753/auto-avsr/output/%x_%j.out
#SBATCH --error=/scratch/pa2753/auto-avsr/output/%x_%j.err
#SBATCH --mail-user=pa2753@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=torch_pr_39_tandon_advanced

# Train the streaming model from scratch (recipe architecture, 88x88 mouth
# ROIs) on the same patient data / roots / modes as the other templates.
# Patient data alone is small for from-scratch RNN-T training; prefer the
# finetune or lora templates. Uses the 1023-piece vocab by default; pass
# GENERATE_SP_MODEL=1 to build a fresh vocab from the patient transcripts.

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/pa2753/auto-avsr-realtime}"
export CONDA_ENV="${CONDA_ENV:-avsr_realtime}"
export WANDB_PROJECT="${WANDB_PROJECT:-avsr_realtime}"
mkdir -p "/scratch/${USER}/auto-avsr/output"

source "${PROJECT_ROOT}/slurm/_patient_data.sh"

export SP_MODEL="${SP_MODEL:-${PROJECT_ROOT}/spm/spm_unigram_1023.model}"
export ARCHITECTURE="${ARCHITECTURE:-recipe}"
export SEGMENT_LENGTH="${SEGMENT_LENGTH:-32}"
export RIGHT_CONTEXT_LENGTH="${RIGHT_CONTEXT_LENGTH:-4}"
export EXP_DIR="${EXP_DIR:-/scratch/pa2753/experiments/avsr_realtime_scratch}"
export EXP_NAME="${EXP_NAME:-scratch_${DATASET_TAG}_av_j${SLURM_JOB_ID:-noslurm}}"
export MAX_STEPS="${MAX_STEPS:-0}"                # 0 -> use epochs
export MAX_EPOCHS="${MAX_EPOCHS:-55}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-8}"
export PRECISION="${PRECISION:-bf16-mixed}"
export LEARNING_RATE="${LEARNING_RATE:-0.0008}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export GENERATE_SP_MODEL="${GENERATE_SP_MODEL:-0}"
export OVERLAY_EXT3="${OVERLAY_EXT3:-/scratch/pa2753/avsr/avsr.ext3}"
export SIF_IMAGE="${SIF_IMAGE:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"

echo "Starting streaming from-scratch run at $(date) on $(hostname)"
echo "EXP_NAME=${EXP_NAME}  ARCHITECTURE=${ARCHITECTURE}  MAX_STEPS=${MAX_STEPS}"
[[ "${DRY_RUN:-0}" == "1" ]] && { echo "DRY_RUN=1, not launching."; exit 0; }

singularity exec --nv \
  --overlay "${OVERLAY_EXT3}:ro" \
  --writable-tmpfs "${SIF_IMAGE}" \
  /bin/bash -lc '
set -euo pipefail
source /ext3/env.sh
conda activate '"${CONDA_ENV}"'
export SSL_CERT_FILE='"${PROJECT_ROOT}"'/cacert.pem
cd '"${PROJECT_ROOT}"'

gen_flag=""
if [[ "'"${GENERATE_SP_MODEL}"'" == "1" ]]; then
  gen_flag="--generate-sp-model"
else
  [[ -s "'"${TRAIN_FILE}"'" ]] || python scripts/regenerate_patient_labels.py --old-csv "'"${RAW_TRAIN_FILE}"'" --out "'"${TRAIN_FILE}"'"
  [[ -s "'"${VAL_FILE}"'"   ]] || python scripts/regenerate_patient_labels.py --old-csv "'"${RAW_VAL_FILE}"'" --out "'"${VAL_FILE}"'"
fi

python train.py \
  --model-source scratch \
  --architecture "'"${ARCHITECTURE}"'" \
  --segment-length "'"${SEGMENT_LENGTH}"'" \
  --right-context-length "'"${RIGHT_CONTEXT_LENGTH}"'" \
  --root-dir "'"${ROOT_DIR}"'" \
  --train-file "'"${TRAIN_FILE}"'" \
  --val-file "'"${VAL_FILE}"'" \
  --sp-model-path "'"${SP_MODEL}"'" \
  ${gen_flag} \
  --exp-dir "'"${EXP_DIR}"'" \
  --exp-name "'"${EXP_NAME}"'" \
  --max-steps "'"${MAX_STEPS}"'" \
  --epochs "'"${MAX_EPOCHS}"'" \
  --batch-size "'"${BATCH_SIZE}"'" \
  --accumulate-grad-batches "'"${ACCUMULATE_GRAD_BATCHES}"'" \
  --precision "'"${PRECISION}"'" \
  --learning-rate "'"${LEARNING_RATE}"'" \
  --num-workers "'"${NUM_WORKERS}"'"
'
echo "Finished at $(date)"
