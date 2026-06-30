#!/bin/bash
#SBATCH --job-name=avsr_rt_finetune
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16GB
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/pa2753/auto-avsr/output/%x_%j.out
#SBATCH --error=/scratch/pa2753/auto-avsr/output/%x_%j.err
#SBATCH --mail-user=pa2753@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=torch_pr_39_tandon_advanced

# Full fine-tune (all params) of the pretrained device_avsr streaming model.
# Same patient data / roots / modes / env as the LoRA template; the only
# difference is no --lora flag. See train_realtime_lora.sbatch for the notes
# on what does and doesn't carry over from the offline runs.

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/pa2753/auto-avsr-realtime}"
export CONDA_ENV="${CONDA_ENV:-avsr_realtime}"
export WANDB_PROJECT="${WANDB_PROJECT:-avsr_realtime}"
mkdir -p "/scratch/${USER}/auto-avsr/output"

source "${PROJECT_ROOT}/slurm/_patient_data.sh"

export PRETRAINED_MODEL_PATH="${PRETRAINED_MODEL_PATH:-${PROJECT_ROOT}/cpts/online_avsr_bootstrap.ckpt}"
export SP_MODEL="${SP_MODEL:-${PROJECT_ROOT}/spm/spm_unigram_1023.model}"
export MODALITY="${MODALITY:-audiovisual}"   # audiovisual | audio | video
export EXP_DIR="${EXP_DIR:-/scratch/pa2753/experiments/avsr_realtime_finetune}"
export EXP_NAME="${EXP_NAME:-finetune_${DATASET_TAG}_av_j${SLURM_JOB_ID:-noslurm}}"
export MAX_STEPS="${MAX_STEPS:-2850}"
export MAX_EPOCHS="${MAX_EPOCHS:-10000}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export ACCUMULATE_GRAD_BATCHES="${ACCUMULATE_GRAD_BATCHES:-4}"
export PRECISION="${PRECISION:-16-mixed}"
export LEARNING_RATE="${LEARNING_RATE:-0.0001}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export SPECAUG="${SPECAUG:-0}"   # 1 -> Green-style SpecAugment (ablation)
export OVERLAY_EXT3="${OVERLAY_EXT3:-/scratch/pa2753/avsr/avsr.ext3}"
export SIF_IMAGE="${SIF_IMAGE:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"

[[ -s "${PRETRAINED_MODEL_PATH}" ]] || { echo "[FATAL] missing ${PRETRAINED_MODEL_PATH}; run scripts/download_assets.py then scripts/bootstrap_from_jit.py" >&2; exit 4; }

SPECAUG_FLAG=""
[[ "${SPECAUG}" == "1" ]] && SPECAUG_FLAG="--specaug"

echo "Starting streaming full fine-tune at $(date) on $(hostname)"
echo "EXP_NAME=${EXP_NAME}  MAX_STEPS=${MAX_STEPS}  SPECAUG=${SPECAUG}"
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

[[ -s "'"${TRAIN_FILE}"'" ]] || python scripts/regenerate_patient_labels.py --old-csv "'"${RAW_TRAIN_FILE}"'" --out "'"${TRAIN_FILE}"'"
[[ -s "'"${VAL_FILE}"'"   ]] || python scripts/regenerate_patient_labels.py --old-csv "'"${RAW_VAL_FILE}"'" --out "'"${VAL_FILE}"'"

python train.py \
  --modality "'"${MODALITY}"'" \
  --model-source bootstrap \
  --root-dir "'"${ROOT_DIR}"'" \
  --train-file "'"${TRAIN_FILE}"'" \
  --val-file "'"${VAL_FILE}"'" \
  --sp-model-path "'"${SP_MODEL}"'" \
  --pretrained-model-path "'"${PRETRAINED_MODEL_PATH}"'" \
  --exp-dir "'"${EXP_DIR}"'" \
  --exp-name "'"${EXP_NAME}"'" \
  --max-steps "'"${MAX_STEPS}"'" \
  --epochs "'"${MAX_EPOCHS}"'" \
  --batch-size "'"${BATCH_SIZE}"'" \
  --accumulate-grad-batches "'"${ACCUMULATE_GRAD_BATCHES}"'" \
  --precision "'"${PRECISION}"'" \
  --learning-rate "'"${LEARNING_RATE}"'" \
  --num-workers "'"${NUM_WORKERS}"'" \
  '"${SPECAUG_FLAG}"'
'
echo "Finished at $(date)"
