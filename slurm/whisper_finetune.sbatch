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
#SBATCH --account=torch_pr_39_tandon_advanced

# Finetune Whisper (small/medium) on patient data. Green recipe: train only the
# first 5 encoder layers + SpecAugment. Reuses the AV patient label CSVs.
#
# SpecAugment ablation: submit twice, once with SPECAUG=1 (default) and once
# with SPECAUG=0 -> compare WER. That is the Phase-1 "SpecAugment as a separate
# experiment" deliverable, measured through the Whisper vehicle.
#
# Needs a conda env with asr_baselines/requirements-whisper.txt (NOT the AV
# torch-2.6 env). Set CONDA_ENV / SIF_IMAGE / OVERLAY_EXT3 to match it.

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/pa2753/auto-avsr-realtime}"
export CONDA_ENV="${CONDA_ENV:-whisper_asr}"
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
export OVERLAY_EXT3="${OVERLAY_EXT3:-/scratch/pa2753/avsr/avsr.ext3}"
export SIF_IMAGE="${SIF_IMAGE:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"

SPECAUG_FLAG=""
[[ "${SPECAUG}" == "0" ]] && SPECAUG_FLAG="--no-specaug"

echo "Starting Whisper finetune at $(date) on $(hostname)"
echo "MODEL=${WHISPER_MODEL} EXP_NAME=${EXP_NAME} SPECAUG=${SPECAUG} UNFREEZE=${UNFREEZE_ENCODER_LAYERS}"
[[ "${DRY_RUN:-0}" == "1" ]] && { echo "DRY_RUN=1, not launching."; exit 0; }

singularity exec --nv \
  --overlay "${OVERLAY_EXT3}:ro" \
  --writable-tmpfs "${SIF_IMAGE}" \
  /bin/bash -lc '
set -euo pipefail
source /ext3/env.sh
conda activate '"${CONDA_ENV}"'
export SSL_CERT_FILE='"${PROJECT_ROOT}"'/cacert.pem
export HF_HOME="${HF_HOME:-/scratch/'"${USER}"'/hf_cache}"
cd '"${PROJECT_ROOT}"'

[[ -s "'"${TRAIN_FILE}"'" ]] || python scripts/regenerate_patient_labels.py --old-csv "'"${RAW_TRAIN_FILE}"'" --out "'"${TRAIN_FILE}"'"
[[ -s "'"${VAL_FILE}"'"   ]] || python scripts/regenerate_patient_labels.py --old-csv "'"${RAW_VAL_FILE}"'" --out "'"${VAL_FILE}"'"

python -m asr_baselines.whisper_finetune \
  --model "'"${WHISPER_MODEL}"'" \
  --root-dir "'"${ROOT_DIR}"'" \
  --train-file "'"${TRAIN_FILE}"'" \
  --val-file "'"${VAL_FILE}"'" \
  --sp-model-path "'"${SP_MODEL}"'" \
  --output-dir "'"${EXP_DIR}/${EXP_NAME}"'" \
  --epochs "'"${EPOCHS}"'" \
  --learning-rate "'"${LEARNING_RATE}"'" \
  --batch-size "'"${BATCH_SIZE}"'" \
  --grad-accum "'"${GRAD_ACCUM}"'" \
  --precision "'"${PRECISION}"'" \
  --unfreeze-encoder-layers "'"${UNFREEZE_ENCODER_LAYERS}"'" \
  --num-workers "'"${NUM_WORKERS}"'" \
  '"${SPECAUG_FLAG}"'

python -m asr_baselines.whisper_eval \
  --model "'"${EXP_DIR}/${EXP_NAME}/best"'" \
  --root-dir "'"${ROOT_DIR}"'" \
  --label-file "'"${VAL_FILE}"'" \
  --sp-model-path "'"${SP_MODEL}"'" \
  --out "'"${EXP_DIR}/${EXP_NAME}/val_hyps.tsv"'"
'
echo "Finished at $(date)"
