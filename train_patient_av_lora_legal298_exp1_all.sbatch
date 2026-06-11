#!/bin/bash
#SBATCH --job-name=av_lora_legal298
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16GB
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/th3482/auto-avsr/output/%x_%j/%x_%j.out
#SBATCH --error=/scratch/th3482/auto-avsr/output/%x_%j/%x_%j.err
#SBATCH --mail-user=th3482@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=torch_pr_39_tandon_advanced

set -euo pipefail

PROJECT_ROOT="/home/th3482/auto-avsr"
WANDB_PROJECT="${WANDB_PROJECT:-video50_vocab_training}"

LEGAL_ROOT="/scratch/th3482/LipVideoData/patient_legal298_crops_unseen"
LEGAL_VOCAB="${LEGAL_ROOT}/labels/patient_legal298_sentences.txt"

LEGACY_ROOT="/scratch/th3482/LipVideoData/patient_25p_crops_unseen"
LEGACY_VOCAB="${PROJECT_ROOT}/infer/50s.txt"

MERGED_ROOT="/scratch/th3482/LipVideoData/patient_legal298_plus_patient_unseen_crops"
MERGED_LABEL_DIR="${MERGED_ROOT}/labels"
MERGED_TRAIN_FILE="${MERGED_LABEL_DIR}/patient_retinaface_train_transcript_lengths_seg24s_legal298_plus_unseen.csv"
MERGED_VAL_FILE="${MERGED_LABEL_DIR}/patient_retinaface_val_transcript_lengths_seg24s_legal298_plus_unseen.csv"
MERGED_VOCAB="${MERGED_LABEL_DIR}/patient_legal298_plus_patient_unseen_sentences.txt"

if [[ -z "${RUN_DATA_MODE:-}" ]]; then
  if [[ "${MERGE_WITH_PATIENT_UNSEEN:-0}" == "1" ]]; then
    RUN_DATA_MODE="merged"
  else
    RUN_DATA_MODE="legal_only"
  fi
fi

MODALITY="${MODALITY:-audiovisual}"
USE_VOCAB_FILE="${USE_VOCAB_FILE:-1}"

if [[ "${USE_VOCAB_FILE}" != "0" && "${USE_VOCAB_FILE}" != "1" ]]; then
  echo "[FATAL] USE_VOCAB_FILE must be 0 or 1. Got: ${USE_VOCAB_FILE}" >&2
  exit 2
fi

case "${RUN_DATA_MODE}" in
  legal_only)
    DATASET_NAME="patient_legal298_retinaface_unseen"
    VOCAB_FILE="${LEGAL_VOCAB}"
    ;;
  merged)
    DATASET_NAME="patient_legal298_plus_patient_unseen"
    VOCAB_FILE="${MERGED_VOCAB}"
    ;;
  legacy_only)
    DATASET_NAME="patient_retinaface_unseen"
    VOCAB_FILE="${LEGACY_VOCAB}"
    ;;
  *)
    echo "[FATAL] RUN_DATA_MODE must be one of: legal_only, merged, legacy_only. Got: ${RUN_DATA_MODE}" >&2
    exit 2
    ;;
esac

case "${MODALITY}" in
  audiovisual)
    PRETRAINED_MODEL_PATH="${PROJECT_ROOT}/cpts/avsr_trlrwlrs2lrs3vox2avsp_base.pth"
    LORA_SCOPES="encoder,aux_encoder,decoder"
    ;;
  video)
    PRETRAINED_MODEL_PATH="${PROJECT_ROOT}/cpts/vsr_trlrwlrs2lrs3vox2avsp_base.pth"
    LORA_SCOPES="encoder,decoder"
    ;;
  audio)
    PRETRAINED_MODEL_PATH="${PROJECT_ROOT}/cpts/asr_trlrwlrs2lrs3vox2avsp_base.pth"
    LORA_SCOPES="encoder,decoder"
    ;;
  *)
    echo "[FATAL] MODALITY must be one of: audiovisual, video, audio. Got: ${MODALITY}" >&2
    exit 2
    ;;
esac

if [[ "${RUN_DATA_MODE}" == "merged" ]]; then
  mkdir -p "${MERGED_LABEL_DIR}"
  exec 9>"${MERGED_ROOT}/.merge_build.lock"
  flock 9

  if [[ ! -L "${MERGED_ROOT}/patient_retinaface_legal298" ]]; then
    ln -sfn "${LEGAL_ROOT}/patient_retinaface" "${MERGED_ROOT}/patient_retinaface_legal298"
  fi
  if [[ ! -L "${MERGED_ROOT}/patient_retinaface_legacy" ]]; then
    ln -sfn "${LEGACY_ROOT}/patient_retinaface" "${MERGED_ROOT}/patient_retinaface_legacy"
  fi

  NEED_MERGED_LABELS=0
  NEED_MERGED_VOCAB=0
  if [[ ! -s "${MERGED_TRAIN_FILE}" || ! -s "${MERGED_VAL_FILE}" ]]; then
    NEED_MERGED_LABELS=1
  fi
  if [[ "${USE_VOCAB_FILE}" == "1" && ! -s "${MERGED_VOCAB}" ]]; then
    NEED_MERGED_VOCAB=1
  fi

  if [[ "${NEED_MERGED_LABELS}" == "1" || "${NEED_MERGED_VOCAB}" == "1" ]]; then
    LEGAL_TRAIN="${LEGAL_ROOT}/labels/patient_retinaface_train_transcript_lengths_seg24s.csv"
    LEGAL_VAL="${LEGAL_ROOT}/labels/patient_retinaface_val_transcript_lengths_seg24s.csv"
    LEGACY_TRAIN="${LEGACY_ROOT}/labels/patient_retinaface_train_transcript_lengths_seg24s.csv"
    LEGACY_VAL="${LEGACY_ROOT}/labels/patient_retinaface_val_transcript_lengths_seg24s.csv"

    for f in "${LEGAL_TRAIN}" "${LEGAL_VAL}" "${LEGACY_TRAIN}" "${LEGACY_VAL}"; do
      if [[ ! -s "${f}" ]]; then
        echo "[FATAL] Missing required merge input: ${f}" >&2
        exit 3
      fi
    done

    if [[ "${NEED_MERGED_LABELS}" == "1" ]]; then
      train_tmp="$(mktemp "${MERGED_LABEL_DIR}/.train_XXXXXX.tmp")"
      val_tmp="$(mktemp "${MERGED_LABEL_DIR}/.val_XXXXXX.tmp")"

      awk -F',' 'BEGIN{OFS=","} {sub(/\r$/, "", $0); $1="patient_retinaface_legal298"; print}' "${LEGAL_TRAIN}" > "${train_tmp}"
      awk -F',' 'BEGIN{OFS=","} {sub(/\r$/, "", $0); $1="patient_retinaface_legacy"; print}' "${LEGACY_TRAIN}" >> "${train_tmp}"
      mv -f "${train_tmp}" "${MERGED_TRAIN_FILE}"

      awk -F',' 'BEGIN{OFS=","} {sub(/\r$/, "", $0); $1="patient_retinaface_legal298"; print}' "${LEGAL_VAL}" > "${val_tmp}"
      awk -F',' 'BEGIN{OFS=","} {sub(/\r$/, "", $0); $1="patient_retinaface_legacy"; print}' "${LEGACY_VAL}" >> "${val_tmp}"
      mv -f "${val_tmp}" "${MERGED_VAL_FILE}"
    fi

    if [[ "${NEED_MERGED_VOCAB}" == "1" ]]; then
      for f in "${LEGAL_VOCAB}" "${LEGACY_VOCAB}"; do
        if [[ ! -s "${f}" ]]; then
          echo "[FATAL] Missing required merge vocab input: ${f}" >&2
          exit 3
        fi
      done

      vocab_tmp="$(mktemp "${MERGED_LABEL_DIR}/.vocab_XXXXXX.tmp")"
      cat "${LEGAL_VOCAB}" "${LEGACY_VOCAB}" | awk '{sub(/\r$/, "", $0)} NF && !seen[$0]++ {print $0}' > "${vocab_tmp}"
      mv -f "${vocab_tmp}" "${MERGED_VOCAB}"
    fi
  fi

  # Release merge lock immediately after artifact preparation.
  flock -u 9
  exec 9>&-
fi

for f in "${PRETRAINED_MODEL_PATH}"; do
  if [[ ! -s "${f}" ]]; then
    echo "[FATAL] Required file missing or empty: ${f}" >&2
    exit 4
  fi
done

if [[ "${USE_VOCAB_FILE}" == "0" ]]; then
  VOCAB_FILE=""
else
  if [[ ! -s "${VOCAB_FILE}" ]]; then
    echo "[FATAL] Required vocab file missing or empty: ${VOCAB_FILE}" >&2
    exit 4
  fi
fi

EXP_DIR="/scratch/th3482/experiments/patient_lora_grid9"
BASE_EXP_NAME="lora_all_${RUN_DATA_MODE}_${MODALITY}"
if [[ "${USE_VOCAB_FILE}" == "0" ]]; then
  BASE_EXP_NAME="${BASE_EXP_NAME}_novocab"
fi
SLURM_TAG="${SLURM_JOB_ID:-noslurm}"
EXP_NAME="${BASE_EXP_NAME}_j${SLURM_TAG}"
MAX_STEPS="${MAX_STEPS:-2850}"
MAX_EPOCHS="${MAX_EPOCHS:-10000}"

if ! [[ "${MAX_STEPS}" =~ ^[0-9]+$ ]] || [[ "${MAX_STEPS}" -le 0 ]]; then
  echo "[FATAL] MAX_STEPS must be a positive integer. Got: ${MAX_STEPS}" >&2
  exit 5
fi
if ! [[ "${MAX_EPOCHS}" =~ ^[0-9]+$ ]] || [[ "${MAX_EPOCHS}" -le 0 ]]; then
  echo "[FATAL] MAX_EPOCHS must be a positive integer. Got: ${MAX_EPOCHS}" >&2
  exit 5
fi

echo "Starting LoRA grid run at $(date) on $(hostname)"
echo "RUN_DATA_MODE=${RUN_DATA_MODE}"
echo "DATASET_NAME=${DATASET_NAME}"
echo "MODALITY=${MODALITY}"
echo "USE_VOCAB_FILE=${USE_VOCAB_FILE}"
echo "PRETRAINED_MODEL_PATH=${PRETRAINED_MODEL_PATH}"
if [[ "${USE_VOCAB_FILE}" == "1" ]]; then
  echo "VOCAB_FILE=${VOCAB_FILE}"
else
  echo "VOCAB_FILE=<disabled>"
fi
echo "LORA_SCOPES=[${LORA_SCOPES}]"
echo "EXP_DIR=${EXP_DIR}"
echo "BASE_EXP_NAME=${BASE_EXP_NAME}"
echo "EXP_NAME=${EXP_NAME}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "MAX_STEPS=${MAX_STEPS}"
echo "MAX_EPOCHS=${MAX_EPOCHS}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, skip launching training."
  exit 0
fi

if [[ "${USE_VOCAB_FILE}" == "1" ]]; then
  singularity exec --nv \
    --overlay /scratch/th3482/avsr/overlay-15GB-500K.ext3:ro \
    --writable-tmpfs /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -lc "source /ext3/env.sh; \
    conda activate auto_avsr; \
    export SSL_CERT_FILE=${PROJECT_ROOT}/cacert.pem; \
    export HYDRA_FULL_ERROR=1; \
    export WANDB_PROJECT='${WANDB_PROJECT}'; \
    cd ${PROJECT_ROOT}; \
    python train.py \
      hydra.job.chdir=False \
      data.modality='${MODALITY}' \
      data.dataset='${DATASET_NAME}' \
      exp_dir='${EXP_DIR}' \
      exp_name='${EXP_NAME}' \
      pretrained_model_path='${PRETRAINED_MODEL_PATH}' \
      vocab_file='${VOCAB_FILE}' \
      beam_size=40 ctc_weight=0.1 pre_beam_ratio=1.5 \
      trainer.gpus=1 \
      +trainer.max_steps=${MAX_STEPS} \
      trainer.max_epochs=${MAX_EPOCHS} \
      lora.enabled=true lora.r=8 lora.alpha=16 lora.dropout=0.05 \
      lora.scopes=[${LORA_SCOPES}] \
      'lora.name_patterns=[]'"
else
  singularity exec --nv \
    --overlay /scratch/th3482/avsr/overlay-15GB-500K.ext3:ro \
    --writable-tmpfs /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
    /bin/bash -lc "source /ext3/env.sh; \
    conda activate auto_avsr; \
    export SSL_CERT_FILE=${PROJECT_ROOT}/cacert.pem; \
    export HYDRA_FULL_ERROR=1; \
    export WANDB_PROJECT='${WANDB_PROJECT}'; \
    cd ${PROJECT_ROOT}; \
    python train.py \
      hydra.job.chdir=False \
      data.modality='${MODALITY}' \
      data.dataset='${DATASET_NAME}' \
      exp_dir='${EXP_DIR}' \
      exp_name='${EXP_NAME}' \
      pretrained_model_path='${PRETRAINED_MODEL_PATH}' \
      beam_size=40 ctc_weight=0.1 pre_beam_ratio=1.5 \
      trainer.gpus=1 \
      +trainer.max_steps=${MAX_STEPS} \
      trainer.max_epochs=${MAX_EPOCHS} \
      lora.enabled=true lora.r=8 lora.alpha=16 lora.dropout=0.05 \
      lora.scopes=[${LORA_SCOPES}] \
      'lora.name_patterns=[]'"
fi

echo "Finished at $(date)"
