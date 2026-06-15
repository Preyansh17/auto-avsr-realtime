#!/bin/bash
# Shared patient-data resolution for the streaming sbatch templates.
#
# Mirrors train_patient_av_lora_legal298_exp1_all_pa2753_outputs.sbatch from
# the offline repo: same data roots, same seg24s label CSVs, same three data
# modes (legal_only / merged / legacy_only) and the same merge logic
# (per-source dataset_name rewrite + symlinks). The ONE addition the streaming
# model needs is label re-tokenization: the offline CSVs carry unigram5000
# token ids, the Emformer RNN-T needs the 1023-piece ids, so the resolved
# train/val CSVs are passed through scripts/regenerate_patient_labels.py
# (producing *_spm1023.csv) before training.
#
# Inputs (env): RUN_DATA_MODE (or MERGE_WITH_PATIENT_UNSEEN=1), PROJECT_ROOT.
# Outputs (exported): ROOT_DIR, RAW_TRAIN_FILE, RAW_VAL_FILE, DATASET_TAG.
# The merge step (bash/awk) runs here, outside the container, exactly as in
# the reference. Re-tokenization (needs python+sentencepiece) is run inside
# the container by each sbatch via regen_labels_in_container helper.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/${USER}/auto-avsr-realtime}"

# Same physical roots as the offline reference (data is shared, read-only).
LEGAL_ROOT="${LEGAL_ROOT:-/scratch/th3482/LipVideoData/patient_legal298_crops_unseen}"
LEGACY_ROOT="${LEGACY_ROOT:-/scratch/th3482/LipVideoData/patient_25p_crops_unseen}"
MERGED_ROOT="${MERGED_ROOT:-/scratch/${USER}/LipVideoData/patient_legal298_plus_patient_unseen_crops}"
MERGED_LABEL_DIR="${MERGED_ROOT}/labels"
MERGED_TRAIN_FILE="${MERGED_LABEL_DIR}/patient_retinaface_train_transcript_lengths_seg24s_legal298_plus_unseen.csv"
MERGED_VAL_FILE="${MERGED_LABEL_DIR}/patient_retinaface_val_transcript_lengths_seg24s_legal298_plus_unseen.csv"

# seg24s label file names (identical to the offline repo).
TRAIN_CSV_NAME="patient_retinaface_train_transcript_lengths_seg24s.csv"
VAL_CSV_NAME="patient_retinaface_val_transcript_lengths_seg24s.csv"

if [[ -z "${RUN_DATA_MODE:-}" ]]; then
  if [[ "${MERGE_WITH_PATIENT_UNSEEN:-0}" == "1" ]]; then
    RUN_DATA_MODE="merged"
  else
    RUN_DATA_MODE="legal_only"
  fi
fi

case "${RUN_DATA_MODE}" in
  legal_only)
    ROOT_DIR="${LEGAL_ROOT}"
    RAW_TRAIN_FILE="${LEGAL_ROOT}/labels/${TRAIN_CSV_NAME}"
    RAW_VAL_FILE="${LEGAL_ROOT}/labels/${VAL_CSV_NAME}"
    DATASET_TAG="legal_only"
    ;;
  legacy_only)
    ROOT_DIR="${LEGACY_ROOT}"
    RAW_TRAIN_FILE="${LEGACY_ROOT}/labels/${TRAIN_CSV_NAME}"
    RAW_VAL_FILE="${LEGACY_ROOT}/labels/${VAL_CSV_NAME}"
    DATASET_TAG="legacy_only"
    ;;
  merged)
    ROOT_DIR="${MERGED_ROOT}"
    RAW_TRAIN_FILE="${MERGED_TRAIN_FILE}"
    RAW_VAL_FILE="${MERGED_VAL_FILE}"
    DATASET_TAG="merged"
    mkdir -p "${MERGED_LABEL_DIR}"
    exec 9>"${MERGED_ROOT}/.merge_build.lock"
    flock 9
    # Symlink each source's clips under a per-source dataset_name (matches the
    # column-1 rewrite below and my fork's root_dir/dataset_name/rel_path).
    [[ -L "${MERGED_ROOT}/patient_retinaface_legal298" ]] || \
      ln -sfn "${LEGAL_ROOT}/patient_retinaface" "${MERGED_ROOT}/patient_retinaface_legal298"
    [[ -L "${MERGED_ROOT}/patient_retinaface_legacy" ]] || \
      ln -sfn "${LEGACY_ROOT}/patient_retinaface" "${MERGED_ROOT}/patient_retinaface_legacy"
    if [[ ! -s "${MERGED_TRAIN_FILE}" || ! -s "${MERGED_VAL_FILE}" ]]; then
      LEGAL_TRAIN="${LEGAL_ROOT}/labels/${TRAIN_CSV_NAME}"
      LEGAL_VAL="${LEGAL_ROOT}/labels/${VAL_CSV_NAME}"
      LEGACY_TRAIN="${LEGACY_ROOT}/labels/${TRAIN_CSV_NAME}"
      LEGACY_VAL="${LEGACY_ROOT}/labels/${VAL_CSV_NAME}"
      for f in "${LEGAL_TRAIN}" "${LEGAL_VAL}" "${LEGACY_TRAIN}" "${LEGACY_VAL}"; do
        [[ -s "${f}" ]] || { echo "[FATAL] Missing merge input: ${f}" >&2; exit 3; }
      done
      train_tmp="$(mktemp "${MERGED_LABEL_DIR}/.train_XXXXXX.tmp")"
      val_tmp="$(mktemp "${MERGED_LABEL_DIR}/.val_XXXXXX.tmp")"
      awk -F',' 'BEGIN{OFS=","} {sub(/\r$/,"",$0); $1="patient_retinaface_legal298"; print}' "${LEGAL_TRAIN}" >  "${train_tmp}"
      awk -F',' 'BEGIN{OFS=","} {sub(/\r$/,"",$0); $1="patient_retinaface_legacy";   print}' "${LEGACY_TRAIN}" >> "${train_tmp}"
      mv -f "${train_tmp}" "${MERGED_TRAIN_FILE}"
      awk -F',' 'BEGIN{OFS=","} {sub(/\r$/,"",$0); $1="patient_retinaface_legal298"; print}' "${LEGAL_VAL}" >  "${val_tmp}"
      awk -F',' 'BEGIN{OFS=","} {sub(/\r$/,"",$0); $1="patient_retinaface_legacy";   print}' "${LEGACY_VAL}" >> "${val_tmp}"
      mv -f "${val_tmp}" "${MERGED_VAL_FILE}"
    fi
    flock -u 9
    exec 9>&-
    ;;
  *)
    echo "[FATAL] RUN_DATA_MODE must be legal_only | merged | legacy_only. Got: ${RUN_DATA_MODE}" >&2
    exit 2
    ;;
esac

for f in "${RAW_TRAIN_FILE}" "${RAW_VAL_FILE}"; do
  [[ -s "${f}" ]] || { echo "[FATAL] Missing label CSV: ${f}" >&2; exit 4; }
done

# Final CSV names the streaming trainer consumes (re-tokenized in-container).
TRAIN_FILE="${RAW_TRAIN_FILE%.csv}_spm1023.csv"
VAL_FILE="${RAW_VAL_FILE%.csv}_spm1023.csv"

export PROJECT_ROOT ROOT_DIR RAW_TRAIN_FILE RAW_VAL_FILE TRAIN_FILE VAL_FILE DATASET_TAG RUN_DATA_MODE

echo "RUN_DATA_MODE=${RUN_DATA_MODE}"
echo "ROOT_DIR=${ROOT_DIR}"
echo "RAW_TRAIN_FILE=${RAW_TRAIN_FILE}"
echo "RAW_VAL_FILE=${RAW_VAL_FILE}"
echo "TRAIN_FILE(spm1023)=${TRAIN_FILE}"
echo "VAL_FILE(spm1023)=${VAL_FILE}"
