#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_patient_av_lora_legal298_exp1_all.sbatch"

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "[FATAL] Missing training sbatch: ${TRAIN_SCRIPT}" >&2
  exit 1
fi

modes=(legal_only merged legacy_only)
modalities=(audiovisual video audio)
MAX_STEPS="${MAX_STEPS:-2850}"
MAX_EPOCHS="${MAX_EPOCHS:-10000}"
USE_VOCAB_FILE="${USE_VOCAB_FILE:-1}"

echo "job_id | mode | modality | exp_name"
for mode in "${modes[@]}"; do
  for modality in "${modalities[@]}"; do
    sbatch_out="$(sbatch --export=ALL,RUN_DATA_MODE="${mode}",MODALITY="${modality}",MAX_STEPS="${MAX_STEPS}",MAX_EPOCHS="${MAX_EPOCHS}",USE_VOCAB_FILE="${USE_VOCAB_FILE}" "${TRAIN_SCRIPT}")"
    job_id="$(awk '{print $4}' <<< "${sbatch_out}")"
    exp_name="lora_all_${mode}_${modality}"
    if [[ "${USE_VOCAB_FILE}" == "0" ]]; then
      exp_name="${exp_name}_novocab"
    fi
    echo "${job_id} | ${mode} | ${modality} | ${exp_name}"
  done
done
