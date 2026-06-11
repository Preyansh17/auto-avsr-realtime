#!/bin/bash
#SBATCH --job-name=prep_legal298_a6_merge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=0:30:00
#SBATCH --output=/scratch/th3482/auto-avsr/output/%x_%j/%x_%j.out
#SBATCH --error=/scratch/th3482/auto-avsr/output/%x_%j/%x_%j.err
#SBATCH --mail-user=th3482@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=torch_pr_39_tandon_advanced

set -euo pipefail

echo "Finalizing array outputs at $(date) on $(hostname)"

NUM_SHARDS=6
INPUT_DIR="/scratch/th3482/LipVideoData/PatientLegal298/Fine"
OUTPUT_ROOT="/scratch/th3482/LipVideoData/patient_legal298_crops_unseen"
LABELS_DIR="${OUTPUT_ROOT}/labels"

TRAIN_FINAL="${LABELS_DIR}/patient_retinaface_train_transcript_lengths_seg24s.csv"
VAL_FINAL="${LABELS_DIR}/patient_retinaface_val_transcript_lengths_seg24s.csv"
VOCAB_FILE="${LABELS_DIR}/patient_legal298_sentences.txt"

mkdir -p "${LABELS_DIR}"
rm -f "${TRAIN_FINAL}" "${VAL_FINAL}" "${VOCAB_FILE}"

for i in $(seq 0 $((NUM_SHARDS - 1))); do
  suffix="_part${i}"
  train_part="${LABELS_DIR}/patient_retinaface_train_transcript_lengths_seg24s${suffix}.csv"
  val_part="${LABELS_DIR}/patient_retinaface_val_transcript_lengths_seg24s${suffix}.csv"

  if [ ! -f "${train_part}" ]; then
    echo "Missing shard train label file: ${train_part}"
    exit 1
  fi
  if [ ! -f "${val_part}" ]; then
    echo "Missing shard val label file: ${val_part}"
    exit 1
  fi

  cat "${train_part}" >> "${TRAIN_FINAL}"
  cat "${val_part}" >> "${VAL_FINAL}"
done

find "${INPUT_DIR}" -maxdepth 1 -type f -name '*.mp4' -printf '%f\n' \
  | sort -V \
  | awk -F'_' '{print $2}' \
  | awk 'NF && !seen[$0]++' \
  > "${VOCAB_FILE}"

train_n=$(wc -l < "${TRAIN_FINAL}")
val_n=$(wc -l < "${VAL_FINAL}")
vocab_n=$(wc -l < "${VOCAB_FILE}")

echo "Final line counts:"
echo "  train: ${train_n}"
echo "  val:   ${val_n}"
echo "  vocab: ${vocab_n}"

if [ "${train_n}" -ne 268 ]; then
  echo "Expected train=268, got ${train_n}"
  exit 1
fi
if [ "${val_n}" -ne 30 ]; then
  echo "Expected val=30, got ${val_n}"
  exit 1
fi
if [ "${vocab_n}" -ne 298 ]; then
  echo "Expected vocab=298, got ${vocab_n}"
  exit 1
fi

echo "Finalize completed at $(date)"
