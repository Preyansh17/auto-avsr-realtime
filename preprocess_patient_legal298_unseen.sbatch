#!/bin/bash
#SBATCH --job-name=prep_patient_legal298_unseen
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/th3482/auto-avsr/output/%x_%j/%x_%j.out
#SBATCH --error=/scratch/th3482/auto-avsr/output/%x_%j/%x_%j.err
#SBATCH --mail-user=th3482@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=torch_pr_39_tandon_advanced

set -euo pipefail

echo "Starting at $(date) on $(hostname)"

INPUT_DIR="/scratch/th3482/LipVideoData/PatientLegal298/Fine"
OUTPUT_ROOT="/scratch/th3482/LipVideoData/patient_legal298_crops_unseen"
LABELS_DIR="${OUTPUT_ROOT}/labels"
VOCAB_FILE="${LABELS_DIR}/patient_legal298_sentences.txt"
NORM_DIR="${OUTPUT_ROOT}/_normalized_25p"

# Use a clean output target so label CSV line counts stay deterministic.
rm -rf "${OUTPUT_ROOT}/patient_retinaface"
rm -f "${LABELS_DIR}/patient_retinaface_train_transcript_lengths_seg24s.csv"
rm -f "${LABELS_DIR}/patient_retinaface_val_transcript_lengths_seg24s.csv"
rm -f "${VOCAB_FILE}"
mkdir -p "${LABELS_DIR}"

singularity exec --nv \
  --overlay /scratch/th3482/avsr/overlay-15GB-500K.ext3:ro \
  --writable-tmpfs /share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif \
  /bin/bash -lc "source /ext3/env.sh; \
  conda activate auto_avsr; \
  export SSL_CERT_FILE=/home/th3482/auto-avsr/cacert.pem; \
  python /home/th3482/auto-avsr/preprocess_patient.py \
    --input_dir ${INPUT_DIR} \
    --output_root ${OUTPUT_ROOT} \
    --detectors retinaface \
    --batch_size 1 \
    --transcript_field 1 \
    --val_sentence_count 30 \
    --normalize_25fps \
    --normalized_dir ${NORM_DIR}"

# Build vocabulary sentence file from Fine filenames (field index 1).
find "${INPUT_DIR}" -maxdepth 1 -type f -name '*.mp4' -printf '%f\n' \
  | sort -V \
  | awk -F'_' '{print $2}' \
  | awk 'NF && !seen[$0]++' \
  > "${VOCAB_FILE}"

echo "Generated vocab file: ${VOCAB_FILE}"
echo "Vocab lines: $(wc -l < "${VOCAB_FILE}")"
echo "Finished at $(date)"
