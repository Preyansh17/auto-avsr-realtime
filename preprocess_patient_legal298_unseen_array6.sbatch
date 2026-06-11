#!/bin/bash
#SBATCH --job-name=prep_legal298_a6
#SBATCH --array=0-5
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/th3482/auto-avsr/output/%x_%A_%a/%x_%A_%a.out
#SBATCH --error=/scratch/th3482/auto-avsr/output/%x_%A_%a/%x_%A_%a.err
#SBATCH --mail-user=th3482@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=torch_pr_39_tandon_advanced

set -euo pipefail

NUM_SHARDS=6
TASK_ID="${SLURM_ARRAY_TASK_ID}"
if [ "${TASK_ID}" -lt 0 ] || [ "${TASK_ID}" -ge "${NUM_SHARDS}" ]; then
  echo "Invalid TASK_ID=${TASK_ID} for NUM_SHARDS=${NUM_SHARDS}"
  exit 1
fi

echo "Starting shard ${TASK_ID}/${NUM_SHARDS} at $(date) on $(hostname)"

INPUT_DIR="/scratch/th3482/LipVideoData/PatientLegal298/Fine"
OUTPUT_ROOT="/scratch/th3482/LipVideoData/patient_legal298_crops_unseen"
LABELS_DIR="${OUTPUT_ROOT}/labels"
NORM_DIR="${OUTPUT_ROOT}/_normalized_25p"
SUFFIX="_part${TASK_ID}"
TRAIN_PART="${LABELS_DIR}/patient_retinaface_train_transcript_lengths_seg24s${SUFFIX}.csv"
VAL_PART="${LABELS_DIR}/patient_retinaface_val_transcript_lengths_seg24s${SUFFIX}.csv"

mkdir -p "${LABELS_DIR}"
rm -f "${TRAIN_PART}" "${VAL_PART}"

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
    --normalized_dir ${NORM_DIR} \
    --num_shards ${NUM_SHARDS} \
    --shard_index ${TASK_ID} \
    --label_suffix ${SUFFIX}"

echo "Finished shard ${TASK_ID}/${NUM_SHARDS} at $(date)"
