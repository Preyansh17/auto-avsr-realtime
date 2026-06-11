#!/bin/bash
set -euo pipefail

cd /home/th3482/auto-avsr

OUTPUT_ROOT=/scratch/th3482/LipVideoData/patient_legal298_crops_unseen
LABEL_DIR=${OUTPUT_ROOT}/labels

# Clean previous incorrect preprocessing artifacts before re-running.
rm -rf "${OUTPUT_ROOT}/patient_retinaface"
rm -rf "${OUTPUT_ROOT}/_normalized_25p"
rm -f "${LABEL_DIR}"/patient_retinaface_train_transcript_lengths_seg24s*.csv
rm -f "${LABEL_DIR}"/patient_retinaface_val_transcript_lengths_seg24s*.csv
rm -f "${LABEL_DIR}"/patient_legal298_sentences.txt

array_job_id=$(sbatch preprocess_patient_legal298_unseen_array6.sbatch | awk '{print $4}')
merge_job_id=$(sbatch --dependency=afterok:${array_job_id} preprocess_patient_legal298_unseen_array6_finalize.sbatch | awk '{print $4}')

echo "Submitted array job:   ${array_job_id}"
echo "Submitted merge job:   ${merge_job_id} (afterok:${array_job_id})"
echo "Track with: squeue -j ${array_job_id},${merge_job_id}"
