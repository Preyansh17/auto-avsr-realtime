#!/bin/bash
#SBATCH --job-name=avsr_rt_facecrop
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24GB
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/pa2753/auto-avsr/output/%x_%j.out
#SBATCH --error=/scratch/pa2753/auto-avsr/output/%x_%j.err
#SBATCH --mail-user=pa2753@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_39_general

# Pre-bake face crops from the ORIGINAL patient videos, matching the label
# split, so the pretrained device frontend sees what it expects. CPU only
# (mediapipe). Produces <OUT_ROOT>/patient_retinaface/<rel_path>.mp4 (+ .wav).
#
# Run once per corpus (legal + legacy). The originals you have:
#   legal:  /scratch/th3482/LipVideoData/patient_legal298_crops_unseen/_normalized_25p
#   legacy: /scratch/th3482/LipVideoData/patient_25p
# Set EVAL_MODE-style RUN_DATA_MODE to pick which corpus's labels drive it.

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/pa2753/auto-avsr-realtime}"
export CONDA_ENV="${CONDA_ENV:-avsr_realtime}"
mkdir -p "/scratch/${USER}/auto-avsr/output"

# Resolve the label CSVs + dataset_name for the chosen corpus.
export RUN_DATA_MODE="${RUN_DATA_MODE:-legal_only}"   # legal_only | legacy_only
source "${PROJECT_ROOT}/slurm/_patient_data.sh"

# Originals + output root (override per corpus).
export ORIG_ROOT="${ORIG_ROOT:?set ORIG_ROOT to the original-videos dir}"
export OUT_ROOT="${OUT_ROOT:-/scratch/pa2753/LipVideoData/${DATASET_TAG}_facecrop}"
export MATCH="${MATCH:-basename}"                     # basename | relpath
export OVERLAY_EXT3="${OVERLAY_EXT3:-/scratch/pa2753/avsr/avsr.ext3}"
export SIF_IMAGE="${SIF_IMAGE:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"

echo "Face-cropping ${RUN_DATA_MODE}: orig=${ORIG_ROOT} -> out=${OUT_ROOT} (match=${MATCH})"
[[ "${DRY_RUN:-0}" == "1" ]] && DRY="--dry-run" || DRY=""

singularity exec \
  --overlay "${OVERLAY_EXT3}:ro" \
  --writable-tmpfs "${SIF_IMAGE}" \
  /bin/bash -lc '
set -euo pipefail
source /ext3/env.sh
conda activate '"${CONDA_ENV}"'
cd '"${PROJECT_ROOT}"'

python scripts/face_crop_patient.py \
  --label-csv "'"${RAW_TRAIN_FILE}"'" "'"${RAW_VAL_FILE}"'" \
  --orig-root "'"${ORIG_ROOT}"'" \
  --out-root "'"${OUT_ROOT}"'" \
  --match "'"${MATCH}"'" '"${DRY}"'
'
echo "Finished at $(date)"
