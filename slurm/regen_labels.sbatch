#!/bin/bash
#SBATCH --job-name=avsr_rt_labels
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=1:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=pa2753@nyu.edu
#SBATCH --mail-type=END,FAIL

# Re-tokenize patient label CSVs (unigram5000 -> spm_unigram_1023). CPU only.
# Pass the CSVs via OLD_CSVS (space-separated absolute paths).

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/${USER}/auto-avsr-realtime}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}}"
export OLD_CSVS="${OLD_CSVS:-${SCRATCH_ROOT}/LipVideoData/patient_25p_crops_unseen/labels/patient_retinaface_train_transcript_lengths_seg24s.csv}"
export OVERLAY_EXT3="${OVERLAY_EXT3:-${SCRATCH_ROOT}/avsr/overlay-15GB-500K.ext3}"
export SIF_IMAGE="${SIF_IMAGE:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"

singularity exec \
  --overlay "${OVERLAY_EXT3}:ro" \
  --writable-tmpfs \
  "${SIF_IMAGE}" \
  /bin/bash -lc '
set -euo pipefail
source /ext3/env.sh
conda activate avsr_realtime
cd "${PROJECT_ROOT}"

# shellcheck disable=SC2086
python scripts/regenerate_patient_labels.py --old-csv ${OLD_CSVS}
'
