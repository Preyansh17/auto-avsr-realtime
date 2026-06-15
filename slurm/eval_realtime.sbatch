#!/bin/bash
#SBATCH --job-name=avsr_rt_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16GB
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/pa2753/auto-avsr/output/%x_%j/%x_%j.out
#SBATCH --error=/scratch/pa2753/auto-avsr/output/%x_%j/%x_%j.err
#SBATCH --mail-user=pa2753@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_39_tandon_advanced

# Streaming evaluation (WER + RTF) on the same patient data / roots / modes as
# training. Defaults to the resolved val split as the test set (matching the
# offline TEST_FILE=val fallback); override with TEST_FILE for a held-out set.
# Patient clips are pre-cropped mouth ROIs, so PREPROCESS=roi (no re-detection).

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/pa2753/auto-avsr-realtime}"
export CONDA_ENV="${CONDA_ENV:-avsr_realtime}"
mkdir -p "/scratch/${USER}/auto-avsr/output"

source "${PROJECT_ROOT}/slurm/_patient_data.sh"

export CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/cpts/online_avsr_bootstrap.ckpt}"
export SP_MODEL="${SP_MODEL:-${PROJECT_ROOT}/spm/spm_unigram_1023.model}"
export MODE="${MODE:-streaming}"
export PREPROCESS="${PREPROCESS:-roi}"
export TEST_FILE="${TEST_FILE:-${VAL_FILE}}"
export RAW_TEST_FILE="${RAW_TEST_FILE:-${RAW_VAL_FILE}}"
export OUTPUT_DIR="${OUTPUT_DIR:-/scratch/pa2753/experiments/avsr_realtime/eval}"
export OVERLAY_EXT3="${OVERLAY_EXT3:-/scratch/pa2753/avsr/avsr.ext3}"
export SIF_IMAGE="${SIF_IMAGE:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"

[[ -s "${CHECKPOINT}" ]] || { echo "[FATAL] missing checkpoint ${CHECKPOINT}" >&2; exit 4; }

echo "Evaluating ${CHECKPOINT} (mode=${MODE}, data=${DATASET_TAG}) at $(date)"
[[ "${DRY_RUN:-0}" == "1" ]] && { echo "DRY_RUN=1, not launching."; exit 0; }

singularity exec --nv \
  --overlay "${OVERLAY_EXT3}:ro" \
  --writable-tmpfs "${SIF_IMAGE}" \
  /bin/bash -lc '
set -euo pipefail
source /ext3/env.sh
conda activate '"${CONDA_ENV}"'
export SSL_CERT_FILE='"${PROJECT_ROOT}"'/cacert.pem
cd '"${PROJECT_ROOT}"'

[[ -s "'"${TEST_FILE}"'" ]] || python scripts/regenerate_patient_labels.py --old-csv "'"${RAW_TEST_FILE}"'" --out "'"${TEST_FILE}"'"

python eval.py \
  --checkpoint "'"${CHECKPOINT}"'" \
  --mode "'"${MODE}"'" \
  --root-dir "'"${ROOT_DIR}"'" \
  --test-file "'"${TEST_FILE}"'" \
  --sp-model-path "'"${SP_MODEL}"'" \
  --preprocess "'"${PREPROCESS}"'" \
  --output-dir "'"${OUTPUT_DIR}"'"
'
echo "Finished at $(date)"
