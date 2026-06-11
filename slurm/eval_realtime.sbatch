#!/bin/bash
#SBATCH --job-name=avsr_rt_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --mail-user=pa2753@nyu.edu
#SBATCH --mail-type=END,FAIL

# Streaming evaluation (WER + RTF) of a checkpoint over the patient test set.

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/${USER}/auto-avsr-realtime}"
export SCRATCH_ROOT="${SCRATCH_ROOT:-/scratch/${USER}}"
export ROOT_DIR="${ROOT_DIR:-${SCRATCH_ROOT}/LipVideoData/patient_25p_crops_unseen}"
export TEST_FILE="${TEST_FILE:-labels/patient_retinaface_test_transcript_lengths_seg24s_spm1023.csv}"
export CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/cpts/online_avsr_bootstrap.ckpt}"
export SP_MODEL="${SP_MODEL:-${PROJECT_ROOT}/spm/spm_unigram_1023.model}"
export MODE="${MODE:-streaming}"
export PREPROCESS="${PREPROCESS:-roi}"
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRATCH_ROOT}/experiments/avsr_realtime/eval}"
export OVERLAY_EXT3="${OVERLAY_EXT3:-${SCRATCH_ROOT}/avsr/overlay-15GB-500K.ext3}"
export SIF_IMAGE="${SIF_IMAGE:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"

singularity exec --nv \
  --overlay "${OVERLAY_EXT3}:ro" \
  --writable-tmpfs \
  "${SIF_IMAGE}" \
  /bin/bash -lc '
set -euo pipefail
source /ext3/env.sh
conda activate avsr_realtime
cd "${PROJECT_ROOT}"

python eval.py \
  --checkpoint "${CHECKPOINT}" \
  --mode "${MODE}" \
  --root-dir "${ROOT_DIR}" \
  --test-file "${TEST_FILE}" \
  --sp-model-path "${SP_MODEL}" \
  --preprocess "${PREPROCESS}" \
  --output-dir "${OUTPUT_DIR}"
'
