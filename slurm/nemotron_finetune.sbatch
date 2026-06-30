#!/bin/bash
#SBATCH --job-name=nemotron_ft
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=6:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/pa2753/auto-avsr/output/%x_%j.out
#SBATCH --error=/scratch/pa2753/auto-avsr/output/%x_%j.err
#SBATCH --mail-user=pa2753@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --account=torch_pr_39_tandon_advanced

# Finetune Nemotron streaming FastConformer-RNNT on patient data. Green recipe:
# train first 5 encoder layers + NeMo SpecAugment (cut freq, blown time).
# Builds NeMo manifests from the AV patient CSVs, finetunes, then evals.
#
# SpecAugment ablation: SPECAUG=1 (default) vs SPECAUG=0.
#
# Runs in its OWN env (NeMo torch pin clashes with the AV + Whisper envs).
# Prefer the NeMo container image if available; else a conda env built from
# asr_baselines/requirements-nemotron.txt. Override SIF_IMAGE / CONDA_ENV.

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/pa2753/auto-avsr-realtime}"
export CONDA_ENV="${CONDA_ENV:-nemo_asr}"
mkdir -p "/scratch/${USER}/auto-avsr/output"

source "${PROJECT_ROOT}/slurm/_patient_data.sh"

export NEMO_MODEL="${NEMO_MODEL:-nvidia/nemotron-speech-streaming-en-0.6b}"
export SP_MODEL="${SP_MODEL:-${PROJECT_ROOT}/spm/spm_unigram_1023.model}"
export EXP_DIR="${EXP_DIR:-/scratch/pa2753/experiments/nemotron_asr}"
export EXP_NAME="${EXP_NAME:-nemotron_${DATASET_TAG}_j${SLURM_JOB_ID:-noslurm}}"
export EPOCHS="${EPOCHS:-30}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export BATCH_SIZE="${BATCH_SIZE:-8}"
export PRECISION="${PRECISION:-bf16-mixed}"
export UNFREEZE_ENCODER_LAYERS="${UNFREEZE_ENCODER_LAYERS:-5}"
export SPECAUG="${SPECAUG:-1}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export OVERLAY_EXT3="${OVERLAY_EXT3:-/scratch/pa2753/avsr/avsr.ext3}"
export SIF_IMAGE="${SIF_IMAGE:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"

SPECAUG_FLAG=""
[[ "${SPECAUG}" == "0" ]] && SPECAUG_FLAG="--no-specaug"

OUT="${EXP_DIR}/${EXP_NAME}"

echo "Starting Nemotron finetune at $(date) on $(hostname)"
echo "MODEL=${NEMO_MODEL} EXP_NAME=${EXP_NAME} SPECAUG=${SPECAUG} UNFREEZE=${UNFREEZE_ENCODER_LAYERS}"
[[ "${DRY_RUN:-0}" == "1" ]] && { echo "DRY_RUN=1, not launching."; exit 0; }

singularity exec --nv \
  --overlay "${OVERLAY_EXT3}:ro" \
  --writable-tmpfs "${SIF_IMAGE}" \
  /bin/bash -lc '
set -euo pipefail
source /ext3/env.sh
conda activate '"${CONDA_ENV}"'
export SSL_CERT_FILE='"${PROJECT_ROOT}"'/cacert.pem
export HF_HOME="${HF_HOME:-/scratch/'"${USER}"'/hf_cache}"
cd '"${PROJECT_ROOT}"'

[[ -s "'"${TRAIN_FILE}"'" ]] || python scripts/regenerate_patient_labels.py --old-csv "'"${RAW_TRAIN_FILE}"'" --out "'"${TRAIN_FILE}"'"
[[ -s "'"${VAL_FILE}"'"   ]] || python scripts/regenerate_patient_labels.py --old-csv "'"${RAW_VAL_FILE}"'" --out "'"${VAL_FILE}"'"

mkdir -p "'"${OUT}"'"
python -m asr_baselines.make_nemo_manifest \
  --root-dir "'"${ROOT_DIR}"'" --label-file "'"${TRAIN_FILE}"'" \
  --sp-model-path "'"${SP_MODEL}"'" --out "'"${OUT}/train_manifest.jsonl"'"
python -m asr_baselines.make_nemo_manifest \
  --root-dir "'"${ROOT_DIR}"'" --label-file "'"${VAL_FILE}"'" \
  --sp-model-path "'"${SP_MODEL}"'" --out "'"${OUT}/val_manifest.jsonl"'"

python -m asr_baselines.nemotron_finetune \
  --model "'"${NEMO_MODEL}"'" \
  --train-manifest "'"${OUT}/train_manifest.jsonl"'" \
  --val-manifest "'"${OUT}/val_manifest.jsonl"'" \
  --output-dir "'"${OUT}"'" \
  --epochs "'"${EPOCHS}"'" \
  --learning-rate "'"${LEARNING_RATE}"'" \
  --batch-size "'"${BATCH_SIZE}"'" \
  --precision "'"${PRECISION}"'" \
  --unfreeze-encoder-layers "'"${UNFREEZE_ENCODER_LAYERS}"'" \
  --num-workers "'"${NUM_WORKERS}"'" \
  '"${SPECAUG_FLAG}"'

python -m asr_baselines.nemotron_eval \
  --model "'"${OUT}/nemotron_patient.nemo"'" \
  --manifest "'"${OUT}/val_manifest.jsonl"'" \
  --out "'"${OUT}/val_hyps.tsv"'"
'
echo "Finished at $(date)"
