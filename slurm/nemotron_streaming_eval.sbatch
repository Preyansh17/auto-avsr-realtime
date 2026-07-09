#!/bin/bash
#SBATCH --job-name=nemotron_stream_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/pa2753/auto-avsr/output/%x_%j.out
#SBATCH --mail-user=pa2753@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_39_general

# Cache-aware streaming WER/latency eval against an already-finetuned Nemotron
# checkpoint. Companion to nemotron_finetune.sbatch's offline eval step --
# point it at a finished run's output dir, no retraining needed.
#
#   EXP_OUT=/scratch/pa2753/experiments/nemotron_asr/<run-name> sbatch slurm/nemotron_streaming_eval.sbatch

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/pa2753/auto-avsr-realtime}"
export NEMO_ENV="${NEMO_ENV:-/scratch/${USER}/envs/nemo_asr}"
export NEMO_PY="${NEMO_ENV}/bin/python"
mkdir -p "/scratch/${USER}/auto-avsr/output"

[[ -n "${EXP_OUT:-}" ]] || { echo "[FATAL] set EXP_OUT=<run output dir> (must contain nemotron_patient.nemo + val_manifest.jsonl)" >&2; exit 4; }
[[ -x "${NEMO_PY}" ]] || { echo "[FATAL] no python at ${NEMO_PY}; set NEMO_ENV to the nemo_asr venv" >&2; exit 4; }

export CHUNK_SIZE="${CHUNK_SIZE:-}"
export LEFT_CHUNKS="${LEFT_CHUNKS:-}"
CHUNK_FLAG=""
[[ -n "${CHUNK_SIZE}" ]] && CHUNK_FLAG="--chunk-size ${CHUNK_SIZE}"
LEFT_FLAG=""
[[ -n "${LEFT_CHUNKS}" ]] && LEFT_FLAG="--left-chunks ${LEFT_CHUNKS}"

cd "${PROJECT_ROOT}"
export PYTHONNOUSERSITE=1
# Same torchcodec/ffmpeg gotcha as nemotron_finetune.sbatch -- LD_LIBRARY_PATH
# isn't set by conda since we invoke the venv python directly (no activate).
export LD_LIBRARY_PATH="${NEMO_ENV}/lib:${LD_LIBRARY_PATH:-}"

echo "Starting Nemotron streaming eval at $(date) on $(hostname): EXP_OUT=${EXP_OUT}"

"${NEMO_PY}" -m asr_baselines.nemotron_streaming_eval \
  --model "${EXP_OUT}/nemotron_patient.nemo" \
  --manifest "${EXP_OUT}/val_manifest.jsonl" \
  --out "${EXP_OUT}/streaming_val_hyps.tsv" \
  ${CHUNK_FLAG} ${LEFT_FLAG}

echo "Finished at $(date)"
