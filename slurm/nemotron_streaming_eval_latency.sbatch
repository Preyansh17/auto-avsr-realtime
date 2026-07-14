#!/bin/bash
#SBATCH --job-name=nemotron_stream_latency
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/%u/auto-avsr/output/%x_%j.out
#SBATCH --account=torch_pr_39_general

# Same contract as nemotron_streaming_eval.sbatch, but via
# nemotron_streaming_eval_latency.py for TTFT (word-commit lag is unavailable
# for Nemotron -- see that script's docstring for the NeMo bug that blocks it).
# Uses --att-context-size (a discrete lookahead preset, e.g. "70,13") instead
# of raw --chunk-size/--left-chunks; found via
# model.encoder.att_context_size_all on the checkpoint if unsure what's valid.
#
#   EXP_OUT=/scratch/pa2753/experiments/nemotron_asr/<run-name> \
#   ATT_CONTEXT_SIZE=70,13 sbatch slurm/nemotron_streaming_eval_latency.sbatch

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/${USER}/auto-avsr-realtime}"
export NEMO_ENV="${NEMO_ENV:-/scratch/${USER}/envs/nemo_asr}"
export NEMO_PY="${NEMO_ENV}/bin/python"
mkdir -p "/scratch/${USER}/auto-avsr/output"

[[ -n "${EXP_OUT:-}" ]] || { echo "[FATAL] set EXP_OUT=<run output dir> (must contain nemotron_patient.nemo + val_manifest.jsonl)" >&2; exit 4; }
[[ -x "${NEMO_PY}" ]] || { echo "[FATAL] no python at ${NEMO_PY}; set NEMO_ENV to the nemo_asr venv" >&2; exit 4; }

export ATT_CONTEXT_SIZE="${ATT_CONTEXT_SIZE:-}"
AC_FLAG=""
[[ -n "${ATT_CONTEXT_SIZE}" ]] && AC_FLAG="--att-context-size ${ATT_CONTEXT_SIZE}"

cd "${PROJECT_ROOT}"
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="${NEMO_ENV}/lib:${LD_LIBRARY_PATH:-}"

echo "Starting Nemotron streaming latency eval at $(date) on $(hostname): EXP_OUT=${EXP_OUT}"

"${NEMO_PY}" -m asr_baselines.nemotron_streaming_eval_latency \
  --model "${EXP_OUT}/nemotron_patient.nemo" \
  --manifest "${EXP_OUT}/val_manifest.jsonl" \
  --out "${EXP_OUT}/streaming_val_hyps_latency.tsv" \
  ${AC_FLAG}

echo "Finished at $(date)"
