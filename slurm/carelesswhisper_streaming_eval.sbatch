#!/bin/bash
#SBATCH --job-name=carelesswhisper_stream_eval
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/%u/auto-avsr/output/%x_%j.out
#SBATCH --account=torch_pr_39_tandon_advanced

# CarelessWhisper/WhisperRT causal streaming eval (asr_baselines/
# carelesswhisper_streaming_eval.py). Zero-shot released checkpoints first
# (no patient finetune yet) -- establishes the latency floor this architecture
# actually delivers, and the WER gap a finetune has to close.
#
# One-time setup (login node). MUST be a conda prefix env, NOT `python -m
# venv`: venv symlinks bin/python to /usr/bin/python3, which is 3.12 on login
# nodes but 3.9 on compute nodes -- the job then runs the wrong interpreter
# with an empty site-packages (found the hard way, job 13551431).
#   git clone https://github.com/tomer9080/CarelessWhisper-streaming \
#     /scratch/$USER/third_party/CarelessWhisper-streaming
#   module load anaconda3/2025.06
#   conda create -y -p /scratch/$USER/envs/whisper_rt python=3.10
#   export PYTHONNOUSERSITE=1   # else pip sees ~/.local py3.10 packages as
#                               # "already satisfied" and skips installing
#                               # them into the env, which this job (with
#                               # PYTHONNOUSERSITE=1) then can't see
#   /scratch/$USER/envs/whisper_rt/bin/pip install \
#     torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
#   /scratch/$USER/envs/whisper_rt/bin/pip install -r \
#     $PROJECT_ROOT/asr_baselines/requirements-carelesswhisper.txt
# (use TMPDIR=/scratch/$USER/tmp -- login-node /tmp is a 2GB tmpfs, torch
# wheel won't fit; checkpoints download anonymously, HF repo is public)
#
#   MANIFEST=/scratch/pa2753/experiments/whisper_asr/whisper_largev3_specaug_splitv1_legal_seed1/test_hyps.tsv \
#   MODEL=small CHUNK_MS=300 sbatch slurm/carelesswhisper_streaming_eval.sbatch

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/${USER}/auto-avsr-realtime}"
export CW_ENV="${CW_ENV:-/scratch/${USER}/envs/whisper_rt}"
export CW_PY="${CW_ENV}/bin/python"
export CARELESS_DIR="${CARELESS_DIR:-/scratch/${USER}/third_party/CarelessWhisper-streaming}"
mkdir -p "/scratch/${USER}/auto-avsr/output"

[[ -n "${MANIFEST:-}" ]] || { echo "[FATAL] set MANIFEST=<tsv with path/ref columns>" >&2; exit 4; }
[[ -x "${CW_PY}" ]] || { echo "[FATAL] no python at ${CW_PY}; set CW_ENV" >&2; exit 4; }
[[ -d "${CARELESS_DIR}/whisper_rt" ]] || { echo "[FATAL] no CarelessWhisper checkout at ${CARELESS_DIR}" >&2; exit 4; }

export MODEL="${MODEL:-small}"          # base | small | large-v2 | /path/to/finetuned.pt
export CHUNK_MS="${CHUNK_MS:-300}"      # released grans: 40/100/200/300/1000 (base lacks 1000)
export BEAMS="${BEAMS:-0}"              # 0 = greedy (word-lag available), >0 = beam
export OUT="${OUT:-/scratch/${USER}/experiments/whisper_asr/carelesswhisper_$(basename ${MODEL})_${CHUNK_MS}ms_beams${BEAMS}_$(basename $(dirname ${MANIFEST}))_hyps.tsv}"
EXTRA_ARGS="${EXTRA_ARGS:-}"            # e.g. "--rcs --extra-initial-blocks 2" or "--multilingual"

cd "${PROJECT_ROOT}"
export PYTHONNOUSERSITE=1
# whisper_rt.audio.load_audio's ffmpeg fallback needs it on PATH; the eval
# script's primary path (soundfile) avoids this for standard 16kHz wavs, but
# keep PATH correct for the fallback and for --rcs/other codepaths that hit
# load_audio directly. conda-forge ffmpeg installed into CW_ENV.
export PATH="${CW_ENV}/bin:${PATH}"

echo "Starting CarelessWhisper streaming eval at $(date) on $(hostname):"
echo "  MODEL=${MODEL} CHUNK_MS=${CHUNK_MS} BEAMS=${BEAMS} MANIFEST=${MANIFEST}"

"${CW_PY}" -m asr_baselines.carelesswhisper_streaming_eval \
  --model "${MODEL}" \
  --chunk-ms "${CHUNK_MS}" \
  --beams "${BEAMS}" \
  --manifest-tsv "${MANIFEST}" \
  --careless-dir "${CARELESS_DIR}" \
  --out "${OUT}" \
  ${EXTRA_ARGS}

echo "Finished at $(date)"
