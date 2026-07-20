#!/bin/bash
#SBATCH --job-name=whisper_splitv1_best
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/%u/auto-avsr/output/%x_%j.out
#SBATCH --account=torch_pr_39_general
set -euo pipefail

# Reproduce the best Whisper result on split_v1, end to end, in one job:
# train -> convert to OpenAI format (bit-exact or abort) -> offline eval ->
# streaming eval at the best decode point.
#
# The recipe is documented in asr_baselines/configs/whisper_splitv1_best.yaml;
# the investigation behind it (including everything closed negative) is in
# results/whisper_streaming_investigation_2026-07-09.md.
#
# Expected on merged (3-seed means): offline 7.62%, streaming 8.97% at
# 1.2s/ft=18, RTF 0.170, TTFT p50 ~2.0s, word-lag p50 ~1.8s. Per-seed
# streaming at that decode point was 8.97/8.97/8.97 -- zero spread -- so a
# single seed reproducing near 8.97% is the expected outcome, not luck.
#
#   DOMAIN=merged SEED=1 sbatch slurm/whisper_splitv1_best.sbatch
#
# DOMAIN in {merged,legal,legacy}; merged is the headline result and the only
# one with high confidence (legacy's 10-clip test makes one word worth 2.5pp).

export PROJECT_ROOT="${PROJECT_ROOT:-/home/${USER}/auto-avsr-realtime}"
export WHISPER_ENV="${WHISPER_ENV:-/scratch/${USER}/envs/whisper_asr}"
export WHISPER_PY="${WHISPER_ENV}/bin/python"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export TMPDIR="${TMPDIR:-/scratch/${USER}/.tmp}"
export PYTHONNOUSERSITE=1
mkdir -p "/scratch/${USER}/auto-avsr/output"

DOMAIN="${DOMAIN:-merged}"
SEED="${SEED:-1}"
MODEL="${MODEL:-openai/whisper-large-v3}"
# Best decode point (see the config's `decode` block for the full Pareto set;
# 0.6/25 is the pick if ~0.5s of TTFT matters more than ~3pp of WER).
SEGMENT="${SEGMENT:-1.2}"
FRAME_THRESHOLD="${FRAME_THRESHOLD:-18}"

SPLIT_ROOT="${SPLIT_ROOT:-/scratch/th3482/LipVideoData/patient_legal298_legacy96_split_v1}"
LABELS_OUT_DIR="${LABELS_OUT_DIR:-/scratch/${USER}/avsr_realtime/labels/splitv1}"
SIMULSTREAMING_DIR="${SIMULSTREAMING_DIR:-/scratch/${USER}/third_party/SimulStreaming}"
CONVERT_DIR="${CONVERT_DIR:-/scratch/${USER}/third_party}"
ROOT_DIR="${SPLIT_ROOT}"
TRAIN_FILE="${LABELS_OUT_DIR}/${DOMAIN}/train_spm1023.csv"
VAL_FILE="${LABELS_OUT_DIR}/${DOMAIN}/val_spm1023.csv"
TEST_FILE="${LABELS_OUT_DIR}/${DOMAIN}/test_spm1023.csv"
# These must already be in the 1023-piece SPM vocab. The split ships
# unigram5000 ids in an identically-shaped CSV, which trains right up until
# sentencepiece raises "piece id is out of range" -- run
# scripts/regenerate_patient_labels.py first if these are missing.
for f in "$TRAIN_FILE" "$VAL_FILE" "$TEST_FILE"; do
  [[ -s "$f" ]] || { echo "[FATAL] missing $f -- run scripts/regenerate_patient_labels.py" >&2; exit 3; }
done

SP_MODEL="${SP_MODEL:-${PROJECT_ROOT}/spm/spm_unigram_1023.model}"
EXP_DIR="${EXP_DIR:-/scratch/${USER}/experiments/whisper_asr}"
TAG="splitv1_${DOMAIN}_best_seed${SEED}"
EXP_NAME="whisper_largev3_specaug_${TAG}"
RUN_DIR="${EXP_DIR}/${EXP_NAME}"
OUT_PT="${RUN_DIR}/openai_format/large-v3-${TAG}.pt"

echo "=== reproduce best Whisper result ==="
echo "DOMAIN=${DOMAIN} SEED=${SEED} MODEL=${MODEL}"
echo "decode: segment=${SEGMENT}s frame_threshold=${FRAME_THRESHOLD}"
echo "EXP_NAME=${EXP_NAME}"
echo "started $(date) on $(hostname)"

cd "${PROJECT_ROOT}"
export SEED

echo
echo "########## 1/4 train (warmup=25, weight_decay=0.01, lr=1e-5, 30 epochs)"
"${WHISPER_PY}" -m asr_baselines.whisper_finetune \
  --model "${MODEL}" \
  --root-dir "${ROOT_DIR}" \
  --train-file "${TRAIN_FILE}" \
  --val-file "${VAL_FILE}" \
  --sp-model-path "${SP_MODEL}" \
  --output-dir "${RUN_DIR}" \
  --epochs 30 \
  --learning-rate 1e-5 \
  --warmup-steps 25 \
  --weight-decay 0.01 \
  --batch-size 8 \
  --grad-accum 2 \
  --precision bf16 \
  --unfreeze-encoder-layers -1 \
  --train-decoder \
  --num-workers 4

echo
echo "########## 2/4 offline eval (held-out test)"
"${WHISPER_PY}" -m asr_baselines.whisper_eval \
  --model "${RUN_DIR}/best" \
  --root-dir "${ROOT_DIR}" \
  --label-file "${TEST_FILE}" \
  --sp-model-path "${SP_MODEL}" \
  --out "${RUN_DIR}/val_hyps.tsv"

echo
echo "########## 3/4 convert to OpenAI format (bit-exact or abort)"
mkdir -p "${RUN_DIR}/openai_format"
# A silently fp16-degraded conversion once produced pure hallucinations that
# looked like a model bug for days, hence the hard assert rather than a warning.
CONVERT_DIR="${CONVERT_DIR}" HF_CKPT="${RUN_DIR}/best" OUT_PT="${OUT_PT}" \
"${WHISPER_PY}" - <<'PYEOF'
import os, sys, torch
sys.path.insert(0, os.environ["CONVERT_DIR"])
from convert_hf_to_openai import convert_tfms_to_openai_whisper, reverse_rename_keys
from transformers import WhisperForConditionalGeneration

hf_ckpt, out_pt = os.environ["HF_CKPT"], os.environ["OUT_PT"]
convert_tfms_to_openai_whisper(hf_ckpt, out_pt)

hf = WhisperForConditionalGeneration.from_pretrained(hf_ckpt)
sd_hf = {k: v.cpu() for k, v in hf.state_dict().items()}
sd_hf = reverse_rename_keys(sd_hf)
del sd_hf["proj_out.weight"]
sd_hf = {k.replace("model.", "", 1): v for k, v in sd_hf.items()}
sd_new = torch.load(out_pt, map_location="cpu")["model_state_dict"]

key_mismatch = set(sd_hf) != set(sd_new)
worst = max((sd_hf[k] - sd_new[k]).abs().max().item()
            for k in sd_new if k in sd_hf) if not key_mismatch else float("inf")
dtypes = {str(v.dtype) for v in sd_new.values()}
ok = (not key_mismatch) and worst == 0.0 and dtypes == {"torch.float32"}
print(f"conversion: key_mismatch={key_mismatch} worst_diff={worst} dtypes={dtypes} bit_exact={ok}")
if not ok:
    print("ABORT: conversion not bit-exact -- refusing to trust this checkpoint")
    os.path.exists(out_pt) and os.remove(out_pt)
    sys.exit(1)
PYEOF

echo
echo "########## 4/4 streaming eval (segment=${SEGMENT}s ft=${FRAME_THRESHOLD})"
# SimulStreaming needs its own env (separate deps from the training env).
export PATH="/scratch/${USER}/envs/simulstream/bin:$PATH"
export LD_LIBRARY_PATH="/scratch/${USER}/envs/simulstream/lib:${LD_LIBRARY_PATH:-}"
cd "${WSE_PKG_DIR:-/scratch/${USER}/wse_pkg}"
python -m asr_baselines.whisper_streaming_eval \
  --model-path "${OUT_PT}" \
  --manifest-tsv "${RUN_DIR}/val_hyps.tsv" \
  --simulstreaming-dir "${SIMULSTREAMING_DIR}" \
  --segment-length "${SEGMENT}" \
  --frame-threshold "${FRAME_THRESHOLD}" \
  --out "${RUN_DIR}/streaming_hyps_seg${SEGMENT}_ft${FRAME_THRESHOLD}.tsv"

echo
echo "=== done $(date) ==="
echo "expected on merged: offline ~7.62%, streaming ~8.97%"
