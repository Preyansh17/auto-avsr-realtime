#!/bin/bash
#SBATCH --job-name=whisper_buftrunc
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=32GB
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s_public
#SBATCH --output=/scratch/%u/auto-avsr/output/%x_%j.out
#SBATCH --account=torch_pr_39_general
set -euo pipefail

# Train-time buffer truncation, stacked on the confirmed warmup=25/wd=0.01/
# lr=1e-5/epochs=30 recipe. Unlike every other lever tried on this model (all
# of which move offline and streaming WER together), this one targets the
# streaming penalty itself: offline 7.62% vs streaming 11.66% on the same
# checkpoint. Training only ever shows the model whole utterances, while
# AlignAtt/SimulStreaming decodes from a growing partial buffer -- so with
# probability BUFFER_TRUNCATION_PROB an example is cut to a random prefix of
# its audio AND its target text is cut to the words that finished being
# spoken before the cut (minus a margin mirroring AlignAtt's hold-back).
#
# The target-side cut needs forced alignments: MFA TextGrids produced for the
# CarelessWhisper pipeline already cover this exact split 314/314, indexed by
# the joined CSV below. Truncating audio while keeping the full target would
# instead teach the model to hallucinate the unheard tail, so the trainer
# hard-errors if the prob is set without --aligned-csv.
#
# SEED in {1,2,3} must be set by the caller; TRUNC_PROB defaults to 0.5:
#   SEED=1 sbatch slurm/whisper_buftrunc_finetune.sbatch
#   SEED=1 TRUNC_PROB=0.3 sbatch slurm/whisper_buftrunc_finetune.sbatch

export PROJECT_ROOT="${PROJECT_ROOT:-/home/${USER}/auto-avsr-realtime}"
export WHISPER_ENV="${WHISPER_ENV:-/scratch/${USER}/envs/whisper_asr}"
export WHISPER_PY="${WHISPER_ENV}/bin/python"
export HF_HOME="${HF_HOME:-/scratch/${USER}/hf_cache}"
export TMPDIR="${TMPDIR:-/scratch/${USER}/.tmp}"
export PYTHONNOUSERSITE=1
mkdir -p "/scratch/${USER}/auto-avsr/output"

: "${SEED:?set SEED=1|2|3}"
TRUNC_PROB="${TRUNC_PROB:-0.5}"
MIN_PREFIX_SEC="${MIN_PREFIX_SEC:-1.0}"
PREFIX_MARGIN_SEC="${PREFIX_MARGIN_SEC:-0.5}"

SPLIT_ROOT="${SPLIT_ROOT:-/scratch/th3482/LipVideoData/patient_legal298_legacy96_split_v1}"
LABELS_OUT_DIR="${LABELS_OUT_DIR:-/scratch/${USER}/avsr_realtime/labels/splitv1}"
ROOT_DIR="${SPLIT_ROOT}"
TRAIN_FILE="${LABELS_OUT_DIR}/merged/train_spm1023.csv"
VAL_FILE="${LABELS_OUT_DIR}/merged/val_spm1023.csv"
TEST_FILE="${LABELS_OUT_DIR}/merged/test_spm1023.csv"
# Train-split alignments only -- val/test stay whole-utterance.
ALIGNED_CSV="${ALIGNED_CSV:-/scratch/${USER}/carelesswhisper/csv_splitv1_merged/train.csv}"
for f in "$TRAIN_FILE" "$VAL_FILE" "$TEST_FILE" "$ALIGNED_CSV"; do
  [[ -s "$f" ]] || { echo "[FATAL] missing $f" >&2; exit 3; }
done

SP_MODEL="${SP_MODEL:-${PROJECT_ROOT}/spm/spm_unigram_1023.model}"
EXP_DIR="${EXP_DIR:-/scratch/${USER}/experiments/whisper_asr}"
EXP_NAME="whisper_largev3_specaug_splitv1_merged_buftrunc_${TRUNC_PROB}_seed${SEED}"

echo "Starting at $(date) on $(hostname): SEED=${SEED} TRUNC_PROB=${TRUNC_PROB} EXP_NAME=${EXP_NAME}"
echo "TRAIN_FILE=${TRAIN_FILE} ($(wc -l < "$TRAIN_FILE") rows)"
echo "VAL_FILE=${VAL_FILE} ($(wc -l < "$VAL_FILE") rows)"
echo "TEST_FILE=${TEST_FILE} ($(wc -l < "$TEST_FILE") rows)"
echo "ALIGNED_CSV=${ALIGNED_CSV}"

cd "${PROJECT_ROOT}"
export SEED

"${WHISPER_PY}" -m asr_baselines.whisper_finetune \
  --model openai/whisper-large-v3 \
  --root-dir "${ROOT_DIR}" \
  --train-file "${TRAIN_FILE}" \
  --val-file "${VAL_FILE}" \
  --sp-model-path "${SP_MODEL}" \
  --output-dir "${EXP_DIR}/${EXP_NAME}" \
  --epochs 30 \
  --learning-rate 1e-5 \
  --warmup-steps 25 \
  --weight-decay 0.01 \
  --aligned-csv "${ALIGNED_CSV}" \
  --buffer-truncation-prob "${TRUNC_PROB}" \
  --min-prefix-sec "${MIN_PREFIX_SEC}" \
  --prefix-margin-sec "${PREFIX_MARGIN_SEC}" \
  --batch-size 8 \
  --grad-accum 2 \
  --precision bf16 \
  --unfreeze-encoder-layers -1 \
  --train-decoder \
  --num-workers 4

"${WHISPER_PY}" -m asr_baselines.whisper_eval \
  --model "${EXP_DIR}/${EXP_NAME}/best" \
  --root-dir "${ROOT_DIR}" \
  --label-file "${TEST_FILE}" \
  --sp-model-path "${SP_MODEL}" \
  --out "${EXP_DIR}/${EXP_NAME}/val_hyps.tsv"

echo "Finished at $(date)"
