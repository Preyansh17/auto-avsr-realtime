#!/bin/bash
#SBATCH --job-name=avsr_select_best
#SBATCH --nodes=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=16GB
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/pa2753/auto-avsr/output/%x_%j.out
#SBATCH --error=/scratch/pa2753/auto-avsr/output/%x_%j.err
#SBATCH --mail-user=pa2753@nyu.edu
#SBATCH --mail-type=END,FAIL
#SBATCH --account=torch_pr_39_general

# Honest three-way checkpoint selection for the AV Emformer, using the metric
# that actually matters. val_loss (§4) and greedy-utterance val_wer (§5) were
# both previously shown NOT to track real streaming-beam WER on this recipe
# (val_wer=0.43 vs actual streaming WER ~60% in one documented case) -- so
# selecting on either would just honestly report a badly-chosen checkpoint.
# This instead decodes every candidate epoch checkpoint in ACTUAL STREAMING
# MODE (same chunked windowed decode as eval.py --mode streaming) against a
# selection set (SELECT_FILE), picks whichever epoch scores best there, then
# reports THAT checkpoint's streaming WER on a disjoint held-out TEST_FILE --
# a real, honest three-way split. Also merges+evaluates last.ckpt (the
# existing default, no selection at all) on the same TEST_FILE, so the two
# can be compared directly.
#
# Requires the training run to have been launched with AVG_LAST_N=<N> so
# Lightning kept recent-epoch=*.ckpt files to select among (see
# train_realtime_lora.sbatch's second ModelCheckpoint, monitor="epoch_idx").
#
#   RUN_DIR=<exp-dir>/<exp-name> SELECT_FILE=<val csv> TEST_FILE=<test csv> \
#     sbatch slurm/select_best_streaming_epoch.sbatch
#
# The inner container script is passed via a fully-literal quoted heredoc
# (no bash interpolation at all) and reads its config from environment
# variables instead -- singularity exec passes the host environment through
# by default here (verified), which sidesteps the fragile '"${VAR}"'
# close-quote/reopen-quote trick the other sbatch templates use (that trick
# broke here: a `${VAR:?message}` with an apostrophe in the message text
# collided with it in a way that was hard to debug by inspection -- simpler
# to avoid the whole class of bug than to get the quoting exactly right).

set -euo pipefail

export PROJECT_ROOT="${PROJECT_ROOT:-/home/pa2753/auto-avsr-realtime}"
export CONDA_ENV="${CONDA_ENV:-avsr_realtime}"
mkdir -p "/scratch/${USER}/auto-avsr/output"

export EVAL_MODE="${EVAL_MODE:-legal_only}"
RUN_DATA_MODE="${EVAL_MODE}" source "${PROJECT_ROOT}/slurm/_patient_data.sh"
export ROOT_DIR RAW_VAL_FILE

if [[ -z "${RUN_DIR:-}" ]]; then
  echo "[FATAL] set RUN_DIR to the training run's EXP_DIR/EXP_NAME" >&2
  exit 4
fi
export RUN_DIR
export SP_MODEL="${SP_MODEL:-${PROJECT_ROOT}/spm/spm_unigram_1023.model}"
export PREPROCESS="${PREPROCESS:-roi}"
export SELECT_FILE="${SELECT_FILE:-${VAL_FILE}}"
export TEST_FILE="${TEST_FILE:-${VAL_FILE}}"
export OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/select_sweep}"
export OVERLAY_EXT3="${OVERLAY_EXT3:-/scratch/pa2753/avsr/avsr.ext3}"
export SIF_IMAGE="${SIF_IMAGE:-/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.4.sif}"

mkdir -p "${OUTPUT_DIR}"

echo "Selecting best streaming-WER epoch in ${RUN_DIR} at $(date) on $(hostname)"
echo "SELECT_FILE=${SELECT_FILE}  TEST_FILE=${TEST_FILE}"
[[ "${DRY_RUN:-0}" == "1" ]] && { echo "DRY_RUN=1, not launching."; exit 0; }

INNER_SCRIPT="$(mktemp)"
cat > "${INNER_SCRIPT}" << 'INNEREOF'
set -euo pipefail
source /ext3/env.sh
conda activate "${CONDA_ENV}"
export SSL_CERT_FILE="${PROJECT_ROOT}/cacert.pem"
cd "${PROJECT_ROOT}"

[[ -s "${SELECT_FILE}" ]] || python scripts/regenerate_patient_labels.py --old-csv "${RAW_VAL_FILE}" --out "${SELECT_FILE}"
[[ -s "${TEST_FILE}" ]]   || python scripts/regenerate_patient_labels.py --old-csv "${RAW_VAL_FILE}" --out "${TEST_FILE}"

best_wer="999"
best_ckpt=""
n_ckpts=0

shopt -s nullglob
for ckpt in "${RUN_DIR}"/recent-epoch=*.ckpt; do
  n_ckpts=$((n_ckpts + 1))
  name=$(basename "$ckpt" .ckpt)
  merged="${OUTPUT_DIR}/${name}_merged.pth"
  python scripts/merge_lora_ckpt.py --ckpt "$ckpt" --out "$merged" --sp-model "${SP_MODEL}"
  outdir="${OUTPUT_DIR}/${name}_val"
  python eval.py --checkpoint "$merged" --mode streaming \
    --root-dir "${ROOT_DIR}" --test-file "${SELECT_FILE}" \
    --sp-model-path "${SP_MODEL}" --preprocess "${PREPROCESS}" \
    --output-dir "$outdir"
  wer=$(python scripts/read_eval_summary.py "${outdir}/eval_streaming_*/summary.json" corpus_wer)
  echo "[sweep] ${name} val_corpus_wer=${wer}"
  if awk -v w="$wer" -v b="$best_wer" 'BEGIN{exit !(w<b)}'; then
    best_wer="$wer"
    best_ckpt="$merged"
  fi
done

echo "n_candidate_ckpts=${n_ckpts}"
if [[ -z "$best_ckpt" ]]; then
  echo "[FATAL] no recent-epoch checkpoints found in ${RUN_DIR} -- was this run launched with AVG_LAST_N greater than 0?" >&2
  exit 4
fi
echo "BEST_ON_SELECTION_SET ckpt=${best_ckpt} val_corpus_wer=${best_wer}"

# Honest number: the selection-set-best checkpoint, evaluated on the DISJOINT held-out test set.
python eval.py --checkpoint "$best_ckpt" --mode streaming \
  --root-dir "${ROOT_DIR}" --test-file "${TEST_FILE}" \
  --sp-model-path "${SP_MODEL}" --preprocess "${PREPROCESS}" \
  --output-dir "${OUTPUT_DIR}/best_on_test"
echo "[REPORT] streaming-selected best-epoch, test set:"
python scripts/read_eval_summary.py "${OUTPUT_DIR}/best_on_test/eval_streaming_*/summary.json"

# Comparison: merge-last (existing default, no selection at all), same test set.
python scripts/merge_lora_ckpt.py --ckpt "${RUN_DIR}/last.ckpt" --out "${OUTPUT_DIR}/last_merged.pth" --sp-model "${SP_MODEL}"
python eval.py --checkpoint "${OUTPUT_DIR}/last_merged.pth" --mode streaming \
  --root-dir "${ROOT_DIR}" --test-file "${TEST_FILE}" \
  --sp-model-path "${SP_MODEL}" --preprocess "${PREPROCESS}" \
  --output-dir "${OUTPUT_DIR}/last_on_test"
echo "[REPORT] merge-last (no selection), test set:"
python scripts/read_eval_summary.py "${OUTPUT_DIR}/last_on_test/eval_streaming_*/summary.json"
INNEREOF

singularity exec --nv \
  --overlay "${OVERLAY_EXT3}:ro" \
  --writable-tmpfs "${SIF_IMAGE}" \
  /bin/bash -lc "bash '${INNER_SCRIPT}'"

rm -f "${INNER_SCRIPT}"
echo "Finished at $(date)"
