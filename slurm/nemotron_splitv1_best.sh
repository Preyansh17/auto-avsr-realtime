#!/bin/bash
# Reproduce the best Nemotron config found so far: full-FT, merged domain,
# 210 epochs + speed perturbation, on the audited
# patient_legal298_legacy96_split_v1 split. See
# asr_baselines/configs/nemotron_splitv1_best.yaml for the recipe summary and
# results/nemotron_splitv1_epoch_sweep_2026-07-18.md for the full writeup
# (epoch sweep, speed perturbation, cross-domain eval, open items). Speed
# perturbation has only been confirmed on merged offline WER so far -- no
# cross-domain/streaming/beam battery yet. If legacy-domain accuracy matters
# more than merged/legal, use EPOCHS=180 SPEED_PERTURB=0 instead -- see the
# writeup's "210 vs 240" section for the tradeoff.
#
# Usage (from the repo root, on the torch cluster):
#   bash slurm/nemotron_splitv1_best.sh                    # submits seeds 1,2,3, speed perturbation on
#   SEEDS="1" bash slurm/nemotron_splitv1_best.sh          # submits just seed 1
#   SPEED_PERTURB=0 bash slurm/nemotron_splitv1_best.sh    # reproduce plain epochs=210 (no speed perturbation)
#   EPOCHS=180 SPEED_PERTURB=0 bash slurm/nemotron_splitv1_best.sh   # reproduce a different point on the sweep
#
# Speed perturbation roughly doubles training wall-clock (per-sample
# resampling cost) -- expect to need RESUME_CKPT more than once per seed.
#
# Long runs (200+ epochs) on this cluster carry a real risk of being killed
# by an automated GPU-utilization monitor after ~2h. If that happens,
# resubmit with RESUME_CKPT=<path to the .ckpt under
# EXP_DIR/<exp_name>/lightning_logs/version_<killed_jobid>/checkpoints/> --
# nemotron_finetune.py picks up optimizer/scheduler/epoch state from there
# instead of restarting. See the writeup's "Checkpoint resume" section.
#
# Each seed's post-training eval (automatic, inside nemotron_finetune.sbatch)
# reports in-domain merged offline greedy WER only. For the full cross-domain
# (legal/legacy) + streaming + beam picture reported in the writeup above,
# run nemotron_eval.py / nemotron_streaming_eval.py separately against the
# resulting checkpoint -- see the writeup's "Full cross-domain / streaming /
# beam battery" section for the exact manifests and commands used.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/${USER}/auto-avsr-realtime}"
LABELS_DIR="${LABELS_DIR:-/scratch/${USER}/avsr_realtime/labels/splitv1}"
SPLIT_ROOT="${SPLIT_ROOT:-/scratch/th3482/LipVideoData/patient_legal298_legacy96_split_v1}"
NEMO_ENV="${NEMO_ENV:-/scratch/${USER}/envs/nemo_asr}"
ACCOUNT="${ACCOUNT:-torch_pr_39_tandon_advanced}"
EPOCHS="${EPOCHS:-210}"
SEEDS="${SEEDS:-1 2 3}"
RESUME_CKPT="${RESUME_CKPT:-}"
SPEED_PERTURB="${SPEED_PERTURB:-1}"

TRAIN_FILE="${LABELS_DIR}/merged_train_spm1023.csv"
VAL_FILE="${LABELS_DIR}/merged_val_spm1023.csv"
TEST_FILE="${LABELS_DIR}/merged_test_spm1023.csv"

# One-time retokenization: split_v1's own label CSVs carry unigram5000 token
# ids, not Nemotron's 1023-piece SPM vocab. Idempotent -- skips any file that
# already exists.
mkdir -p "${LABELS_DIR}"
for domain in legal legacy merged; do
  for split in train val test; do
    out="${LABELS_DIR}/${domain}_${split}_spm1023.csv"
    [[ -s "${out}" ]] && continue
    "${NEMO_ENV}/bin/python" "${PROJECT_ROOT}/scripts/regenerate_patient_labels.py" \
      --old-csv "${SPLIT_ROOT}/labels/${domain}/${split}.csv" \
      --out "${out}"
  done
done

# NOTE: slurm/_patient_data.sh's `merged` branch unconditionally resets
# ROOT_DIR to its own MERGED_ROOT default, ignoring the override below --
# harmless here only because that default's symlinks point at the same
# source directories as split_v1's, confirmed by inspection. The printed
# ROOT_DIR= line in the job log will show MERGED_ROOT, not SPLIT_ROOT; that
# is expected, not a bug in this script.
SUFFIX=""
[[ "${SPEED_PERTURB}" == "1" ]] && SUFFIX="_speedperturb"

for seed in ${SEEDS}; do
  sbatch --account="${ACCOUNT}" \
    --export=ALL,RUN_DATA_MODE=merged,ROOT_DIR="${SPLIT_ROOT}",TRAIN_FILE="${TRAIN_FILE}",VAL_FILE="${VAL_FILE}",TEST_FILE="${TEST_FILE}",UNFREEZE_ENCODER_LAYERS=-1,TRAIN_DECODER=1,BATCH_SIZE=2,GRAD_ACCUM=2,LEARNING_RATE=1e-4,EPOCHS="${EPOCHS}",SEED="${seed}",SPEED_PERTURB="${SPEED_PERTURB}",RESUME_CKPT="${RESUME_CKPT}",EXP_NAME="nemotron_fullft_epochs${EPOCHS}_splitv1merged_seed${seed}${SUFFIX}" \
    "${PROJECT_ROOT}/slurm/nemotron_finetune.sbatch"
done
