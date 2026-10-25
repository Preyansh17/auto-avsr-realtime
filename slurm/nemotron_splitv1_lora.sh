#!/bin/bash
# LoRA (bottleneck-adapter) Nemotron finetune on the audited
# patient_legal298_legacy96_split_v1 merged domain -- the parameter-efficient
# counterpart to slurm/nemotron_splitv1_best.sh's full finetune.
#
# Why this exists: --lora has been plumbed through nemotron_finetune.py and
# nemotron_finetune.sbatch since the start of the investigation, and loading a
# saved adapter checkpoint back was fixed and verified separately (see
# asr_baselines/nemotron_restore.py), but no LoRA run has ever been trained or
# scored on this split. There is no LoRA WER to compare against full-FT's
# 10.76% merged.
#
# What "LoRA" means here: NeMo's LinearAdapterConfig, a Houlsby-style
# bottleneck adapter (down-proj -> swish -> up-proj, zero-init'd output)
# inserted into every FastConformer encoder layer -- NOT a literal low-rank
# weight decomposition like online_avsr/lora.py. Everything else, INCLUDING
# the RNN-T decoder and joint, stays frozen. On dysarthric speech that
# freeze may bind harder than any hyperparameter: if every arm below lands
# far off full-FT, suspect the frozen decoder before the LR.
#
# Usage (from the repo root, on the torch cluster):
#   bash slurm/nemotron_splitv1_lora.sh                      # LR probe: 1e-4/5e-4/1e-3 at seed 1
#   LRS="5e-4" SEEDS="1 2 3" bash slurm/nemotron_splitv1_lora.sh   # 3 seeds at one LR
#   LORA_DIM=64 bash slurm/nemotron_splitv1_lora.sh          # wider adapter
#
# The default sweep is over LR because full-FT's 1e-4 is tuned for updating
# every weight in the model; an adapter carrying the whole domain shift on
# ~1% of the parameters usually wants more. Treat the probe's RANKING as
# provisional -- it is n=1 per arm, and the AV Emformer's LoRA runs varied
# ~24pp across seeds. Only the 3-seed follow-up at the winner decides
# anything.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/${USER}/auto-avsr-realtime}"
LABELS_DIR="${LABELS_DIR:-/scratch/${USER}/avsr_realtime/labels/splitv1}"
SPLIT_ROOT="${SPLIT_ROOT:-/scratch/th3482/LipVideoData/patient_legal298_legacy96_split_v1}"
ACCOUNT="${ACCOUNT:-torch_pr_39_tandon_advanced}"
EPOCHS="${EPOCHS:-210}"
SEEDS="${SEEDS:-1}"
LRS="${LRS:-1e-4 5e-4 1e-3}"
LORA_DIM="${LORA_DIM:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.0}"
# 1 -> keep the encoder frozen (adapters aside) but let the RNN-T decoder+joint
# train. See the flag's rationale in nemotron_finetune.py: the dim=32 LR probe
# came back flat across 1e-4/5e-4/1e-3, which points at the frozen decoder
# rather than the LR.
LORA_TRAIN_DECODER="${LORA_TRAIN_DECODER:-0}"
SPEED_PERTURB="${SPEED_PERTURB:-1}"
RESUME_CKPT="${RESUME_CKPT:-}"
# 2h rather than the sbatch's 6h default: shorter walltime backfills far
# sooner on this cluster, and the GPU-utilization monitor kills these jobs
# around 2h15 anyway -- so a 6h request buys nothing but queue time.
TIME_LIMIT="${TIME_LIMIT:-2:00:00}"
# Route explicitly across all three tandon GPU partitions. Left to its own
# devices SLURM sends tandon_advanced jobs to a100_tandon alone, which sits at
# or near its 60-GPU QOS group cap most of the time -- jobs then pend with
# QOSMaxGRESPerUser behind a queue that is not actually short. h200_tandon
# (cap 112) usually has the most headroom.
#
# Consequence to remember when reading results: arms can land on different GPU
# generations, so per-arm WALL-CLOCK is not comparable across arms (an earlier
# A/B in this investigation was invalidated exactly this way). WER is
# unaffected, which is all the LR probe is judged on.
PARTITIONS="${PARTITIONS:-h200_tandon,h100_tandon,a100_tandon}"

TRAIN_FILE="${LABELS_DIR}/merged_train_spm1023.csv"
VAL_FILE="${LABELS_DIR}/merged_val_spm1023.csv"
TEST_FILE="${LABELS_DIR}/merged_test_spm1023.csv"

for f in "${TRAIN_FILE}" "${VAL_FILE}" "${TEST_FILE}"; do
  [[ -s "${f}" ]] || { echo "[FATAL] missing ${f} -- run slurm/nemotron_splitv1_best.sh once to retokenize the split's labels to the 1023-piece SPM vocab" >&2; exit 3; }
done

SUFFIX=""
[[ "${SPEED_PERTURB}" == "1" ]] && SUFFIX="_speedperturb"
[[ "${LORA_TRAIN_DECODER}" == "1" ]] && SUFFIX="${SUFFIX}_dectrain"

for lr in ${LRS}; do
  for seed in ${SEEDS}; do
    name="nemotron_lora${LORA_DIM}_lr${lr}_epochs${EPOCHS}_splitv1merged_seed${seed}${SUFFIX}"
    # Auto-resume: a 210-epoch LoRA run does not fit the 2h walltime (~0.86
    # min/epoch measured), so re-running this script over an existing run picks
    # up its newest .ckpt instead of restarting from scratch. An explicit
    # RESUME_CKPT still wins.
    resume="${RESUME_CKPT}"
    if [[ -z "${resume}" ]]; then
      resume=$(ls -t "${EXP_DIR:-/scratch/${USER}/experiments/nemotron_asr}/${name}"/lightning_logs/version_*/checkpoints/*.ckpt 2>/dev/null | head -1 || true)
      [[ -n "${resume}" ]] && echo "  resuming ${name} from $(basename "${resume}")"
    fi
    sbatch --account="${ACCOUNT}" --time="${TIME_LIMIT}" --partition="${PARTITIONS}" \
      --export=ALL,RUN_DATA_MODE=merged,ROOT_DIR="${SPLIT_ROOT}",TRAIN_FILE="${TRAIN_FILE}",VAL_FILE="${VAL_FILE}",TEST_FILE="${TEST_FILE}",LORA=1,LORA_DIM="${LORA_DIM}",LORA_DROPOUT="${LORA_DROPOUT}",LORA_TRAIN_DECODER="${LORA_TRAIN_DECODER}",BATCH_SIZE=2,GRAD_ACCUM=2,LEARNING_RATE="${lr}",EPOCHS="${EPOCHS}",SEED="${seed}",SPEED_PERTURB="${SPEED_PERTURB}",RESUME_CKPT="${resume}",EXP_NAME="${name}" \
      "${PROJECT_ROOT}/slurm/nemotron_finetune.sbatch"
    echo "submitted ${name}"
  done
done
