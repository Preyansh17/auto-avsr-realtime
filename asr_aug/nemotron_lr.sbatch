#!/bin/bash
# Nemotron real-only (R) run at a REAL learning rate (2026-10-20). Every earlier Nemotron run trained under the
# pretrained NoamAnnealing schedule, whose effective LR was a flat 1e-6 whatever LEARNING_RATE said. Same recipe as
# nemotron_aug.sbatch (full FT, batch 2 x accum 2, speed perturbation, SpecAugment) except LR_SCHEDULE / LR / EPOCHS.
# Resumable links of 1:55 (GPU-util monitor); decodes best.nemo and the final-epoch model on val and test at the end.
#   nemotron_lr_launch.sh TAG LR LR_SCHEDULE EPOCHS SEED ACCOUNT [LINKS]
#SBATCH --job-name=nemo_lr
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --time=01:55:00
#SBATCH --output=/scratch/pa2753/nemo_lr/logs/%x_%j.out
set -euo pipefail
: "${TAG:?}" "${LR:?}" "${LR_SCHEDULE:?}" "${EPOCHS:?}" "${SEED:?}"
ARM=${ARM:-R}
RUNS=/scratch/pa2753/nemo_lr/runs
OUT=$RUNS/nemotron_${ARM}_${TAG}_seed${SEED}
PY=/scratch/pa2753/envs/nemo_asr/bin/python
mkdir -p "$OUT"
[[ -f "$OUT/DONE" ]] && { echo "already done"; exit 0; }
[[ -s "$OUT/train.csv" ]] || $PY /scratch/pa2753/asr_aug/code/make_mix.py --arm "$ARM" --seed "$SEED" --out "$OUT/train.csv"
RESUME_CKPT=$(python3 - "$OUT" <<'PY'
import glob, os, sys
c = glob.glob(f"{sys.argv[1]}/lightning_logs/*/checkpoints/*.ckpt")
if c:
    big = max(os.path.getsize(p) for p in c)
    ok = [p for p in c if os.path.getsize(p) >= 0.9 * big]
    print(max(ok, key=os.path.getmtime))
PY
)
JL=$(ls /usr/local/cuda/lib64/libnvJitLink.so.13 2>/dev/null || true)
[[ -n "$JL" ]] && export LD_PRELOAD="$JL${LD_PRELOAD:+:$LD_PRELOAD}"
echo "TAG=$TAG ARM=$ARM LR=$LR LR_SCHEDULE=$LR_SCHEDULE EPOCHS=$EPOCHS SEED=$SEED resume=${RESUME_CKPT:-none} job=${SLURM_JOB_ID:-} host=$(hostname) $(date)"
L=/scratch/pa2753/avsr_realtime/labels/splitv1/merged
export RUN_DATA_MODE=merged ROOT_DIR=/scratch/pa2753/asr_aug/root TRAIN_FILE="$OUT/train.csv" \
  VAL_FILE=$L/val_spm1023.csv TEST_FILE=$L/test_spm1023.csv UNFREEZE_ENCODER_LAYERS=-1 TRAIN_DECODER=1 \
  BATCH_SIZE=2 GRAD_ACCUM=2 LEARNING_RATE="$LR" LR_SCHEDULE EPOCHS SEED SPEED_PERTURB=1 RESUME_CKPT \
  EXP_DIR=$RUNS EXP_NAME=nemotron_${ARM}_${TAG}_seed${SEED} NUM_WORKERS=4
bash /home/pa2753/auto-avsr-realtime/slurm/nemotron_finetune.sbatch
cd /home/pa2753/auto-avsr-realtime
for s in val test; do
  [[ -s "$OUT/nemotron_best.nemo" ]] && [[ $s == val ]] && $PY -m asr_baselines.nemotron_eval \
    --model "$OUT/nemotron_best.nemo" --manifest "$OUT/${s}_manifest.jsonl" --out "$OUT/${s}_hyps.tsv"
  $PY -m asr_baselines.nemotron_eval --model "$OUT/nemotron_patient.nemo" --manifest "$OUT/${s}_manifest.jsonl" \
    --out "$OUT/${s}_hyps_last.tsv"
done
[[ -s "$OUT/test_hyps.tsv" && -s "$OUT/test_hyps_last.tsv" && -s "$OUT/val_hyps_last.tsv" ]] || { echo "decode failed"; exit 1; }
rm -rf "$OUT"/lightning_logs/*/checkpoints
touch "$OUT/DONE"
echo "done $(date)"
