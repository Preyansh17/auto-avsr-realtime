#!/bin/bash
# TORGO: one arm x seed with the best Nemotron recipe (slurm/nemotron_splitv1_best.sh: full FT, lr 1e-4,
# batch 2 x accum 2, speed perturbation), resumable. The cluster's GPU-utilisation monitor kills these runs after
# ~2h of low (<50%) average GPU use, so links stop at 1:55 (TIMEOUT, then resume) and submit a chain of these jobs with --dependency=afterany: each link resumes from the newest intact
# Lightning .ckpt and exits at once if the run already finished (DONE marker).
#   torgo/nemotron_launch.sh ARM SEED EPOCHS ACCOUNT [LINKS]. EPOCHS=39 ~ patient R's 49.5k optimizer steps on 1693 clips.
#SBATCH --job-name=torgo_nemo
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24GB
#SBATCH --gres=gpu:1
#SBATCH --time=01:55:00
#SBATCH --output=/scratch/pa2753/torgo_aug/logs/%x_%j.out
set -euo pipefail
: "${ARM:?}" "${SEED:?}" "${EPOCHS:?}"
OUT=/scratch/pa2753/torgo_aug/runs/nemotron_${ARM}_seed${SEED}
mkdir -p "$OUT"
[[ -f "$OUT/DONE" ]] && { echo "already done"; exit 0; }
[[ -s "$OUT/train.csv" ]] || /scratch/pa2753/envs/nemo_asr/bin/python /scratch/pa2753/torgo_aug/code/make_mix.py \
  --arm "$ARM" --seed "$SEED" --out "$OUT/train.csv"
# newest .ckpt that is not truncated (a SIGTERM mid-write leaves a short file): require >= 90% of the largest
RESUME_CKPT=$(python3 - "$OUT" <<'PY'
import glob, os, sys
c = glob.glob(f"{sys.argv[1]}/lightning_logs/*/checkpoints/*.ckpt")
if c:
    big = max(os.path.getsize(p) for p in c)
    ok = [p for p in c if os.path.getsize(p) >= 0.9 * big]
    print(max(ok, key=os.path.getmtime))
PY
)
# Nodes now ship CUDA 13.3 at /usr/local/cuda; numba-cuda (RNNT loss) compiles with that libnvvm, but the env's pip
# nvJitLink is 13.0 -> "ERROR_OUTDATED_LIBRARY" at epoch 0. Preloading the system nvJitLink keeps them in step.
JL=$(ls /usr/local/cuda/lib64/libnvJitLink.so.13 2>/dev/null || true)
[[ -n "$JL" ]] && export LD_PRELOAD="$JL${LD_PRELOAD:+:$LD_PRELOAD}"
echo "ARM=$ARM SEED=$SEED EPOCHS=$EPOCHS LD_PRELOAD=${LD_PRELOAD:-} resume=${RESUME_CKPT:-none} job=${SLURM_JOB_ID:-} host=$(hostname) $(date)"
L=/scratch/pa2753/torgo_aug/data
export RUN_DATA_MODE=merged ROOT_DIR=/scratch/pa2753/torgo_aug/root TRAIN_FILE="$OUT/train.csv" \
  VAL_FILE=$L/val_spm1023.csv TEST_FILE=$L/test_spm1023.csv UNFREEZE_ENCODER_LAYERS=-1 TRAIN_DECODER=1 \
  BATCH_SIZE=2 GRAD_ACCUM=2 LEARNING_RATE=1e-4 LR_SCHEDULE=${LR_SCHEDULE:-noam} EPOCHS SEED SPEED_PERTURB=1 RESUME_CKPT \
  EXP_DIR=/scratch/pa2753/torgo_aug/runs EXP_NAME=nemotron_${ARM}_seed${SEED} NUM_WORKERS=4
bash /home/pa2753/auto-avsr-realtime/slurm/nemotron_finetune.sbatch
# nemotron_finetune.sbatch cd's into the repo in its own process; this shell is still in the sbatch dir
(cd /home/pa2753/auto-avsr-realtime && /scratch/pa2753/envs/nemo_asr/bin/python -m asr_baselines.nemotron_eval \
  --model "$OUT/nemotron_best.nemo" --manifest "$OUT/val_manifest.jsonl" --out "$OUT/val_hyps.tsv")
[[ -s "$OUT/val_hyps.tsv" ]] || { echo "val decode failed"; exit 1; }
for sp in val test; do  # val-free secondary: final-epoch model
  (cd /home/pa2753/auto-avsr-realtime && /scratch/pa2753/envs/nemo_asr/bin/python -m asr_baselines.nemotron_eval \
    --model "$OUT/nemotron_patient.nemo" --manifest "$OUT/${sp}_manifest.jsonl" --out "$OUT/${sp}_hyps_last.tsv")
  [[ -s "$OUT/${sp}_hyps_last.tsv" ]] || { echo "last decode failed"; exit 1; }
done
touch "$OUT/DONE"
echo "done $(date)"
