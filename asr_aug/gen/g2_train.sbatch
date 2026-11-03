#!/bin/bash
# G2 training (g2_train.py; asr_aug/gen/README.md amendment 3): repo recipe (20 epochs, batch 2), resumable.
# Builds the run dir once: model/ = IndexTTS-1.5 files (symlinks) + checkpoints, g2.yaml = repo config, num_classes 8.
#   sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint=b200 g2_train.sbatch
#   SMOKE=1: 300 batches, no validation, no saved weights (speed check).
#SBATCH --job-name=g2_train
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96GB
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/pa2753/gen/logs/%x_%j.out
#SBATCH --open-mode=append
set -euo pipefail
G=/scratch/pa2753/gen; R=$G/g2/run; M=$G/models/IndexTTS-1.5; PY=/scratch/pa2753/envs/pdis/bin/python
REPO=$G/code/Controllable-Dysarthric-Speech-Synthesis
export HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
echo "host=$(hostname) job=${SLURM_JOB_ID:-} restart=${SLURM_RESTART_COUNT:-0} $(nvidia-smi --query-gpu=name --format=csv,noheader) $(date)"
mkdir -p $R/model
for f in bpe.model dvae.pth gpt.pth bigvgan_generator.pth; do ln -sfn $M/$f $R/model/$f; done
[ -s $R/g2.yaml ] || $PY -c "
from omegaconf import OmegaConf
c = OmegaConf.load('$REPO/configs/controllable_dysarthric_speech_synthesis.yaml'); c.pathology.num_classes = 8
OmegaConf.save(c, '$R/g2.yaml')"
( while sleep 300; do echo "[gpu] $(date +%H:%M) $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader)"; done ) &
MON=$!
if [ "${SMOKE:-0}" = 1 ]; then EXTRA="--max-train-batches 300 --skip-validation --no-save --epochs 1"; else EXTRA="--epochs 20"; fi
$PY $G/code/g2_train.py --config $R/g2.yaml --model-dir $R/model --data-dir $G/g2/prepared --batch-size 2 \
  --num-workers 4 $EXTRA
kill $MON 2>/dev/null || true
echo "done $(date)"
