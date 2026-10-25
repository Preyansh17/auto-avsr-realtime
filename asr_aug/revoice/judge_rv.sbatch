#!/bin/bash
# ASR judge for the re-voicing filter: real-only Whisper large-v3 (arm R, seed 1) decodes real ASR val (threshold
# calibration) and every re-voiced clip; then filter_rv.py score. Runs filter_rv.py collect first if rv_all.csv is missing.   sbatch --account=<A> judge_rv.sbatch
#SBATCH --job-name=rv_judge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/pa2753/asr_aug/revoice/logs/%x_%j.out
set -euo pipefail
PY=/scratch/pa2753/envs/whisper_asr/bin/python
export HF_HOME=/scratch/pa2753/hf_cache PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
JUDGE=/scratch/pa2753/asr_aug/runs/whisper_R_seed1/best
D=/scratch/pa2753/asr_aug/revoice/data
[[ -s $D/rv_all.csv ]] || $PY /scratch/pa2753/asr_aug/revoice/code/filter_rv.py collect
cd /home/pa2753/auto-avsr-realtime
$PY -m asr_baselines.whisper_eval --model "$JUDGE" --root-dir /scratch/pa2753/asr_aug/root \
  --label-file /scratch/pa2753/avsr_realtime/labels/splitv1/merged/val_spm1023.csv --out $D/judge_real_val.tsv
$PY -m asr_baselines.whisper_eval --model "$JUDGE" --root-dir /scratch/pa2753/asr_aug/root \
  --label-file $D/rv_all.csv --out $D/judge_rv.tsv
$PY /scratch/pa2753/asr_aug/revoice/code/filter_rv.py score
