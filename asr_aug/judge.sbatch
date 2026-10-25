#!/bin/bash
# ASR judge for the synthetic filter: the real-only Whisper large-v3 best-recipe model (arm R, seed 1) decodes the
# real val clips (threshold calibration) and every synthetic clip. Then: python filter_synth.py score
#   sbatch --account=<A> -p l40s_public judge.sbatch
#SBATCH --job-name=aug_judge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/pa2753/asr_aug/logs/%x_%j.out
set -euo pipefail
PY=/scratch/pa2753/envs/whisper_asr/bin/python
export HF_HOME=/scratch/pa2753/hf_cache PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
JUDGE=${JUDGE:-/scratch/pa2753/experiments/whisper_asr/whisper_largev3_specaug_splitv1_merged_wdwarmup_seed1/best}
D=/scratch/pa2753/asr_aug/data
cd /home/pa2753/auto-avsr-realtime
$PY -m asr_baselines.whisper_eval --model "$JUDGE" --root-dir /scratch/pa2753/asr_aug/root \
  --label-file /scratch/pa2753/avsr_realtime/labels/splitv1/merged/val_spm1023.csv --out $D/judge_real_val.tsv
$PY -m asr_baselines.whisper_eval --model "$JUDGE" --root-dir /scratch/pa2753/asr_aug/root \
  --label-file $D/synth_all.csv --out $D/judge_synth.tsv
$PY /scratch/pa2753/asr_aug/code/filter_synth.py score
# arm R = the existing real-only best-recipe runs; re-decode val/test per utterance for the paired report
for s in 1 2 3; do
  R=/scratch/pa2753/asr_aug/runs/whisper_R_seed$s; mkdir -p $R
  ln -sfn /scratch/pa2753/experiments/whisper_asr/whisper_largev3_specaug_splitv1_merged_wdwarmup_seed$s/best $R/best
  for sp in val test; do
    $PY -m asr_baselines.whisper_eval --model $R/best --root-dir /scratch/pa2753/asr_aug/root \
      --label-file /scratch/pa2753/avsr_realtime/labels/splitv1/merged/${sp}_spm1023.csv --out $R/${sp}_hyps.tsv
  done
done
