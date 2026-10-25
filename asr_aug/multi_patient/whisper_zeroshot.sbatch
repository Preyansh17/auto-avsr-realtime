#!/bin/bash
# Zero-shot Whisper large-v3 on semg1(oct1)/semg2 val/test/oov.
#SBATCH --job-name=mp_zeroshot
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/pa2753/multi_patient/logs/%x_%j.out
set -euo pipefail
PY=/scratch/pa2753/envs/whisper_asr/bin/python
export HF_HOME=/scratch/pa2753/hf_cache PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
M=/scratch/pa2753/multi_patient; mkdir -p $M/runs/zeroshot; cd /home/pa2753/auto-avsr-realtime
for p in s1 s2; do for s in val test oov; do
  $PY -m asr_baselines.whisper_eval --model openai/whisper-large-v3 --root-dir $M/root --label-file $M/labels/${p}_${s}_spm1023.csv --out $M/runs/zeroshot/${p}_${s}_hyps.tsv
done; done
