#!/bin/bash
# ASR judge for the stack synthetic filter: each patient's synthetic clips are decoded by that patient's real-only
# finetuned large-v3 (s1: asr_aug oct1_N seed 1; s2: multi_patient s2 lang_task seed 1). Then filter_synth.py score.
#   sbatch --account=<A> -p l40s_public judge.sbatch
#SBATCH --job-name=stack_judge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/pa2753/stack/logs/%x_%j.out
set -euo pipefail
PY=/scratch/pa2753/envs/whisper_asr/bin/python
export HF_HOME=/scratch/pa2753/hf_cache PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
D=/scratch/pa2753/stack/data
declare -A J=([s1]=/scratch/pa2753/asr_aug/runs/oct1_N_seed1/best [s2]=/scratch/pa2753/multi_patient/runs/whisper_s2_lang_task_seed1/best)
cd /home/pa2753/auto-avsr-realtime
for p in s1 s2; do
  $PY -m asr_baselines.whisper_eval --model "${J[$p]}" --root-dir /scratch/pa2753/stack/root \
    --label-file $D/synth_all_$p.csv --out $D/judge_synth_$p.tsv --prefix lang_task
done
$PY /scratch/pa2753/stack/code/filter_synth.py score
