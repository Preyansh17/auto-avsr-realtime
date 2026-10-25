#!/bin/bash
# Zero-shot whisper-large-v3 judge for the TORGO synthetic filter.
#   STAGE=real : decode real train (threshold calibration) and real test (zero-shot reference point)
#   STAGE=synth: decode data/synth_all.csv, then filter_synth.py score
#SBATCH --job-name=torgo_judge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/pa2753/torgo_aug/logs/%x_%j.out
set -euo pipefail
: "${STAGE:?}"
PY=/scratch/pa2753/envs/whisper_asr/bin/python
export HF_HOME=/scratch/pa2753/hf_cache PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
D=/scratch/pa2753/torgo_aug/data
R=/scratch/pa2753/torgo_aug/root
cd /home/pa2753/auto-avsr-realtime
if [[ $STAGE == real ]]; then
  $PY -m asr_baselines.whisper_eval --model openai/whisper-large-v3 --root-dir $R --label-file $D/train_spm1023.csv --out $D/judge_real_train.tsv
  $PY -m asr_baselines.whisper_eval --model openai/whisper-large-v3 --root-dir $R --label-file $D/test_spm1023.csv --out $D/zeroshot_test.tsv
else
  $PY -m asr_baselines.whisper_eval --model openai/whisper-large-v3 --root-dir $R --label-file $D/synth_all.csv --out $D/judge_synth.tsv
  $PY /scratch/pa2753/torgo_aug/code/filter_synth.py score
fi
echo done
