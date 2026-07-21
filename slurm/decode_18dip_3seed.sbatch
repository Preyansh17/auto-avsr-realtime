#!/bin/bash
#SBATCH --job-name=decode_18dip_3seed
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s_public
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/pa2753/auto-avsr/output/%x_%j.out
#SBATCH --account=torch_pr_39_general
set -euo pipefail

# Is the 1.8s decode dip real or a seed1 fluke?
#
# On merged seed1 the WER-vs-segment curve was non-monotonic: 1.2s=8.97%,
# 1.8s=11.66%, 2.4s=8.97% (at ft=18) -- 1.8s worse than BOTH neighbours,
# consistently across ft 14/18/22, with no obvious mechanism. Single seed
# only. Re-decode all 3 merged wdwarmup checkpoints across the local curve
# (1.2 / 1.8 / 2.4s at ft=18) to see whether the dip reproduces.
# Decode-only, checkpoints already converted.

export TMPDIR=/scratch/pa2753/.tmp
export HF_HOME=/scratch/pa2753/hf_cache
export PATH=/scratch/pa2753/envs/simulstream/bin:$PATH
export LD_LIBRARY_PATH=/scratch/pa2753/envs/simulstream/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1

EXP=/scratch/pa2753/experiments/whisper_asr
cd /scratch/pa2753/wse_pkg

for SEED in 1 2 3; do
  TAG="splitv1_merged_wdwarmup_seed${SEED}"
  DIR="$EXP/whisper_largev3_specaug_${TAG}"
  CKPT="$DIR/openai_format/large-v3-${TAG}.pt"
  if [[ ! -f "$CKPT" ]]; then
    echo "########## SKIP seed${SEED}: no converted checkpoint at $CKPT"
    continue
  fi
  for SEG in 1.2 1.8 2.4; do
    echo "########## seed${SEED} segment=${SEG}s frame_threshold=18"
    python -m asr_baselines.whisper_streaming_eval \
      --model-path "$CKPT" \
      --manifest-tsv "$DIR/val_hyps.tsv" \
      --simulstreaming-dir /scratch/pa2753/third_party/SimulStreaming \
      --segment-length "$SEG" \
      --frame-threshold 18 \
      --out "$DIR/streaming_hyps_seg${SEG}_ft18.tsv"
  done
done

echo "Finished at $(date)"
