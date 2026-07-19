#!/bin/bash
#SBATCH --job-name=decode_newpoint_3seed
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s_public
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/pa2753/auto-avsr/output/%x_%j.out
#SBATCH --account=torch_pr_39_general
set -euo pipefail

# Confirm the decode-sweep winner across all 3 wdwarmup seeds.
#
# The 9-point sweep on seed1 found segment=1.2s + frame_threshold=18 gives
# 8.97% streaming WER vs 11.66% at the long-standing 0.6s/ft=25 operating
# point -- decode-only, no retraining. seed1 alone isn't trustworthy on this
# data (see the speed-perturb reversal earlier), so re-decode all three
# already-converted checkpoints at both points for a like-for-like 3-seed
# comparison.

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
  for CFG in "1.2 18" "0.6 25"; do
    set -- $CFG
    SEG=$1; FT=$2
    echo "########## seed${SEED} segment=${SEG}s frame_threshold=${FT}"
    python -m asr_baselines.whisper_streaming_eval \
      --model-path "$CKPT" \
      --manifest-tsv "$DIR/val_hyps.tsv" \
      --simulstreaming-dir /scratch/pa2753/third_party/SimulStreaming \
      --segment-length "$SEG" \
      --frame-threshold "$FT" \
      --out "$DIR/streaming_hyps_seg${SEG}_ft${FT}.tsv"
  done
done

echo "Finished at $(date)"
