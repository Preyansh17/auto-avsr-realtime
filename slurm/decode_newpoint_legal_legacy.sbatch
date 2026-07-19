#!/bin/bash
#SBATCH --job-name=decode_newpoint_ll
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s_public
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/pa2753/auto-avsr/output/%x_%j.out
#SBATCH --account=torch_pr_39_general
set -euo pipefail

# Does the decode-sweep winner generalize off the merged domain?
#
# On merged, segment=1.2s/ft=18 gives 8.97% streaming WER on all 3 seeds vs
# 11.96% mean (8.07-16.14% spread) at the old 0.6s/ft=25 point. Legal and
# legacy were only ever evaluated at 0.6s/ft=25 (legal 3-seed 11.66%, legacy
# 32.50%), so re-decode their tuned checkpoints at both points.
# Decode-only, no retraining -- checkpoints already converted.

export TMPDIR=/scratch/pa2753/.tmp
export HF_HOME=/scratch/pa2753/hf_cache
export PATH=/scratch/pa2753/envs/simulstream/bin:$PATH
export LD_LIBRARY_PATH=/scratch/pa2753/envs/simulstream/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1

EXP=/scratch/pa2753/experiments/whisper_asr
cd /scratch/pa2753/wse_pkg

for DOMAIN in legal legacy; do
  for SEED in 1 2 3; do
    TAG="splitv1_${DOMAIN}_wdwarmup_seed${SEED}"
    DIR="$EXP/whisper_largev3_specaug_${TAG}"
    CKPT="$DIR/openai_format/large-v3-${TAG}.pt"
    if [[ ! -f "$CKPT" ]]; then
      echo "########## SKIP ${DOMAIN} seed${SEED}: no converted checkpoint at $CKPT"
      continue
    fi
    for CFG in "1.2 18" "0.6 25"; do
      set -- $CFG
      SEG=$1; FT=$2
      echo "########## ${DOMAIN} seed${SEED} segment=${SEG}s frame_threshold=${FT}"
      python -m asr_baselines.whisper_streaming_eval \
        --model-path "$CKPT" \
        --manifest-tsv "$DIR/val_hyps.tsv" \
        --simulstreaming-dir /scratch/pa2753/third_party/SimulStreaming \
        --segment-length "$SEG" \
        --frame-threshold "$FT" \
        --out "$DIR/streaming_hyps_seg${SEG}_ft${FT}.tsv"
    done
  done
done

echo "Finished at $(date)"
