#!/bin/bash
#SBATCH --job-name=whisper_decode_sweep
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --partition=l40s_public
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/pa2753/auto-avsr/output/%x_%j.out
#SBATCH --account=torch_pr_39_general
set -euo pipefail

# Re-sweep the two AlignAtt decode knobs on the CURRENT best checkpoint.
#
# The standing conclusion -- "0.6s segments + frame_threshold=25 is optimal,
# and ~1.4-1.6s word lag is a structural floor" -- was established on the OLD
# pre-tune checkpoints. The recipe has since changed (warmup=25/wd=0.01 cut
# merged streaming WER 17.04% -> 11.96%), and a different model can want a
# different decode policy, so the conclusion is worth re-testing rather than
# inherited.
#
# Decode-only: no retraining, just the converted checkpoint re-decoded 9 ways.
#
# The grid is configurable so the search can be extended past its original
# edge (the first sweep topped out at 1.2s, which turned out to be the
# winner -- so the optimum may lie beyond it):
#
#   RUN=whisper_largev3_specaug_splitv1_merged_wdwarmup_seed1 \
#     sbatch slurm/whisper_decode_sweep.sbatch
#   SEGMENTS="1.8 2.4 3.0" FRAME_THRESHOLDS="14 18 22" \
#     sbatch slurm/whisper_decode_sweep.sbatch
#
# Note the degenerate limit: merged clips average ~5.1s, so once the segment
# approaches clip length the buffer is the whole utterance and this stops
# being streaming at all -- WER should converge to the offline number while
# TTFT converges to the clip duration.

export TMPDIR=/scratch/pa2753/.tmp
export HF_HOME=/scratch/pa2753/hf_cache
export PATH=/scratch/pa2753/envs/simulstream/bin:$PATH
export LD_LIBRARY_PATH=/scratch/pa2753/envs/simulstream/lib:${LD_LIBRARY_PATH:-}
export PYTHONNOUSERSITE=1

EXP=/scratch/pa2753/experiments/whisper_asr
RUN="${RUN:-whisper_largev3_specaug_splitv1_merged_wdwarmup_seed1}"
RUN_DIR="$EXP/$RUN"
TAG="${TAG:-splitv1_merged_wdwarmup_seed1}"
CKPT="$RUN_DIR/openai_format/large-v3-${TAG}.pt"
MANIFEST="$RUN_DIR/val_hyps.tsv"   # this run's held-out test set
[[ -f "$CKPT" ]] || { echo "[FATAL] no $CKPT" >&2; exit 3; }
[[ -s "$MANIFEST" ]] || { echo "[FATAL] no $MANIFEST" >&2; exit 3; }

SEGMENTS="${SEGMENTS:-0.3 0.6 1.2}"
FRAME_THRESHOLDS="${FRAME_THRESHOLDS:-12 18 25}"

cd /scratch/pa2753/wse_pkg
for SEG in $SEGMENTS; do
  for FT in $FRAME_THRESHOLDS; do
    echo "########## segment=${SEG}s frame_threshold=${FT}"
    python -m asr_baselines.whisper_streaming_eval \
      --model-path "$CKPT" \
      --manifest-tsv "$MANIFEST" \
      --simulstreaming-dir /scratch/pa2753/third_party/SimulStreaming \
      --segment-length "$SEG" \
      --frame-threshold "$FT" \
      --out "$RUN_DIR/streaming_hyps_seg${SEG}_ft${FT}.tsv"
  done
done

echo "Finished at $(date)"
