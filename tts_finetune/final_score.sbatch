#!/bin/bash
# Phase 4 PathBench scoring: one array task per items chunk written by `final_eval.py items --chunks N`
# (final/pb_items_00.csv ... ). Cheap pass (PER, ArtP, DPER, CPP, ...) lands in final/pb_out/chunk_NN/, the DArtP pass
# (DARTP=1, loads the 14.6 GB LM; needs the `lms` dir in pb_eval, which this script cd's into) in final/pb_out_dartp/.
# `final_eval.py table` merges both. Account/partition on the command line, spread the chunks over accounts if useful:
#   sbatch --account=<A> -p <P> --array=0-1 --export=ALL,FINAL=/scratch/pa2753/higgs_ft/final final_score.sbatch
#   sbatch --account=<B> -p <P> --array=2-2 --export=ALL,FINAL=/scratch/pa2753/higgs_ft/final final_score.sbatch
#   sbatch --account=<A> -p <P> --array=0-2 --time=01:15:00 --mem=64G --export=ALL,FINAL=...,DARTP=1 final_score.sbatch
# The neural half (TitaNet/WavLM) is the existing tts_eval/v2 pipeline: see final/submit_v2_ft.sh (final_eval.py items).
# --time is for ~220 clips per chunk without DArtP (PathBench runs at roughly 0.3-1 s per clip); use 01:15:00 with DARTP=1.
#SBATCH --job-name=ft_final_pb
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/pa2753/higgs_ft/logs/%x_%A_%a.out
#SBATCH --error=/scratch/pa2753/higgs_ft/logs/%x_%A_%a.out
set -euo pipefail
FINAL=${FINAL:-/scratch/pa2753/higgs_ft/final}
CHUNK=$(printf %02d "${SLURM_ARRAY_TASK_ID:-0}")
ITEMS="$FINAL/pb_items_$CHUNK.csv"
[ -f "$ITEMS" ] || { echo "missing $ITEMS (run: final_eval.py items --chunks N)" >&2; exit 2; }
if [ "${DARTP:-0}" = 1 ]; then OUT="$FINAL/pb_out_dartp/chunk_$CHUNK"; FLAG=--dartp; else OUT="$FINAL/pb_out/chunk_$CHUNK"; FLAG=; fi
export HF_HUB_OFFLINE=1
source /scratch/pa2753/pb_eval/pb_env.sh
cd /scratch/pa2753/pb_eval
echo "host=$(hostname) job=${SLURM_JOB_ID:-} items=$ITEMS out=$OUT dartp=${DARTP:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PB_PY" pb_score_items.py --items "$ITEMS" --out-dir "$OUT" $FLAG
echo DONE
