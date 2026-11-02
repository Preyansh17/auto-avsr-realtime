#!/bin/bash
# Score one model on Tianyu's semg2 package sets (tianyu_eval.py), in every mode we compare at (PLAN.md 3.3 item 0, v2):
#   offline; unified: buffered streaming at 1.12 s (C0.56 R0.56), 0.56 s (C0.16 R0.40), 0.16 s (C0.08 R0.08), 0.08 s (C0.08 R0) latency;
#   nemotron_en: cache-aware at 80 ms (70,1) and 1.04 s (70,13) look-ahead, Tianyu's two phone modes.
#   bash teval.sh <model .nemo or HF name> <tag> <unified|nemotron_en> <out dir>     (env from the calling sbatch)
set -euo pipefail
M=$1; T=$2; BASE=$3; OUT=$4
PY=/scratch/pa2753/envs/nemo31/bin/python; E=/scratch/pa2753/unified/code/tianyu_eval.py
$PY $E --model $M --tag $T --out $OUT --mode offline
if [ "$BASE" = unified ]; then
  for CR in "0.56 0.56" "0.16 0.40" "0.08 0.08" "0.08 0.00"; do set -- $CR; $PY $E --model $M --tag $T --out $OUT --mode stream --chunk $1 --right $2; done
else
  for A in 70,1 70,13; do $PY $E --model $M --tag $T --out $OUT --mode cache --att $A; done
fi
