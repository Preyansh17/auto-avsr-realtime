#!/bin/bash
#SBATCH --job-name=a18_gen
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/pa2753/synvid/a18/logs/%x_%j.out
#SBATCH --open-mode=append
# A18: sample generated clips for the extra DailyDialog sentences (seed 1) and S re-samples (seed k), with gen.pth.
set -eo pipefail
SV=/scratch/pa2753/synvid; A=$SV/a18; PY=$SV/envs/usr2/bin/python
export PYTHONNOUSERSITE=1
cd $SV/t2m2/code
[ -f $A/x/rows.jsonl ] || $PY gen.py sample --items $A/x/items.jsonl --out $A/x --seed 1
for d in $A/r*/; do s=$(basename $d); s=${s#r}
  [ -f $d/rows.jsonl ] || $PY gen.py sample --items $d/items.jsonl --out $d --seed $s
done
echo a18 gen done
