#!/bin/bash
#SBATCH --job-name=t2m2_g4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/pa2753/synvid/t2m2/logs/%x_%j.out
# Gate G4: sample sets a/b/d with the trained generator, judge (USR 2.0 Huge, beam 40), real heldout reference, score.
set -euo pipefail
SV=/scratch/pa2753/synvid
PY=$SV/envs/usr2/bin/python
G4=$SV/t2m2/g4
cd $SV/t2m2/code
for s in a b d; do
  [ -f $G4/$s/rows.jsonl ] || $PY gen.py sample --items $G4/$s/items.jsonl --out $G4/$s --seed 1
done
[ -f $G4/real_a/rows.jsonl ] || $PY g4.py realref
[ -f $G4/apnet.pth ] || $PY g4.py aperture
cd $SV/code/usr2
for s in a b d real_a; do
  [ -f $G4/$s/judge.jsonl ] || $PY $SV/code/judge.py --rows $G4/$s/rows.jsonl --ckpt $SV/models/usr2/vsr_huge_hr.pth \
    --backbone huge --out $G4/$s/judge.jsonl
done
cd $SV/t2m2/code
$PY g4.py score
