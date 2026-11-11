#!/bin/bash
#SBATCH --job-name=t2m2_smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/scratch/pa2753/synvid/t2m2/logs/%x_%j.out
set -euo pipefail
SV=/scratch/pa2753/synvid
PY=$SV/envs/usr2/bin/python
cd $SV/t2m2/code
[ -f $SV/t2m2/g4/a/items.jsonl ] || /scratch/pa2753/envs/idx2/bin/python phonemize.py
[ -f $SV/t2m2/gen/pack.pt ] || $PY gen.py prep
export T2M2_G=$SV/t2m2/smoke/gen T2M2_G4=$SV/t2m2/smoke/g4
mkdir -p $T2M2_G $T2M2_G4/a
cp $SV/t2m2/gen/pack.pt $SV/t2m2/gen/durations.json $T2M2_G/
head -5 $SV/t2m2/g4/a/items.jsonl > $T2M2_G4/a/items.jsonl
time $PY gen.py ae --steps 300 --batch 256
time $PY gen.py gen --steps 400 --batch 16 --ctc-start 100 --ctc-ramp 100
$PY gen.py sample --items $T2M2_G4/a/items.jsonl --out $T2M2_G4/a --seed 1
$PY g4.py realref
$PY g4.py aperture --steps 300
cd $SV/code/usr2
$PY $SV/code/judge.py --rows $T2M2_G4/a/rows.jsonl --ckpt $SV/models/usr2/vsr_huge_hr.pth --backbone huge --out $T2M2_G4/a/judge.jsonl
cd $SV/t2m2/code
$PY g4.py score
nvidia-smi --query-gpu=memory.used --format=csv
