#!/bin/bash
# Submit a resume chain for one Nemotron augmentation arm x seed: nemotron_launch.sh ARM SEED EPOCHS ACCOUNT [LINKS=5]
set -euo pipefail
ARM=$1 SEED=$2 EPOCHS=$3 ACC=$4 LINKS=${5:-5}
dep=""
for i in $(seq 1 "$LINKS"); do
  j=$(sbatch --parsable --account="$ACC" -p l40s_public --job-name="aug_n_${ARM}_s${SEED}" $dep \
      --export=ALL,ARM="$ARM",SEED="$SEED",EPOCHS="$EPOCHS" /scratch/pa2753/asr_aug/code/nemotron_aug.sbatch)
  dep="--dependency=afterany:$j"
  echo "nemotron $ARM s$SEED link$i $j $ACC" | tee -a /scratch/pa2753/asr_aug/jobs.txt
done
