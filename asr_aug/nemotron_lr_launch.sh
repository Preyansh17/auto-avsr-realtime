#!/bin/bash
# Submit a resume chain for one Nemotron LR-probe run:
#   nemotron_lr_launch.sh TAG LR LR_SCHEDULE EPOCHS SEED ACCOUNT [LINKS=3]
set -euo pipefail
TAG=$1 LR=$2 SCHED=$3 EPOCHS=$4 SEED=$5 ACC=$6 LINKS=${7:-3}
mkdir -p /scratch/pa2753/nemo_lr/logs /scratch/pa2753/nemo_lr/runs
dep=""
for i in $(seq 1 "$LINKS"); do
  j=$(sbatch --parsable --account="$ACC" --job-name="nemo_lr_${TAG}_s${SEED}" $dep \
      --export=ALL,TAG="$TAG",LR="$LR",LR_SCHEDULE="$SCHED",EPOCHS="$EPOCHS",SEED="$SEED" \
      /scratch/pa2753/asr_aug/code/nemotron_lr.sbatch)
  dep="--dependency=afterany:$j"
  echo "nemo_lr $TAG lr=$LR $SCHED ep=$EPOCHS s$SEED link$i $j $ACC" | tee -a /scratch/pa2753/nemo_lr/jobs.txt
done
