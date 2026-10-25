#!/bin/bash
# Submit Whisper arm x seed runs in LANES parallel dependency chains, so at most LANES runs hold checkpoints at once.
# Each running job keeps up to 4 x 18 GB (save_total_limit=3 + best, with optimizer state) until its final cleanup;
# 31 unthrottled jobs filled the 5 TB /scratch quota on 2026-10-13 and all failed.
#   bash whisper_launch.sh LANES ACCOUNT_LIST "ARM:SEED ARM:SEED ..."
#   e.g. bash whisper_launch.sh 8 torch_pr_39_general,torch_pr_60_general "R:4 M0.5:4 A0.5:4 D0.5:4"
# Env passes through (--export=ALL): LABEL_PREFIX, RUN_TAG (whisper_aug.sbatch), TIME; PART picks the partition.
# Lane i uses ACCOUNT_LIST[i % n]; jobs are dealt round-robin over lanes in the order given.
set -euo pipefail
LANES=$1; IFS=, read -ra ACCTS <<< "$2"; read -ra JOBS <<< "$3"
CODE=/scratch/pa2753/asr_aug/code
PART=${PART-l40s_public}   # PART= (empty) lets the account pick any GPU partition, including B200 (sm_100), which the
                           # whisper_asr torch build cannot run ("no kernel image"); list partitions explicitly instead
declare -a LAST
for k in "${!JOBS[@]}"; do
  ARM=${JOBS[$k]%%:*}; SEED=${JOBS[$k]##*:}
  lane=$((k % LANES)); acct=${ACCTS[$((lane % ${#ACCTS[@]}))]}
  dep=${LAST[$lane]:+--dependency=afterany:${LAST[$lane]}}
  id=$(sbatch --parsable --account="$acct" ${PART:+-p $PART} --time=${TIME:-01:15:00} --job-name="aug_w_${ARM}${RUN_TAG:+_$RUN_TAG}_s${SEED}" $dep \
       --export=ALL,ARM="$ARM",SEED="$SEED" "$CODE/whisper_aug.sbatch")
  LAST[$lane]=$id
  echo "whisper $ARM${RUN_TAG:+_$RUN_TAG} s$SEED $id $acct lane$lane label_prefix=${LABEL_PREFIX:-legacy}" | tee -a /scratch/pa2753/asr_aug/jobs.txt
done
