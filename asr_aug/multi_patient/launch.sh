#!/bin/bash
# bash launch.sh LANES "ACCT1,ACCT2" PART TIME "SET:LP:SEED ..."   -- lane-chained to bound checkpoint disk (see asr_aug/whisper_launch.sh)
set -euo pipefail
LANES=$1; IFS=, read -ra A <<< "$2"; PART=$3; T=$4; read -ra J <<< "$5"
declare -a LAST
for k in "${!J[@]}"; do
  IFS=: read SET LP SEED <<< "${J[$k]}"
  lane=$((k % LANES)); acct=${A[$((lane % ${#A[@]}))]}
  dep=${LAST[$lane]:+--dependency=afterany:${LAST[$lane]}}
  id=$(sbatch --parsable --account=$acct -p $PART --time=$T --job-name=mp_${SET}_${LP:0:2}_s$SEED $dep \
     --export=ALL,SET=$SET,SEED=$SEED,LABEL_PREFIX=$LP /scratch/pa2753/multi_patient/code/whisper_patient.sbatch)
  LAST[$lane]=$id
  echo "mp $SET $LP s$SEED $id $acct $PART lane$lane" | tee -a /scratch/pa2753/multi_patient/jobs.txt
done
