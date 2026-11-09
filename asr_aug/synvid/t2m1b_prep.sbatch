#!/bin/bash
#SBATCH --job-name=sv_t2m1b_prep
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/pa2753/synvid/logs/%x_%j.out
#SBATCH --open-mode=append
# A12 step 1b prep: s3ov_train (all clips; the build holds out 26) mouth crops with crop_pipeline.py (step 1.4 rebuild),
# video CTC word positions with align_words.py video (same judge and method as A9), lip-activity span with the frozen
# step 1.4 threshold. Resumable. Outputs under t2m/step1b/ov/.
# sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint="a100|h100|h200" t2m1b_prep.sbatch
set -eo pipefail
SV=/scratch/pa2753/synvid; C=$SV/code; P=$SV/envs/usr2/bin/python; O=$SV/t2m/step1b/ov; mkdir -p $O
R=/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles
export PREP_ROOT=$C/repo_prep
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for i in 0 1; do
  $P $C/crop_pipeline.py --rows $R/s3ov_train.jsonl --out-dir $O/crop --shard $i --nshard 2 > $O/crop_$i.log 2>&1 &
done
(cd $C/usr2 && $P $C/align_words.py video --rows $R/s3ov_train.jsonl --ckpt $SV/models/usr2/vsr_huge_hr.pth \
   --backbone huge --out $O/video_align_s3ov_train.jsonl) > $O/video_align.log 2>&1 &
wait
echo "crops: $(ls $O/crop/crops/*.mp4 | wc -l); aligned: $(grep -c words $O/video_align_s3ov_train.jsonl)"
$P $C/lip_activity.py --lm-dir $O/crop/landmarks --rows $R/s3ov_train.jsonl --out $O/lip_s3ov_train.json \
  --thr-file $SV/p1/thr.json | head -20
