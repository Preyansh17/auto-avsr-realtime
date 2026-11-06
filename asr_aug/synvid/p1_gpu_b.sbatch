#!/bin/bash
#SBATCH --job-name=sv_p1b
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96GB
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/pa2753/synvid/logs/%x_%j.out
#SBATCH --open-mode=append
# Phase 1.3 follow-up and 1.4 full run, after the sanity check and the 20-clip crop match pass.
#  - judge (JUDGE=huge|large, the one 1.3 chose) on all s3_train (240) and s3ov_train (262), 3 shards in parallel
#  - crop all 240 s3_train clips with crop_pipeline.py -> p1/crop240; lip_activity.py --fit-thr -> p1/thr.json and
#    p1/lip_activity_s3_train.json (lead-in, tail, active span, active sec/word)
# Resumable. sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint="a100|h100|h200" \
#   --export=ALL,JUDGE=huge p1_gpu_b.sbatch
set -o pipefail
SV=/scratch/pa2753/synvid; C=$SV/code; O=$SV/p1; P=$SV/envs/usr2/bin/python; JUDGE=${JUDGE:-huge}
R=/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles
export PREP_ROOT=$C/repo_prep
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# one role at a time, 3 judge shards (6 judges + 2 crop workers at once ran out of memory on an 80 GB A100)
for i in 0 1; do
  $P $C/crop_pipeline.py --rows $R/s3_train.jsonl --out-dir $O/crop240 --shard $i --nshard 2 > $O/crop240_$i.log 2>&1
done
for role in s3_train s3ov_train; do
  PIDS=""
  for i in 0 1 2; do
    (cd $C/usr2 && $P $C/judge.py --rows $R/$role.jsonl --ckpt $SV/models/usr2/vsr_${JUDGE}_hr.pth --backbone $JUDGE \
       --out $O/judge_${JUDGE}_${role}.s$i.jsonl --shard $i --nshard 3) >> $O/judge_${role}_$i.log 2>&1 &
    PIDS="$PIDS $!"
  done
  for p in $PIDS; do wait $p || echo "proc $p exit $?"; done
done
for role in s3_train s3ov_train; do cat $O/judge_${JUDGE}_${role}.s?.jsonl > $O/judge_${JUDGE}_${role}.jsonl; done
$P $C/lip_activity.py --lm-dir $O/crop240/landmarks --rows $R/s3_train.jsonl --out $O/lip_activity_s3_train.json \
  --fit-thr --thr-file $O/thr.json
$P - <<EOF
import json, numpy as np
for role in ("s3_train", "s3ov_train"):
    r = [json.loads(l) for l in open("$O/judge_${JUDGE}_" + role + ".jsonl")]
    w = np.array([x["wer"] for x in r]); e = sum(x["errors"] for x in r); n = sum(x["ref_words"] for x in r)
    print(role, "n=%d pooled=%.4f mean=%.4f mean_cap1=%.4f p95=%.4f wer0=%.3f" % (len(r), e / n, w.mean(),
          np.minimum(w, 1).mean(), np.percentile(w, 95), (w == 0).mean()))
c = [json.loads(l) for f in ("$O/crop240/crop_log_0.jsonl", "$O/crop240/crop_log_1.jsonl") for l in open(f)]
print("crop240: n=%d ok=%d no_face_clips=%d" % (len(c), sum(x["ok"] for x in c), sum(x["no_face_frames"] > 0 for x in c)))
EOF
