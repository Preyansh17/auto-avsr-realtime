#!/bin/bash
#SBATCH --job-name=sv_p1a
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/pa2753/synvid/logs/%x_%j.out
#SBATCH --open-mode=append
# Phase 1.3 sanity check (zero-shot Large + Huge on s3ov_test) and 1.4 first check (rebuild 20 s3_train crops,
# judge Huge on original vs rebuilt). Resumable: judge/crop skip finished clips.
# sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint="l40s|a100|h100|h200" p1_gpu_a.sbatch
set -eo pipefail
SV=/scratch/pa2753/synvid; P=$SV/envs/usr2/bin/python; C=$SV/code; R=/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles
O=$SV/p1; mkdir -p $O; export PREP_ROOT=$C/repo_prep
nvidia-smi --query-gpu=name,memory.total --format=csv
(while true; do nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; sleep 60; done) &
cd $C/usr2
# 1.3: both judges on s3ov_test, in parallel with the crop rebuild
$P $C/judge.py --rows $R/s3ov_test.jsonl --ckpt $SV/models/usr2/vsr_large_hr.pth --backbone large --out $O/judge_large_s3ov_test.jsonl &
J1=$!
$P $C/judge.py --rows $R/s3ov_test.jsonl --ckpt $SV/models/usr2/vsr_huge_hr.pth --backbone huge --out $O/judge_huge_s3ov_test.jsonl &
J2=$!
# 1.5 driving clip: healthy-voice Higgs (no patient voice, no adapter), in parallel
(PYTHONPATH=/scratch/pa2753/higgs_site HF_HOME=/scratch/pa2753/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /scratch/pa2753/envs/nemo_asr/bin/python $C/smoke_drive.py $SV/smoke > $O/smoke_drive.log 2>&1 || echo "drive clip FAILED, see $O/smoke_drive.log") &
J5=$!
# 1.4: rebuild 20 crops
$P $C/crop_pipeline.py --rows $R/s3_train.jsonl --take-ids $C/crop20_ids.txt --out-dir $O/rebuild20
wait $J1 $J2
# judge Huge on the same 20 takes: original crops and rebuilt crops
$P - <<PY
import json
rows={json.loads(l)["take_id"]:json.loads(l) for l in open("$R/s3_train.jsonl")}
ids=[l.strip() for l in open("$C/crop20_ids.txt") if l.strip()]
open("$O/crop20_orig.tsv","w").write("".join(f'{rows[t]["video_path"]}\t{rows[t]["transcript"]}\n' for t in ids))
open("$O/crop20_rebuilt.tsv","w").write("".join(f'$O/rebuild20/crops/{t}.mp4\t{rows[t]["transcript"]}\n' for t in ids))
PY
$P $C/judge.py --rows $O/crop20_orig.tsv --ckpt $SV/models/usr2/vsr_huge_hr.pth --backbone huge --out $O/judge_huge_crop20_orig.jsonl &
J3=$!
$P $C/judge.py --rows $O/crop20_rebuilt.tsv --ckpt $SV/models/usr2/vsr_huge_hr.pth --backbone huge --out $O/judge_huge_crop20_rebuilt.jsonl &
J4=$!
wait $J3 $J4
$P $C/compare_crops.py --rows $R/s3_train.jsonl --rebuilt-dir $O/rebuild20/crops --judge-orig $O/judge_huge_crop20_orig.jsonl \
   --judge-rebuilt $O/judge_huge_crop20_rebuilt.jsonl --out $O/compare20.json
for m in large huge; do $P -c "
import json; r=[json.loads(l) for l in open('$O/judge_${m}_s3ov_test.jsonl')]
print('$m s3ov_test n=%d WER=%.2f%%'%(len(r),100*sum(x['errors'] for x in r)/sum(x['ref_words'] for x in r)))"; done
wait $J5; cat $SV/smoke/drive_smoke.json 2>/dev/null
echo DONE
