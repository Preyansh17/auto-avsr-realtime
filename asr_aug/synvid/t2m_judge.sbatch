#!/bin/bash
#SBATCH --job-name=sv_t2m_judge
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/pa2753/synvid/logs/%x_%j.out
#SBATCH --open-mode=append
# A11 step 1: zero-shot USR 2.0 Huge judge (step 1.2 settings) on set a, set b and the 24 real held-out crops.
# sbatch --account=<A> --comment="preemption=yes;requeue=true" --constraint="l40s|a100|h100|h200" t2m_judge.sbatch
set -eo pipefail
SV=/scratch/pa2753/synvid; C=$SV/code; T=$SV/t2m
$SV/envs/usr2/bin/python - <<PY
import json
T = "$T"; SV = "$SV"
for name in ("a", "b"):
    with open(f"{T}/judge_{name}.tsv", "w") as g:
        for l in open(f"{T}/{name}/rows.jsonl"):
            r = json.loads(l)
            if r.get("path"):
                g.write(r["path"] + "\t" + r["text"] + "\n")
tx = {}
for l in open("/scratch/th3482/Dysarthric_ASR/data/patients/splits/LATEST_combined/roles/s3_train.jsonl"):
    if l.strip():
        x = json.loads(l); tx[x.get("take_id", x["video_path"].split("/")[-1][:-4])] = x["transcript"]
with open(f"{T}/judge_a_real.tsv", "w") as g:
    for t in open(f"{T}/heldout.txt").read().split():
        g.write(f"{SV}/p1/crop240/crops/{t}.mp4\t{tx[t]}\n")
PY
cd $C/usr2
for n in a b a_real; do
  $SV/envs/usr2/bin/python $C/judge.py --rows $T/judge_$n.tsv --ckpt $SV/models/usr2/vsr_huge_hr.pth --backbone huge \
    --out $T/judge_$n.jsonl
  echo "$n judged: $(wc -l < $T/judge_$n.jsonl)"
done
